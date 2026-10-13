"""Explanations of decisions: which context sections pushed a decision, which way and how far.

A decision is a position where the model chose token a over token b (for example ``<tool_call>`` over starting
a message). ``explain`` stacks four layers:

1. **reading**: the head-mean attention each section gets at the decision and over the reasoning before it
   (the read map, ``vitals.probes.head_reading``);
2. **calibrated reading**: per-head reading weighted by head weights fit to ablation effects
   (``vitals.analysis.calibration``), an estimate of how much each section matters;
3. **ablation**: the exact effect Δm of replacing each of the top sections (ranked by layer 2, else layer 1)
   with filler, their joint effect, and a no-op check (``vitals.probes.ablation``);
4. **claims**: per ablated section, the probability of a over b with and without it, whether the decision flips,
   and, given a behavior map, the expected change in how often the model would choose b instead.

Reading and effect can disagree (a section read heavily that changes nothing); both are reported. Probabilities
are renormalized over a and b; with long forced reasoning they are close to 0 or 1, so the behavior map, fit on
sampled behavior, is the calibrated quantity for removal claims.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from vitals.core.io import read_jsonl, write_jsonl
from vitals.core.paths import store_cache
from vitals.core.sections import spans_from_labels
from vitals.core.store import turn_inputs  # noqa: F401  (re-exported: older code imports it from here)
from vitals.probes.ablation import SectionAblation, prob

EXCLUDE = ("output", "region:", "cue")      # sections never ablated: the turn's own output, template text, cues


def _features(hr: dict, decision: int, features: list) -> np.ndarray:
    """Feature matrix [F, R] in the order of a weights file: ("span"|"at", layer, head)."""
    idx = {tuple(h): i for i, h in enumerate(hr["heads"])}
    rows = []
    for kind, layer, head in features:
        src = hr["span"] if kind == "span" else hr["at"][decision]
        rows.append(src[idx[(int(layer), int(head))]])
    return np.stack(rows)


def explain(backend, tokens: list[int], labels: list[str], decision: tuple[int, int, int], *, span: tuple[int, int],
            weights: tuple[np.ndarray, list] | None = None, behavior: dict | None = None, top_k: int = 4,
            exclude: tuple[str, ...] = EXCLUDE, seed: int = 0) -> dict:
    """``labels``: a section per position of ``tokens`` (through the decision). ``decision``: (position, a, b),
    the distribution at ``position`` choosing a over b. ``span``: positions whose reading is averaged (the
    reasoning before the decision). ``weights``: (w, features) from ``calibration.load_weights``."""
    from vitals.probes.head_reading import HeadReading

    t0 = time.time()
    eng = backend.engine
    pos, a, b = (int(x) for x in decision)
    hr = HeadReading(backend).measure(tokens, labels, span=span, decisions=[pos])
    names = hr["sections"]
    read_at, read_span = hr["at"][pos].mean(0), hr["span"].mean(0)
    cal = weights[0] @ _features(hr, pos, weights[1]) if weights is not None else None
    spans = spans_from_labels(labels[:pos])
    candidates = [i for i, n in enumerate(names) if n in spans and not n.startswith(exclude)]
    rank_by = cal if cal is not None else read_span + read_at
    top = sorted(candidates, key=lambda i: -rank_by[i])[:top_k]
    abl = SectionAblation(backend, seed=seed)
    ablations = {names[i]: spans[names[i]] for i in top}
    if len(top) > 1:
        ablations["__joint__"] = [sp for i in top for sp in spans[names[i]]]
    noop = names[top[0]] if top else None
    reps = {"__noop__": [tokens[s:e] for s, e in spans[noop]]} if noop else {}
    if noop:
        ablations["__noop__"] = spans[noop]
    res = abl.effects(tokens, {"d": (pos, a, b)}, ablations, replacements=reps)
    m = res["base"]["d"]
    p0 = prob(m)
    rows = []
    for i in sorted(candidates, key=lambda i: -rank_by[i]):
        n = names[i]
        row = {"section": n, "tokens": sum(e - s for s, e in spans[n]),
               "excerpt": eng.detokenize(tokens[spans[n][0][0]:spans[n][0][1]])[:160].strip(),
               "reading_at_decision": float(read_at[i]), "reading_reasoning": float(read_span[i]),
               "calibrated": float(cal[i]) if cal is not None else None, "ablated": n in res["delta"]}
        if row["ablated"]:
            d = res["delta"][n]["d"]
            row.update(delta=d, p_without=prob(m - d), flips=(m > 0) != (m - d > 0))
            if behavior is not None:
                from vitals.analysis.calibration import apply_behavior_map

                row["behavior_change"] = apply_behavior_map(behavior, d)
        rows.append(row)
    ab = [r for r in rows if r["ablated"]]
    joint = res["delta"].get("__joint__", {}).get("d")
    a_text, b_text = eng.detokenize([a]), eng.detokenize([b])
    claims = []
    for r in sorted(ab, key=lambda r: -abs(r["delta"])):
        direction = "toward" if r["delta"] > 0 else "away from"
        text = (f"Section '{r['section']}' pushes {direction} {a_text!r} (Δ logit margin {r['delta']:+.2f}): without it, "
                f"P({a_text!r} over {b_text!r}) goes from {p0:.3f} to {r['p_without']:.3f}"
                + (", and the decision flips." if r["flips"] else "."))
        if "behavior_change" in r:
            text += f" Expected change in how often the model chooses {b_text!r} instead: {r['behavior_change']:+.2f}."
        claims.append(text)
    if ab and max(abs(r["delta"]) for r in ab) < 0.1 * max(1.0, abs(m)):
        claims.append(f"No ablated section moves the decision much: it rests on the model's own output or on "
                      f"sections ranked lower (margin {m:+.2f}).")
    return {"decision": {"position": pos, "a": a, "b": b, "a_text": a_text, "b_text": b_text, "margin": m, "p_a": p0},
            "sections": rows,
            "joint": {"sections": [names[i] for i in top], "delta": joint, "sum_of_parts": float(sum(r["delta"] for r in ab))},
            "checks": {"noop_delta": res["delta"].get("__noop__", {}).get("d"),
                       "additivity_gap": None if joint is None else float(joint - sum(r["delta"] for r in ab))},
            "claims": claims,
            "method": {"calibrated": weights is not None, "behavior_map": behavior is not None, "top_k": top_k,
                       "ablations": len(ablations), "seconds": round(time.time() - t0, 2)}}


def agent_view(ex: dict, top: int = 5) -> dict:
    """The compact form for agents: the decision, its causes by exact effect, what removing each would do, and
    the checks that say how far to trust it."""
    d = ex["decision"]
    causes = sorted((r for r in ex["sections"] if r["ablated"]), key=lambda r: -abs(r["delta"]))[:top]
    return {"turn": ex.get("turn"), "decision": f"{d['a_text']} over {d['b_text']}", "p": round(d["p_a"], 4),
            "causes": [{"section": r["section"], "direction": "toward" if r["delta"] > 0 else "against",
                        "effect": round(r["delta"], 3), "p_without": round(r["p_without"], 4), "flips": r["flips"],
                        **({"behavior_change": round(r["behavior_change"], 3)} if "behavior_change" in r else {}),
                        "text": r["excerpt"][:120]} for r in causes],
            "checks": {"noop_exact": ex["checks"]["noop_delta"] == 0.0,
                       "overlap": None if ex["joint"]["delta"] is None else round(ex["checks"]["additivity_gap"], 3),
                       "ranked_by": "calibrated reading" if ex["method"]["calibrated"] else "reading"}}


# -- trace stores -------------------------------------------------------------------------------------------
def auto_decision(backend, inp: dict) -> tuple[int, int, int, int] | None:
    """The action onset after the reasoning: the first ``<tool_call>`` token against its strongest stored
    competitor when the turn calls a tool, else the first non-blank token against ``<tool_call>``. Returns
    (step, position, a, b)."""
    eng = backend.engine
    out, L = inp["output"], len(inp["prompt"])
    think = eng.tokenize("</think>", add_special=False, parse_special=True)
    call = eng.tokenize("<tool_call>", add_special=False, parse_special=True)
    start = out.index(think[0]) + 1 if len(think) == 1 and think[0] in out else 0
    call_id = call[0] if len(call) == 1 else None
    j = next((k for k in range(start, len(out)) if out[k] == call_id), None) if call_id is not None else None
    if j is None:
        j = next((k for k in range(start, len(out)) if eng.detokenize([out[k]]).strip()), None)
    if j is None or L + j - 1 < 1:
        return None
    a = out[j]
    if call_id is not None and a != call_id:
        b = call_id
    else:
        b = next((t for t in inp["top_ids"][j] if t != a), None)
        if b is None:
            return None
    return j, L + j - 1, a, b


def cache_path(out_dir: str | Path | None, store: str | Path, episode: str) -> Path:
    """The explanation cache file of one episode: ``<out_dir>/<store key>/<episode>.jsonl`` (``out_dir`` defaults
    to the explanations cache, ``vitals.core.paths``)."""
    return store_cache("explanations", store, out_dir) / f"{episode}.jsonl"


def explain_store(backend, store: str | Path, episode: str, turns: list[int], *, out_dir: str | Path | None = None,
                  weights=None, behavior=None, top_k: int = 4) -> list[dict]:
    """Explain the action onset of each turn and add the results to the explanation cache (outside the store).
    Returns every cached explanation of the episode, by turn."""
    path = cache_path(out_dir, store, episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    have = {r["turn"]: r for r in read_jsonl(path)}
    for t in turns:
        inp = turn_inputs(Path(store) / episode, t)
        d = auto_decision(backend, inp)
        if d is None:
            continue
        j, pos, a, b = d
        L = len(inp["prompt"])
        r = explain(backend, inp["prompt"] + inp["output"][:j + 1], inp["labels"][:L + j + 1], (pos, a, b),
                    span=(L - 1, pos + 1), weights=weights, behavior=behavior, top_k=top_k)
        have[t] = {"store": str(Path(store).resolve()), "episode": episode, "turn": t, "step": j, **r}
    write_jsonl(path, (have[t] for t in sorted(have)))
    return [have[t] for t in sorted(have)]
