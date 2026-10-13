"""Section ledger: per turn's decision, how every attention head read every section, plus the activations later
probes need, from one teacher-forced pass.

For each turn with an action onset (``vitals.explain.auto_decision``), ``ledger_episode`` records:
- per-head attention from the decision token to each section, and averaged over the reasoning before it
  (``vitals.probes.head_reading``, the full-attention layers);
- the head-mean attention each prompt position gets over that reasoning (``span_pos``), and from it every section's
  read peak and hottest line over the reasoning (``meta["reason_peaks"]``, ``vitals.monitor.measures.section_peak``);
- the residual at the decision token at a few layers (for residual baselines and probes);
- each section's mean residual at those layers (the section activation cache: a new section probe trains on it
  without another forward pass);
- with the model file given, each section's writes into the hidden state at those layers
  (``vitals.probes.section_writes``, attention layers, plus the DeltaNet layers with ``linear``): a probe trained
  later is split by section from the ledger alone (``vitals.monitor.measures.split_report``). This is a second
  teacher-forced pass per decision.

Each decision is one ``.npz`` under ``<out_dir>/<store key>/<episode>/<turn>.npz``; ``ledger.parquet`` beside the
episodes holds one row per decision and section with the head-mean readings, and ``_ledger.json`` what was captured
(model, layers, writes), so a later request the cache does not cover is recaptured instead of reused. Nothing is
written into the store. Layout: ``docs/reference/outputs.md``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from vitals.core.io import atomic_savez, atomic_write_json, atomic_write_parquet
from vitals.core.paths import store_cache

LEDGER_VERSION = 1
DEFAULT_LAYERS = (23, 31, 39, 47)          # Qwen3.6-27B (64 layers); other depths: ``default_layers``
DEPTH = tuple(L / 64 for L in DEFAULT_LAYERS)
META = "_ledger.json"


def ledger_dir(out_dir: str | Path | None, store: str | Path) -> Path:
    """The ledger directory of a store: ``<out_dir>/<store key>`` (``out_dir`` defaults to the ledger cache,
    ``vitals.core.paths``)."""
    return store_cache("ledger", store, out_dir)


def default_layers(node_map) -> list[int]:
    """Residual layers at 36%, 48%, 61% and 73% of the model's depth (23, 31, 39 and 47 of 64), each moved to the
    nearest full-attention layer on hybrid models."""
    n = node_map.n_layers
    cands = list(node_map.full_attn_layers) or list(range(n))
    return sorted({min(cands, key=lambda L, f=f: (abs(L - f * n), L)) for f in DEPTH})


def model_id(gguf_path: str | Path | None) -> dict | None:
    """What identifies the model file a ledger was captured with: its name and size."""
    if not gguf_path:
        return None
    p = Path(gguf_path)
    return {"name": p.name, "size": p.stat().st_size if p.exists() else None}


def covers(have: dict | None, want: dict) -> bool:
    """Whether a ledger captured as ``have`` (its ``_ledger.json``) serves a request ``want`` (model, layers,
    writes, linear). A ledger without a recorded model is taken to match."""
    if not have:
        return False
    if have.get("model") and want.get("model") and have["model"] != want["model"]:
        return False
    return (set(want.get("layers") or []) <= set(have.get("layers") or [])
            and (have.get("writes") or not want.get("writes"))
            and (have.get("linear") or not want.get("linear")))


def read_meta(root: str | Path) -> dict | None:
    """What a ledger directory holds (``_ledger.json``); for ledgers written before it existed, read from one
    decision file (the model is then unknown)."""
    root = Path(root)
    if (root / META).exists():
        return json.loads((root / META).read_text())
    for f in root.glob("*/*.npz"):
        try:
            z = np.load(f)
            return {"version": 0, "model": None, "layers": [int(x) for x in z["residual_layers"]],
                    "writes": "writes" in z.files, "linear": "writes_linear" in z.files}
        except (OSError, ValueError, KeyError):
            continue
    return None


def ledger_episode(backend, store: str | Path, episode: str, *, turns: list[int] | None = None,
                   residual_layers=None, out_dir: str | Path | None = None, force: bool = False,
                   gguf_path: str | None = None, linear: bool = False, log=print) -> list[dict]:
    """Capture the ledger for the named turns (default: every turn with an action onset) of one episode.
    ``residual_layers`` default to ``default_layers``. With ``gguf_path`` (the file the backend loaded) the
    per-section writes are stored too. Turns already captured are kept when the ledger covers the request (same
    model, these layers, writes and linear split as asked), else everything is recaptured."""
    import pandas as pd

    from vitals.core.store import read_messages, turn_inputs, turn_prompt_text
    from vitals.explain import auto_decision
    from vitals.monitor.measures import section_peak
    from vitals.probes.head_reading import HeadReading

    root = ledger_dir(out_dir, store)
    ep_dir = Path(store) / episode
    dest = root / episode
    dest.mkdir(parents=True, exist_ok=True)
    if turns is None:
        at = pd.read_parquet(ep_dir / "attn_tokens.parquet", columns=["turn"])
        turns = sorted(int(t) for t in at.turn.unique())
    n = backend.node_map.n_layers
    layers = [int(L) for L in (residual_layers or default_layers(backend.node_map)) if int(L) < n]
    want = {"version": LEDGER_VERSION, "model": model_id(gguf_path), "layers": layers, "writes": bool(gguf_path),
            "linear": bool(linear and gguf_path)}
    have = read_meta(root)
    if have is not None and not force and not covers(have, want):
        log(f"[ledger] {root} was captured with another model, layers or writes: recapturing")
        force = True
        for f in root.glob("*/*.npz"):
            f.unlink()
        (root / "ledger.parquet").unlink(missing_ok=True)
    if have is None or force:
        atomic_write_json(root / META, want)
    messages = read_messages(ep_dir)
    hr = HeadReading(backend, residual_layers=layers)
    sw = None
    if gguf_path:
        from vitals.probes.section_writes import section_writes

        sw = section_writes(backend, gguf_path, layers=layers, linear=linear)
    rows = []
    for t in turns:
        path = dest / f"{t}.npz"
        if path.exists() and not force:
            continue
        t0 = time.time()
        inp = turn_inputs(ep_dir, t)
        d = auto_decision(backend, inp)
        if d is None:
            continue
        j, pos, a, b = d
        L = len(inp["prompt"])
        tokens = inp["prompt"] + inp["output"][:j + 1]
        labels = inp["labels"][:L + j + 1]
        res = hr.measure(tokens, labels, span=(L - 1, pos + 1), decisions=[pos], positions=True)
        names = res["sections"]
        span_pos = res["span_pos"][:L].astype(np.float32)
        tt = pd.read_parquet(ep_dir / "attn_tokens.parquet", columns=["turn", "idx", "token_str"],
                             filters=[("turn", "==", t)]).sort_values("idx").token_str.tolist()
        peaks = section_peak(span_pos, labels[:L], tt, prompt_text=turn_prompt_text(ep_dir, t, messages)) if len(tt) == L else {}
        reason_peaks = {n: {"peak": round(v["peak"], 4), "hot_line": v["hot_line"], "positions": v["positions"]}
                        for n, v in peaks.items()}
        n_tok = np.array([sum(1 for x in labels[:pos + 1] if x == n) for n in names])
        extra = {}
        if sw is not None:
            q = sw.capture(tokens, labels, [pos])["positions"][pos]
            order = {n: i for i, n in enumerate(dict.fromkeys(labels[:pos + 1]))}
            idx = [order[n] for n in names]
            extra["writes"] = q["writes"][:, idx].astype(np.float16)                     # [layers, R, n_embd]
            if "writes_linear" in q:
                extra["writes_linear"] = q["writes_linear"][:, idx].astype(np.float16)
        atomic_savez(path, sections=np.array(names), heads=np.array(res["heads"]), tokens=n_tok, **extra,
                 span=res["span"].astype(np.float16), at=res["at"][pos].astype(np.float16), span_pos=span_pos,
                 residual_layers=np.array(layers), residual_at=res["residual_at"][pos].astype(np.float16),
                 section_residual=res["section_residual"].astype(np.float16),
                 meta=np.array(json.dumps({"episode": episode, "turn": t, "step": j, "position": pos, "a": a, "b": b,
                                           "a_text": backend.engine.detokenize([a]), "b_text": backend.engine.detokenize([b]),
                                           "prompt_tokens": L, "reason_peaks": reason_peaks,
                                           "seconds": round(time.time() - t0, 2)})))
        at_mean, span_mean = res["at"][pos].mean(0), res["span"].mean(0)
        rows += [{"episode": episode, "turn": t, "position": pos, "section": n, "tokens": int(n_tok[i]),
                  "read_at": float(at_mean[i]), "read_reasoning": float(span_mean[i]),
                  "reason_peak": (reason_peaks.get(n) or {}).get("peak")} for i, n in enumerate(names)]
    if rows:
        lp = root / "ledger.parquet"
        df = pd.DataFrame(rows)
        if lp.exists():
            old = pd.read_parquet(lp)
            keep = ~old.set_index(["episode", "turn"]).index.isin(df.set_index(["episode", "turn"]).index)
            df = pd.concat([old[keep], df], ignore_index=True)
        atomic_write_parquet(df, lp)
    return rows


def read_decision(path: str | Path, episode: str | None = None, turn: int | None = None) -> dict:
    """One ledger decision file as a dict of arrays, with ``sections`` as a list, ``meta`` parsed, and ``episode``
    and ``turn`` (from the file's place in the ledger unless given)."""
    path = Path(path)
    z = np.load(path)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["sections"] = list(d["sections"])
    d["meta"] = json.loads(str(z["meta"]))
    d["episode"] = episode if episode is not None else d["meta"].get("episode", path.parent.name)
    d["turn"] = turn if turn is not None else int(d["meta"].get("turn", path.stem))
    return d


def ledger_decisions(root: str | Path, episodes: list[str] | None = None, *, log=print) -> list[dict]:
    """Every decision under a ledger directory (``ledger_dir``), loaded (``read_decision``). Files that cannot be
    read (an interrupted capture) are skipped with a message."""
    out = []
    for ep in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if episodes and ep.name not in episodes:
            continue
        files = [f for f in ep.glob("*.npz") if f.stem.isdigit()]
        for f in sorted(files, key=lambda f: int(f.stem)):
            try:
                out.append(read_decision(f, ep.name, int(f.stem)))
            except (OSError, ValueError, KeyError) as e:
                log(f"[ledger] skipping unreadable {f}: {e}")
    return out


def load_decision(out_dir: str | Path | None, store: str | Path, episode: str, turn: int) -> dict | None:
    """One decision of a store's ledger by episode and turn (``read_decision``, heads as tuples), or None."""
    path = ledger_dir(out_dir, store) / episode / f"{turn}.npz"
    if not path.exists():
        return None
    d = read_decision(path, episode, turn)
    d["heads"] = [tuple(h) for h in d["heads"]]
    return d


def ledger_read_normals(root: str | Path, *, floor: float | None = None, min_ref: int = 5) -> dict | None:
    """Label-free read normals (``vitals.monitor.measures.read_normals``) over every decision of a ledger
    directory (``ledger_dir``), from its ``ledger.parquet``: what reading at the decision and over the reasoning
    is typical in this store. None when the ledger has no decisions yet."""
    import pandas as pd

    from vitals.monitor.measures import READ_FLOOR, read_normals, read_rows

    lp = Path(root) / "ledger.parquet"
    if not lp.exists():
        return None
    df = pd.read_parquet(lp, columns=["episode", "turn", "section", "tokens", "read_at", "read_reasoning"])
    per = [read_rows(g.section.tolist(), g.tokens.to_numpy(), g.read_at.to_numpy(), g.read_reasoning.to_numpy())
           for _, g in df.groupby(["episode", "turn"], sort=False)]
    return read_normals(per, floor=READ_FLOOR if floor is None else floor, min_ref=min_ref) if per else None
