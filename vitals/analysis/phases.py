"""The three parts of a reasoning model's read, per harness section.

A model that thinks before it answers reads in three places:

  harness <- thinking   the think block's attention to the prompt sections
  harness <- answer     the answer tokens' direct attention to the sections
  thinking <- answer    the answer tokens' attention to the model's own
                        reasoning

and the composition  harness <- thinking <- answer  is what a section
contributed to the answer *through* the reasoning. The first two are
per-phase versions of the read-map measures and come from any store with
attention; the third and the composition need the generated-side joint
(`gpos`/`gw` in attn_sparse.parquet), and report n/a without it.

Phase boundary: the number of leading generated steps that belong to the
reasoning, per turn. Replay records it in `reasoning_steps.jsonl` with
`reasoning_steps()` (Qwen's `</think>`, or a leading Harmony `to=self`
segment through its `<|eom|>` as Muse Glimmer writes it). Stores without that
file fall back to the `</think>` token (id = `<think>` id + 1 on Qwen
tokenizers, read off the prompt cue when available).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vitals.analysis.reading import consult_ratio
from vitals.core.store import read_received_attention, read_regions, read_token_rows, span_index

THINK_END_FALLBACK = 248069  # Qwen3.5/3.6 `</think>` (Qwen3: 151668); other vocabularies need attn_tokens.parquet
HARMONY_BODY, HARMONY_END, HARMONY_START = "<|message|>", "<|eom|>", "<|start|>"


def think_end_id(ep: Path) -> int:
    """`</think>` token id for this episode: the prompt cue's `<think>` + 1
    when attn_tokens.parquet carries the cue, else the fallback."""
    try:
        import pandas as pd
        a = pd.read_parquet(ep / "attn_tokens.parquet", columns=["token_id", "token_str"])
        hit = a[a.token_str == "<think>"]
        if len(hit):
            return int(hit.token_id.iloc[0]) + 1
    except (OSError, ValueError, KeyError):   # no attn_tokens.parquet, or one without token strings
        pass
    return THINK_END_FALLBACK


def reasoning_steps(pieces: list[str]) -> int:
    """Leading generated steps that are reasoning, from the decoded token pieces of one turn.

    Qwen style: everything through `</think>`. Harmony style (Muse Glimmer): a turn whose
    header (the text before its first `<|message|>`) names the recipient `self` is reasoning
    through the `<|eom|>` that closes that segment. Anything else: 0 (no reasoning block)."""
    for i, p in enumerate(pieces):
        if p == "</think>":
            return i + 1
    try:
        body = pieces.index(HARMONY_BODY)
    except ValueError:
        return 0
    header = "".join(pieces[:body]).replace(HARMONY_START, "").split()
    if not header or header[-1] != "to=self":
        return 0
    for i in range(body + 1, len(pieces)):
        if pieces[i] == HARMONY_END:
            return i + 1
        if pieces[i] == HARMONY_START:  # segment ended without <|eom|>
            return i
    return len(pieces)  # still reasoning when the turn stopped


def read_reasoning_steps(ep: Path) -> dict[int, int]:
    """turn -> reasoning steps, as recorded by replay (empty for older stores)."""
    path = Path(ep) / "reasoning_steps.jsonl"
    out: dict[int, int] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                out[int(rec["turn"])] = int(rec["n_reasoning"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return out


def think_steps(gen_turn, end_id: int) -> int:
    """Number of think steps in one turn's generated rows (step-sorted):
    the `</think>` token's step + 1, or 0 when the turn has no think block."""
    ids = gen_turn.token_id.to_numpy()
    hit = np.flatnonzero(ids == end_id)
    return int(hit[0]) + 1 if len(hit) else 0


def _load(ep: Path, tokens=None):
    ep = Path(ep)
    if tokens is None:
        tokens = read_token_rows(ep)
        if tokens is None:
            return None, None, None
    gen = tokens[tokens.region_id == "gen"]
    gen = gen[gen.attn_share.map(lambda s: s is not None and len(s) > 0)]
    return gen, read_regions(ep), read_received_attention(ep)


def section_phase_rows(ep_dir, tokens=None) -> list[dict]:
    """Per (turn, section): read lift and consult ratio of the think block
    and of the answer separately — harness<-thinking and harness<-answer."""
    ep = Path(ep_dir)
    gen, regions, attn = _load(ep, tokens)
    if gen is None or len(gen) == 0:
        return []
    end_id = think_end_id(ep)
    recorded = read_reasoning_steps(ep)
    out = []
    for t, g in gen.groupby("turn"):
        g = g.sort_values("step")
        regs = regions.get(int(t), [])
        spans = span_index(regs)
        L = len(attn.get(int(t), []))
        if not spans or not L:
            continue
        nt = recorded.get(int(t), think_steps(g, end_id))
        ents = g.ent.to_numpy()
        # attn_share = one share per region + a trailing generated-tokens bucket
        ctx = np.array([max(1.0 - float(s[-1]), 1e-9) if len(s) == len(regs) + 1 else 1.0 for s in g.attn_share])
        for name, (i, size) in spans.items():
            shares = np.array([float(s[i]) if i < len(s) else 0.0 for s in g.attn_share])
            lifts = shares / ctx / (size / L)
            th, an = slice(0, nt), slice(nt, None)
            ct, ca = consult_ratio(ents[th], shares[th]), consult_ratio(ents[an], shares[an])
            out.append({"turn": int(t), "section": name,
                        "n_think": nt, "n_answer": int(len(g) - nt),
                        "lift_think": float(lifts[th].mean()) if nt else None,
                        "lift_answer": float(lifts[an].mean()) if len(g) > nt else None,
                        "consult_think": None if ct is None else round(ct, 4),
                        "consult_answer": None if ca is None else round(ca, 4)})
    return out


def answer_flow(ep_dir, tokens=None, answer_weights: str = "uniform", min_tokens: int = 8) -> dict:
    """What reached the answer, per section — the full recursion.

    Every generated step t's attention row splits into a prompt part (its
    per-region shares, exact) and a generated part (top-K earlier steps
    `gpos`/`gw`, plus the exact total `gen_mass`). Resolving the generated
    part through those steps' own rows, recursively down to the prompt,
    gives each step's *traced* prompt share

        F_t(s) = share_t(s) + sum_{r<t} w(t->r) * F_r(s)

    (attention rollout with the prompt as the only source). Steps only attend
    backwards, so one sweep in step order computes it: O(T * K * regions).
    With complete rows F_t(prompt) = 1 for every t — all attention traces back
    to the prompt. Top-K truncation loses the uncaptured generated tail; two
    treatments are returned: `reach` renormalizes each step's captured
    generated weights to its exact `gen_mass` (the tail is assumed to
    distribute like the captured edges), `reach_captured` drops it, and
    `traced` = mean F_t(prompt) of the answer steps without renormalization
    (1 = nothing lost). Per section, all values are lifts (share / fair
    share), averaged over the answer steps (uniform, or weighted by the
    step's next-token entropy with answer_weights="entropy": the decision
    tokens count more than the format).

    Both digest measures exist on the traced shares: lift (all the `reach*`
    values) and the consult ratio — `consult_think_traced` /
    `consult_answer_traced` = entropy-weighted mean of F_t(s) over its plain
    mean within the phase (identically reach(entropy-weighted) / reach(uniform)),
    i.e. whether the section's part of the *resolved* reading rises when the
    model is unsure. `reach_think` is the same recursion averaged over the
    think steps, the like-for-like baseline for `reach` (raw think shares sum
    to the prompt mass, traced shares to ~1), so reach / reach_think says
    whether a section's share of the resolved reading grows or shrinks from
    deliberation to commitment. Also returned: `direct` (harness<-answer, the
    prompt-only term), `one_hop` (direct + one hop through earlier generated
    tokens, each resolved by its own prompt share only), `indirect`
    (one_hop - direct) and `total` (= reach).
    Stores without gpos/gw get `available` False and direct only."""
    import pandas as pd

    ep = Path(ep_dir)
    gen, regions, attn = _load(ep, tokens)
    if gen is None or len(gen) == 0:
        return {"available": False, "sections": {}}
    end_id = think_end_id(ep)
    recorded = read_reasoning_steps(ep)
    sp = ep / "attn_sparse.parquet"
    sparse = pd.read_parquet(sp) if sp.exists() else None
    joint = sparse is not None and "gpos" in sparse.columns
    sections: dict[str, dict] = {}
    diag = {"prompt": [], "gen": [], "reasoning": [], "answer": [], "traced": [], "captured": []}
    for t, g in gen.groupby("turn"):
        g = g.sort_values("step").reset_index(drop=True)
        spans = span_index(regions.get(int(t), []))
        L = len(attn.get(int(t), []))
        if not spans or not L:
            continue
        nt = recorded.get(int(t), think_steps(g, end_id))
        T = len(g)
        if T <= nt:
            continue
        R = max(len(s) for s in g.attn_share)
        S = np.zeros((T, R + 1))  # per-region prompt shares; last column = whole prompt
        for k, sh in enumerate(g.attn_share):
            S[k, :len(sh)] = np.asarray(sh, float)
        wts = np.ones(T)
        if answer_weights == "entropy":
            wts = np.maximum(g.ent.to_numpy(dtype=float), 1e-6)
        aw = wts[nt:] / wts[nt:].sum()
        idx = {name: i for name, (i, _) in spans.items()}
        fair = {name: size / L for name, (_, size) in spans.items()}
        if joint:
            st = sparse[sparse.turn == int(t)].set_index("step")
            pm = st.prompt_mass.reindex(range(T)).to_numpy(dtype=float)
            gm = st.gen_mass.reindex(range(T)).to_numpy(dtype=float)
            S[:, R] = np.nan_to_num(pm, nan=0.0)
            F_ren, F_cap = S.copy(), S.copy()  # traced shares, renormalized / captured only
            H1 = S.copy()  # one hop only
            for k in range(T):
                if k not in st.index:
                    continue
                gp, gw = np.asarray(st.gpos.loc[k], int), np.asarray(st.gw.loc[k], float)
                if len(gp) == 0:
                    continue
                gp_ok = gp < k
                gp, gw = gp[gp_ok], gw[gp_ok]
                cap = gw.sum()
                F_cap[k] += gw @ F_cap[gp]
                H1[k] += gw @ S[gp]
                if cap > 0 and gm[k] > 0:
                    F_ren[k] += (gw * (gm[k] / cap)) @ F_ren[gp]
                if k >= nt:
                    diag["captured"].append(float(cap / gm[k]) if gm[k] > 0 else 1.0)
                    diag["reasoning"].append(float(gw[gp < nt].sum()))
                    diag["answer"].append(float(gw[gp >= nt].sum()))
            ans = slice(nt, T)
            diag["prompt"].extend(S[ans, R].tolist())
            diag["gen"].extend(np.nan_to_num(gm[ans], nan=0.0).tolist())
            diag["traced"].extend(F_cap[ans, R].tolist())
            ents = g.ent.to_numpy(dtype=float)
            for name, i in idx.items():
                d = sections.setdefault(name, {"direct": [], "one_hop": [], "reach": [], "reach_captured": [],
                                               "reach_think": [], "consult_think_traced": [], "consult_answer_traced": []})
                d["direct"].append(float(aw @ S[ans, i]) / fair[name])
                d["one_hop"].append(float(aw @ H1[ans, i]) / fair[name])
                d["reach"].append(float(aw @ F_ren[ans, i]) / fair[name])
                d["reach_captured"].append(float(aw @ F_cap[ans, i]) / fair[name])
                if nt:
                    d["reach_think"].append(float(F_ren[:nt, i].mean()) / fair[name])
                # consult ratio on the resolved reading, per phase; stored as log2 for the geometric mean
                ca = consult_ratio(ents[ans], F_ren[ans, i], min_tokens=min_tokens)
                ct = consult_ratio(ents[:nt], F_ren[:nt, i], min_tokens=min_tokens) if nt else None
                if ca is not None:
                    d["consult_answer_traced"].append(float(np.log2(ca)) if ca > 0 else np.nan)
                if ct is not None:
                    d["consult_think_traced"].append(float(np.log2(ct)) if ct > 0 else np.nan)
        else:
            for name, i in idx.items():
                d = sections.setdefault(name, {"direct": [], "one_hop": [], "reach": [], "reach_captured": [], "reach_think": []})
                d["direct"].append(float(aw @ S[nt:, i]) / fair[name])
            if sparse is not None:
                st = sparse[(sparse.turn == int(t)) & (sparse.step >= nt)]
                diag["prompt"].extend(st.prompt_mass.astype(float).tolist())
                diag["gen"].extend(st.gen_mass.astype(float).tolist())
    out = {"available": joint, "sections": {}}
    for name, d in sections.items():
        sec = {k: (float(np.mean(v)) if v else None) for k, v in d.items() if not k.startswith("consult_")}
        for k in ("consult_think_traced", "consult_answer_traced"):
            v = [x for x in d.get(k, []) if np.isfinite(x)]
            sec[k] = float(2 ** np.mean(v)) if v else None  # geometric mean over turns
        sec["indirect"] = sec["one_hop"] - sec["direct"] if sec["one_hop"] is not None else None
        sec["total"] = sec["reach"]
        out["sections"][name] = sec
    out["answer_row"] = {k: (float(np.mean(v)) if v else None) for k, v in diag.items()}
    return out
