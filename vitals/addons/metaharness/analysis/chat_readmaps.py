"""Read-map rows for multi-turn chat episodes: one row per (episode, turn, section)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vitals.analysis.reading import token_uptake_turns
from vitals.analysis.readmap import PEAK_K, PEAK_MIN_TOK, prompt_line
from vitals.analysis.readmap import content_token as _content_token


def group_of(section: str) -> str:
    if section.startswith("policy:"):
        return "policy:" + section[7:].split("/")[0]
    if section.startswith("tool:"):
        return "tools"
    if section.startswith("doc:"):
        return "handbook"
    if section.startswith("sys:"):
        return "system"
    for p in ("user:", "obs:", "self:"):
        if section.startswith(p):
            return p[:-1]
    return section


def _peak(vec: np.ndarray, toks: list[str], st: int, en: int, L: int, prompt: str | None):
    n = en - st
    if n < PEAK_MIN_TOK:
        return None, None
    tl = vec[st:en] * L
    cand = np.flatnonzero([_content_token(t) for t in toks[st:en]])
    if len(cand) < PEAK_K:
        return None, None
    top = cand[np.argpartition(tl[cand], -PEAK_K)[-PEAK_K:]]
    line_id, cur = [], 0
    for t in toks[st:en]:
        line_id.append(cur)
        # real newlines, and the literal \n escapes inside JSON-carried documents
        cur += t.count("\n") + t.count("\\n")
    line_id = np.asarray(line_id)
    mass: dict[int, float] = {}
    for i in top:
        mass[line_id[i]] = mass.get(line_id[i], 0.0) + float(tl[i])
    best = max(mass, key=mass.get)
    joined = "".join(t for t, li in zip(toks[st:en], line_id, strict=True) if li == best).strip().replace("\n", " ")
    text = (prompt_line(prompt, joined) if prompt else None)
    if not text or len(text) > 3 * max(40, len(joined)):
        text = joined.replace("�", "")   # anchored line is a whole document: keep the token run
    return float(tl[top].mean()), text[:80]


def episode_table(ep: Path, actions: list[dict] | None = None) -> list[dict]:
    man = json.loads((ep / "manifest.json").read_text())["meta"]
    at = pd.read_parquet(ep / "attn_tokens.parquet")
    tk_all = pd.concat([pd.read_parquet(p) for p in sorted(ep.glob("tokens-*.parquet"))])
    prompts = {}
    for line in (ep / "messages.jsonl").read_text(errors="replace").splitlines():
        rec = json.loads(line)
        prompts[int(rec["turn"])] = "\n".join(str(m.get("content") or "") for m in rec["messages"])
    rows: dict[tuple, dict] = {}
    for t, g in at.groupby("turn"):
        t = int(t)
        g = g.sort_values("idx")
        vec = g.attn.to_numpy(dtype=float)
        tot = vec.sum() or 1e-9
        vec = vec / tot
        toks = g.token_str.tolist()
        L = len(g)
        rid = g.region_id.fillna("").to_numpy()
        secmap: dict[int, str] = {}
        idxs = g.idx.to_numpy()
        for s in pd.unique(rid[np.char.startswith(rid.astype(str), "span:")]):
            pos = np.flatnonzero(rid == s)
            st, en = int(pos.min()), int(pos.max()) + 1
            name = s[5:]
            share = float(vec[st:en].sum())
            pk, line = _peak(vec, toks, st, en, L, prompts.get(t))
            rows[(t, name)] = {"turn": t, "section": name, "group": group_of(name),
                               "span_tok": int(en - st), "prompt_tok": L, "share": share,
                               "lift": share / ((en - st) / L), "peak": pk, "hot_line": line,
                               "consult_ratio": np.nan, "consult_contrast": np.nan}
            for i in idxs[pos]:
                secmap[int(i)] = name
    # consult ratio / contrast from the dense per-token section shares (the
    # sparse top-K table misses cold sections at most steps and biases them down)
    for r in token_uptake_turns(ep, tk_all):
        key = (int(r["turn"]), r["section"])
        if key in rows:
            rows[key]["consult_ratio"] = np.nan if r["consult_ratio"] is None else float(r["consult_ratio"])
            rows[key]["consult_contrast"] = np.nan if r["consult_contrast"] is None else float(r["consult_contrast"])
            rows[key]["n_steps"] = int(r["n"])
    if actions is None:   # stores whose manifests carry the per-turn actions (HANDBOOK.md)
        actions = man.get("turn_actions")
    for r in rows.values():
        a = actions[r["turn"]] if actions and r["turn"] < len(actions) else {}
        r.update(episode_id=ep.name, task=man.get("example_idx"), reward=man.get("reward"),
                 passed=bool((man.get("reward") or 0) > 0), db_match=man.get("db_match"),
                 actions_unmet="|".join(man.get("actions_unmet") or []),
                 comm_unmet="|".join(str(c) for c in (man.get("comm_unmet") or [])),
                 n_turns=man.get("n_turns_replayed"), **a)
    return list(rows.values())
