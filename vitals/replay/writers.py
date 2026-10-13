"""The per-episode files a replay writes."""

from __future__ import annotations

import json
from pathlib import Path

from vitals.analysis.phases import reasoning_steps
from vitals.core.io import atomic_write_parquet, atomic_write_text


def write_attn_sparse_turns(ep, rows_by_turn: dict):
    """Per-forced-token sparse attention for several turns in one
    read-modify-write of attn_sparse.parquet: one row per forced token with the
    top-K (position, weight) pairs of its attention row over the PROMPT
    (``pos``/``w``, positions on the attn_tokens.parquet idx axis), the full
    prompt-side and generated-side mass totals, and the top-K pairs over the
    GENERATED positions (``gpos``/``gw``; gpos = generated-step index, the
    tokens-*.parquet ``step`` axis, so a reasoning token's own read is one
    join away). Each listed turn replaces its rows. Older stores lack gpos/gw."""
    import pandas as pd

    if not rows_by_turn:
        return
    frames = []
    for turn, rows in sorted(rows_by_turn.items()):
        frames.append(pd.DataFrame({
            "turn": [turn] * len(rows),
            "step": [r[0] for r in rows],
            "pos": [r[1] for r in rows],
            "w": [r[2] for r in rows],
            "prompt_mass": [r[3] for r in rows],
            "gen_mass": [r[4] for r in rows],
            "gpos": [r[5] for r in rows],
            "gw": [r[6] for r in rows],
        }))
    path = ep / "attn_sparse.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        frames.insert(0, old[~old.turn.isin(list(rows_by_turn))])
    atomic_write_parquet(pd.concat(frames, ignore_index=True), path)


def write_reasoning_steps(backend, ep: Path, gen) -> None:
    """Record, per turn, how many leading generated steps are reasoning (phases.reasoning_steps:
    Qwen `</think>` or a Harmony `to=self` segment), so phase analyses need no tokenizer."""
    piece = getattr(backend, "_piece", None)
    if piece is None or gen is None or not len(gen):
        return

    lines = []
    for turn, g in gen.groupby("turn"):
        ids = g.sort_values("step").token_id.astype(int).tolist()
        pieces = [piece(t).decode("utf-8", errors="replace") for t in ids]
        lines.append(json.dumps({"turn": int(turn), "n_reasoning": reasoning_steps(pieces), "n_gen": len(ids)}))
    atomic_write_text(ep / "reasoning_steps.jsonl", "\n".join(lines) + "\n")


def jsonl_turns(path: Path) -> set[int]:
    """Turn indices recorded in a jsonl file; malformed lines are skipped."""
    turns: set[int] = set()
    for line in path.read_text().splitlines():
        try:
            turns.add(int(json.loads(line)["turn"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return turns


def write_back(ep, parts, gen, shares_by_turn, lstats_by_turn, *, sidecar, force):
    """Write replayed channels: sidecar jsonl, or parquet columns in place
    (missing-only unless force)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if sidecar:
        n = 0
        if shares_by_turn:
            with (ep / "replay_attn_share.jsonl").open("w") as f:
                for t, shares in sorted(shares_by_turn.items()):
                    f.write(json.dumps({"turn": t, "shares": [
                        [round(x, 6) for x in s] for s in shares]}) + "\n")
            n += sum(len(s) for s in shares_by_turn.values())
        if lstats_by_turn:
            with (ep / "replay_logits.jsonl").open("w") as f:
                for t, ls in sorted(lstats_by_turn.items()):
                    f.write(json.dumps({"turn": t, "stats": ls}) + "\n")
            n += sum(len(x) for x in lstats_by_turn.values())
        return n

    if not shares_by_turn and not lstats_by_turn:
        return 0
    # step order within a turn indexes the per-token lists
    ranks: dict[int, dict[int, int]] = {}
    for t in set(shares_by_turn) | set(lstats_by_turn):
        ranks[t] = {int(s): i for i, s in
                    enumerate(sorted(gen[gen.turn == t].step.astype(int)))}
    n = 0
    for p in parts:
        tab = pq.read_table(p)
        cols = {c: tab.column(c).to_pylist()
                for c in ("turn", "step", "region_id", "attn_share",
                          "ent", "m2", "surprisal", "top_ids", "top_lp")}
        changed = False
        for i in range(len(cols["turn"])):
            if cols["region_id"][i] != "gen":
                continue
            t = int(cols["turn"][i])
            idx = ranks.get(t, {}).get(int(cols["step"][i]))
            if idx is None:
                continue
            if t in shares_by_turn and idx < len(shares_by_turn[t]):
                cur = cols["attn_share"][i]
                if force or cur is None or not len(cur):
                    cols["attn_share"][i] = [float(x) for x in shares_by_turn[t][idx]]
                    changed = True
                    n += 1
            if t in lstats_by_turn and idx < len(lstats_by_turn[t]):
                if force or cols["top_ids"][i] is None:
                    st = lstats_by_turn[t][idx]
                    cols["ent"][i] = st["ent"]
                    cols["m2"][i] = st["m2"]
                    cols["surprisal"][i] = st["surprisal"]
                    cols["top_ids"][i] = st["top_ids"]
                    cols["top_lp"][i] = st["top_lp"]
                    changed = True
                    n += 1
        if changed:
            meta = tab.schema.metadata
            for name, typ in (("attn_share", pa.list_(pa.float32())),
                              ("ent", pa.float32()), ("m2", pa.float32()),
                              ("surprisal", pa.float32()),
                              ("top_ids", pa.list_(pa.int32())),
                              ("top_lp", pa.list_(pa.float32()))):
                ci = tab.schema.get_field_index(name)
                tab = tab.set_column(ci, tab.schema.field(ci),
                                     pa.array(cols[name], type=typ))
            atomic_write_parquet(tab.replace_schema_metadata(meta), p)    # the logged tokens: never half written
    return n


