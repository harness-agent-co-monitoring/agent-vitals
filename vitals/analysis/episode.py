"""Episode-level aggregation over the parquet trace schema."""

from __future__ import annotations

import pyarrow as pa

from vitals.core.schema import TurnSummary


def turn_summaries(table: pa.Table) -> list[TurnSummary]:
    """Per-turn aggregates from a token table (any subset of columns beyond the
    core schema is ignored). Attention/commit fields stay None until their
    channels are wired."""
    if table.num_rows == 0:
        return []
    turns = table.column("turn").to_pylist()
    ents = table.column("ent").to_pylist()
    forks = table.column("fork").to_pylist()
    agg: dict[int, list[tuple[float, bool]]] = {}
    for t, e, f in zip(turns, ents, forks, strict=True):
        agg.setdefault(t, []).append((e, f))
    out = []
    for t in sorted(agg):
        rows = agg[t]
        out.append(
            TurnSummary(
                turn=t,
                n_tokens=len(rows),
                mean_ent=sum(e for e, _ in rows) / len(rows),
                fork_rate=sum(1 for _, f in rows if f) / len(rows),
            )
        )
    return out
