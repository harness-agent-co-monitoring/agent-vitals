"""The linear-attention reading files of an episode: linear_reading.parquet and linear_steps.parquet."""

from __future__ import annotations

from pathlib import Path

from vitals.core.io import atomic_write_parquet

_FILE = "linear_reading.parquet"
_STEPS = "linear_steps.parquet"


def write_linear_turns(ep: str | Path, turns: dict[int, tuple[list[dict], list]]) -> None:
    """Replace the listed turns in linear_reading.parquet (one row per turn and section) and
    linear_steps.parquet (one row per forced token: ``share`` indexed by the section's ``slot``)."""
    import pandas as pd

    if not turns:
        return
    ep = Path(ep)
    rows = [r for t in sorted(turns) for r in turns[t][0]]
    steps = [{"turn": t, "step": s, "share": v} for t in sorted(turns) for s, v in turns[t][1]]
    for fname, new in ((_FILE, pd.DataFrame(rows)), (_STEPS, pd.DataFrame(steps))):
        path = ep / fname
        frames = [new]
        if path.exists():
            old = pd.read_parquet(path)
            frames.insert(0, old[~old.turn.isin(list(turns))])
        atomic_write_parquet(pd.concat(frames, ignore_index=True), path)


def read_linear(ep: str | Path):
    """linear_reading.parquet as a DataFrame, or None."""
    import pandas as pd

    path = Path(ep) / _FILE
    return pd.read_parquet(path) if path.exists() else None


def linear_turns_done(ep: str | Path) -> set[int]:
    df = read_linear(ep)
    return set() if df is None else {int(t) for t in df.turn.unique()}
