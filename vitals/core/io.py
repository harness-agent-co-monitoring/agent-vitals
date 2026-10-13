"""Whole-file writes that readers never see half done, and tolerant JSON-lines reads.

Every file vitals rewrites (token parts during replay, caches, alarm and ledger files) goes through these
helpers: the content is written beside the target and renamed over it, so a crash or a concurrent reader (the
dashboard, ``vitals pipeline --watch``) sees the old file or the new one, never a truncated one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _tmp(path: Path) -> Path:
    # the suffix keeps the temporary out of every glob vitals uses (tokens-*.parquet, *.npz, *.jsonl)
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Replace ``path`` with ``data``: written beside it, then renamed over it."""
    path = Path(path)
    tmp = _tmp(path)
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    """``atomic_write_bytes`` of UTF-8 text."""
    atomic_write_bytes(path, text.encode())


def atomic_write_json(path: str | Path, obj: Any, **kw) -> None:
    """``atomic_write_text`` of ``json.dumps(obj, **kw)``."""
    atomic_write_text(path, json.dumps(obj, **kw))


def write_jsonl(path: str | Path, rows: Iterable[Any], **kw) -> None:
    """Replace ``path`` with one JSON object per line."""
    atomic_write_text(path, "".join(json.dumps(r, **kw) + "\n" for r in rows))


def read_jsonl(path: str | Path) -> list[Any]:
    """The JSON objects of a JSON-lines file; blank and malformed lines (a partial last line) are skipped. A
    missing file reads as empty."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def atomic_write_parquet(obj, path: str | Path, **kw) -> None:
    """Write a pandas DataFrame (``index=False`` unless given) or a pyarrow Table as parquet."""
    import pyarrow.parquet as pq

    path = Path(path)
    tmp = _tmp(path)
    if hasattr(obj, "to_parquet"):
        obj.to_parquet(tmp, **{"index": False, **kw})
    else:
        pq.write_table(obj, tmp, **kw)
    os.replace(tmp, path)


def atomic_savez(path: str | Path, **arrays) -> None:
    """``numpy.savez`` to ``path`` (which should end in ``.npz``)."""
    import numpy as np

    path = Path(path)
    tmp = _tmp(path)
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
