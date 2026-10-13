"""Finding trace stores and Meta-Harness runs under the dashboard's roots."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from vitals.core.store import is_episode_dir

MAX_DEPTH = 4


@dataclass
class Store:
    id: str
    path: Path
    label: str
    run: str | None = None


@dataclass
class Run:
    id: str
    path: Path
    label: str
    jobs: Path | None = None
    stores: list[str] = field(default_factory=list)


def _id(path: Path) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()[:10]


def is_store(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(is_episode_dir(p) for p in path.iterdir())
    except OSError:
        return False


def _label(path: Path, root: Path) -> str:
    rel = path.relative_to(root.parent) if path.is_relative_to(root.parent) else path
    parts = [p for p in rel.parts if p != "store"]
    return "/".join(parts[-3:])


def discover(roots: list[str | Path]) -> tuple[dict[str, Store], dict[str, Run]]:
    """Every store and Meta-Harness run (a ``logs/<run>`` directory) under the roots."""
    stores: dict[str, Store] = {}
    runs: dict[str, Run] = {}

    def add_store(path: Path, root: Path, run: Run | None = None) -> None:
        sid = _id(path)
        if sid not in stores:
            stores[sid] = Store(sid, path, _label(path, root), run.id if run else None)
        if run and sid not in run.stores:
            run.stores.append(sid)

    def walk(path: Path, root: Path, depth: int) -> None:
        if (path / "evolution_summary.jsonl").exists():
            jobs = path.parents[1] / "jobs" / path.name
            run = Run(_id(path), path, _label(path, root), jobs if jobs.is_dir() else None)
            runs[run.id] = run
            for store in sorted((path / "vitals").glob("*/store")) if (path / "vitals").is_dir() else []:
                add_store(store, root, run)
            return
        if is_store(path):
            add_store(path, root)
            return
        if depth >= MAX_DEPTH:
            return
        try:
            children = sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            return
        for child in children:
            walk(child, root, depth + 1)

    for root in roots:
        root = Path(root).expanduser().resolve()
        walk(root, root, 0)
    return stores, runs
