"""Where vitals keeps what it computes outside a trace store, and the key that ties it to one store.

Caches live under one root: ``$VITALS_CACHE_DIR``, else ``$XDG_CACHE_HOME/vitals``, else ``~/.cache/vitals``. Each
kind has its own directory there, and its own variable to move it:

=============  ======================  ==========================================================
kind           variable                written by
=============  ======================  ==========================================================
alarms         ``VITALS_ALARM_DIR``    ``vitals monitor``, ``vitals pipeline`` (flags, explanations)
ledger         ``VITALS_LEDGER_DIR``   ``vitals ledger``, ``vitals pipeline``
explanations   ``VITALS_EXPLAIN_DIR``  ``vitals explain``
readmaps       ``VITALS_READMAP_CACHE`` read maps of episodes whose directory is not writable
=============  ======================  ==========================================================

Inside a kind's directory, each store has a subdirectory named by ``store_key`` (a hash of the store's resolved
path), so moving a store leaves its cached results behind.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

CACHE_KINDS = {"alarms": "VITALS_ALARM_DIR", "ledger": "VITALS_LEDGER_DIR", "explanations": "VITALS_EXPLAIN_DIR",
               "readmaps": "VITALS_READMAP_CACHE"}


def cache_root() -> Path:
    """``$VITALS_CACHE_DIR``, else ``$XDG_CACHE_HOME/vitals``, else ``~/.cache/vitals``."""
    if root := os.environ.get("VITALS_CACHE_DIR"):
        return Path(root).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser() / "vitals"


def cache_dir(kind: str, override: str | Path | None = None) -> Path:
    """The directory of one cache kind: ``override`` when given, else the kind's variable, else under
    ``cache_root()``."""
    if kind not in CACHE_KINDS:
        raise ValueError(f"unknown cache kind {kind!r}; one of {sorted(CACHE_KINDS)}")
    chosen = override or os.environ.get(CACHE_KINDS[kind])
    return Path(chosen).expanduser() if chosen else cache_root() / kind


def store_key(store: str | Path) -> str:
    """12 hex characters naming a store's cached results: the SHA-1 of its resolved path."""
    return hashlib.sha1(str(Path(store).resolve()).encode()).hexdigest()[:12]


def store_cache(kind: str, store: str | Path, override: str | Path | None = None) -> Path:
    """``cache_dir(kind, override) / store_key(store)``."""
    return cache_dir(kind, override) / store_key(store)
