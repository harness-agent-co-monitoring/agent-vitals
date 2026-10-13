"""Per-head weights for the attention pooling.

The read map pools each captured attention tensor (heads, q, kv) over heads.
By default that is the uniform mean. A weight file — JSON {layer: [w_h ...]}
with one entry per full-attention layer — turns it into a weighted mean
(rows stay convex combinations of softmax rows, so lift semantics are
unchanged). Selected with the VITALS_HEAD_WEIGHTS env var (a path) or
``load(path)``; layers absent from the file, length mismatches and all-zero
vectors fall back to the uniform mean. Used by the on-device pooling in
engine/binding.py and by the host-side pooling in replay/pooling.py.
"""

from __future__ import annotations

import json
import os

import numpy as np

_cache: dict = {"path": None, "w": None}


def load(path: str | None = None) -> dict[int, np.ndarray] | None:
    path = path or os.environ.get("VITALS_HEAD_WEIGHTS")
    if not path:
        return None
    if _cache["path"] != path:
        with open(path) as f:
            d = json.load(f)
        _cache["path"] = path
        _cache["w"] = {int(k): np.asarray(v, dtype=np.float32) for k, v in d.items()}
    return _cache["w"]


def layer_of(name) -> int | None:
    """Layer index from a capture node name (`kq_soft_max-19`), a plain layer
    key (`19`, the harvest dict's keys) or an int."""
    if isinstance(name, (int, np.integer)):
        return int(name)
    try:
        return int(str(name).rsplit("-", 1)[-1])
    except ValueError:
        return None


def _layer_vector(name, path: str | None) -> np.ndarray | None:
    w = load(path)
    if not w:
        return None
    lay = layer_of(name)
    return w.get(lay) if lay is not None else None


def layer_dropped(name: str, path: str | None = None) -> bool:
    """True when the weight file lists this layer with an all-zero vector: the
    layer has no kept heads and should leave the layer mean too (the
    'content heads only' variant), rather than fall back to its uniform mean."""
    v = _layer_vector(name, path)
    return v is not None and float(v.sum()) <= 0


def for_node(name: str, heads: int, path: str | None = None) -> np.ndarray | None:
    """The weight vector for a capture node, or None for the uniform mean."""
    v = _layer_vector(name, path)
    if v is None or len(v) != heads or float(v.sum()) <= 0:
        return None
    return v


def pool(a: np.ndarray, name: str, path: str | None = None) -> np.ndarray:
    """(heads, q, kv) -> (q, kv): weighted mean over heads, uniform by default."""
    a = np.asarray(a)
    if a.ndim < 3:
        return a
    v = for_node(name, a.shape[0], path)
    if v is None:
        return a.mean(axis=0)
    return np.tensordot(v / v.sum(), a, axes=(0, 0))
