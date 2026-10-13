"""Routing features from the section ledger: which sections a decision draws on, read per attention head.

- ``routing_ratio``: per head, how much the decision reads the anchors (the user's turns and the system prompt)
  against a candidate section: rho_h = A_anchor / (A_anchor + A_candidate). Low rho means the candidate pulls the
  decision away from what the user and the harness asked.
- ``role_similarity``: cosine between the per-head reading profiles (attention per token of the section) of a
  candidate section and of a reference section such as the latest user turn: does the model read the candidate the
  way it reads the user?
- ``distraction_heads``: training-free head selection from paired decisions with and without a candidate section
  (the heads whose anchor reading drops most when it is present), as Attention Tracker selects heads.
- ``selected_reading``: section reading from selected heads, or the head mean (the read map).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def is_anchor(section: str) -> bool:
    head = section.split(":")[0].split("/")[0]
    return head in ("user", "system", "policy", "instructions") or section.startswith(("region:user", "region:sys"))


def indices(sections: list[str], pred) -> list[int]:
    return [i for i, s in enumerate(sections) if pred(s)]


def selected_reading(at: np.ndarray, heads: list[int] | None = None) -> np.ndarray:
    """[R] reading per section from the given head rows (default: all heads, the read map)."""
    a = np.asarray(at, dtype=np.float64)
    return (a if heads is None else a[heads]).mean(0)


def routing_ratio(at: np.ndarray, sections: list[str], candidate: list[int], anchors: list[int] | None = None) -> np.ndarray:
    """[H] rho per head for the candidate sections against the anchors."""
    a = np.asarray(at, dtype=np.float64)
    anc = anchors if anchors is not None else indices(sections, is_anchor)
    aa, ac = a[:, anc].sum(1), a[:, candidate].sum(1)
    return aa / (aa + ac + EPS)


def role_similarity(at: np.ndarray, tokens: np.ndarray, candidate: int, reference: int) -> float:
    a = np.asarray(at, dtype=np.float64) / np.maximum(np.asarray(tokens, dtype=np.float64), 1.0)[None]
    x, y = a[:, candidate], a[:, reference]
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + EPS))


def distraction_heads(pairs: list[tuple[np.ndarray, list[int], np.ndarray, list[int]]], k: int = 8) -> list[int]:
    """``pairs``: (at without the candidate, its anchor columns, at with the candidate, its anchor columns). Returns
    the k heads whose anchor reading drops most, on average, when the candidate is present."""
    drops = np.mean([np.asarray(c)[:, ca].sum(1) - np.asarray(x)[:, xa].sum(1) for c, ca, x, xa in pairs], axis=0)
    return [int(i) for i in np.argsort(-drops)[:k]]
