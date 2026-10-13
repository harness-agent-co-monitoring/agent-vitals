"""Pure-logprob metric math: no torch, no engine.

All entropies/divergences are in nats. Inputs are raw logits or logprobs as 1-D
numpy arrays over the vocabulary, or top-k ``{token: logprob}`` mappings (tail
treated as a single bucket, an approximation).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def entropy(logits: np.ndarray) -> float:
    """Shannon entropy of softmax(logits), in nats."""
    lp = log_softmax(logits)
    p = np.exp(lp)
    return float(-(p * lp).sum())


def top2_margin(logits: np.ndarray) -> float:
    """p(top1) - p(top2) of softmax(logits)."""
    lp = log_softmax(logits)
    if lp.size < 2:
        return 1.0
    top2 = np.partition(lp, -2)[-2:]
    return float(np.exp(top2[1]) - np.exp(top2[0]))


def kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(P || Q) between softmax distributions, in nats."""
    lp, lq = log_softmax(p_logits), log_softmax(q_logits)
    p = np.exp(lp)
    return float((p * (lp - lq)).sum())


def js_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """Jensen–Shannon divergence, in nats. Symmetric, bounded by ln 2."""
    lp, lq = log_softmax(p_logits), log_softmax(q_logits)
    p, q = np.exp(lp), np.exp(lq)
    m = 0.5 * (p + q)
    lm = np.log(m + _EPS)
    return float(0.5 * (p * (lp - lm)).sum() + 0.5 * (q * (lq - lm)).sum())


# ---------------------------------------------------------------------------
# Metrics from top-k logprobs (OpenAI-compatible `top_logprobs`).
# ---------------------------------------------------------------------------

def entropy_from_top_logprobs(top_logprobs: dict[str, float]) -> float:
    """Entropy estimate from a top-k logprob map, treating the tail as one bucket.

    This is an approximation: it lower-bounds the true entropy contribution of
    the tail (all tail mass on one outcome) and is exact as k → V.
    """
    lps = np.asarray(list(top_logprobs.values()), dtype=np.float64)
    p = np.exp(lps)
    total = p.sum()
    ent = float(-(p * lps).sum())
    tail = 1.0 - total
    if tail > _EPS:
        ent += float(-tail * np.log(tail))
    return ent


def nll_summary(logprobs: np.ndarray, min_k: float = 0.2) -> dict[str, float]:
    """Prefill-NLL summary: mean NLL plus the min-k% variant (mean over the WORST k fraction --
    the tokens the model finds least fluent, where scaffold awkwardness
    concentrates). Input: per-token logprobs from ``LlamaEngine.score_tokens``.
    """
    lp = np.asarray(logprobs, dtype=np.float64)
    if lp.size == 0:
        return {"mean_nll": float("nan"), f"min{int(min_k * 100)}_nll": float("nan"), "n": 0}
    nll = -lp
    k = max(1, int(round(min_k * nll.size)))
    worst = np.sort(nll)[-k:]
    return {"mean_nll": float(nll.mean()),
            f"min{int(min_k * 100)}_nll": float(worst.mean()),
            "n": int(nll.size)}


# ---------------------------------------------------------------------------
# Streaming accumulation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StepMetrics:
    ent: float
    m2: float
    js: float          # JS(prev step, this step); 0.0 on the first step
    fork: bool
    kl_ref: float | None = None


class MetricsAccumulator:
    """Per-episode streaming metrics with forking-token detection.

    Forking flag follows arXiv:2506.01939 / 2606.00206: a token is a fork when
    its entropy is above the running ``fork_percentile`` (default 80th)
    of this episode's entropy history. The flag stays False during ``warmup``
    steps while the percentile estimate is unstable.
    """

    def __init__(self, fork_percentile: float = 80.0, warmup: int = 64, history: int = 4096):
        self.fork_percentile = fork_percentile
        self.warmup = warmup
        self._ents: deque[float] = deque(maxlen=history)
        self._prev_logits: np.ndarray | None = None

    def update(self, logits: np.ndarray, ref_logits: np.ndarray | None = None) -> StepMetrics:
        logits = np.asarray(logits, dtype=np.float64)
        ent = entropy(logits)
        m2 = top2_margin(logits)
        js = 0.0 if self._prev_logits is None else js_divergence(self._prev_logits, logits)
        fork = False
        if len(self._ents) >= max(1, self.warmup):  # empty history has no percentile
            # strictly above: with heavy ties (e.g. many identical low-entropy steps)
            # `>=` would flag every step; strict `>` keeps "fork" meaning top ~(100-p)%.
            fork = ent > float(np.percentile(np.asarray(self._ents), self.fork_percentile))
        kl_ref = None if ref_logits is None else kl_divergence(logits, np.asarray(ref_logits))
        self._ents.append(ent)
        self._prev_logits = logits
        return StepMetrics(ent=ent, m2=m2, js=js, fork=fork, kl_ref=kl_ref)

    def reset_step_chain(self) -> None:
        """Break the step-JS chain (e.g. across an injected observation)."""
        self._prev_logits = None
