"""Instrument gates: checks that must pass before captured evidence counts.

`fa_equivalence` -- attention capture needs flash-attention OFF while
production serves FA ON; if the two paths disagree beyond kernel noise, every
attention-captured episode is a silently divergent render path. Greedy-generate
the reference stream under FA-on, teacher-force it under FA-off, compare
logprobs.

Measured scale on this stack: batched-vs-stepwise GEMM paths differ by ~1e-2
logprob; the default threshold 0.05 sits above kernel noise and far below any
behavioral difference (on Qwen3.6-27B: max 0.035, mean 0.002, greedy streams
identical for 200 tokens).
"""

from __future__ import annotations

import numpy as np


def fa_equivalence(
    gguf: str,
    prompt: str,
    *,
    n: int = 200,
    n_ctx: int = 16384,
    threshold: float = 0.05,
    engine_kwargs: dict | None = None,
) -> dict:
    """Two sequential engine loads (FA-on reference, FA-off comparison).
    Returns the report dict with a PASS/CHECK verdict."""
    from vitals.engine.binding import LlamaEngine

    kw = engine_kwargs or {}

    eng = LlamaEngine(gguf, n_ctx=n_ctx, **kw)
    eng.prefill(prompt)
    stream: list[int] = []
    lp_on: list[float] = []
    for _ in range(n):
        lg = eng.logits(-1).astype(np.float64)
        lg -= lg.max()
        lp = lg - np.log(np.exp(lg).sum())
        tok = int(np.argmax(lp))
        stream.append(tok)
        lp_on.append(float(lp[tok]))
        if eng.is_eog(tok):
            break
        eng.decode([tok])
    eng.close()

    eng = LlamaEngine(gguf, n_ctx=n_ctx, **kw)
    eng.reset_context(flash_attn=0)
    eng.prefill(prompt)
    lp_off: list[float] = []
    diverge_at = None
    gap_at_div = None
    for i, tok in enumerate(stream):
        lg = eng.logits(-1).astype(np.float64)
        lg -= lg.max()
        lp = lg - np.log(np.exp(lg).sum())
        lp_off.append(float(lp[tok]))
        am = int(np.argmax(lp))
        if am != tok and diverge_at is None:
            diverge_at = i
            top2 = np.partition(lp, -2)[-2:]
            gap_at_div = float(top2[0] - top2[1])  # negative gap = near-tie
        eng.decode([tok])
    eng.close()

    d = np.abs(np.asarray(lp_on) - np.asarray(lp_off))
    ok = bool(d.max() <= threshold and (diverge_at is None or (gap_at_div or 0) >= -threshold))
    return {
        "n_tokens": len(stream),
        "max_abs_dlogprob": float(d.max()),
        "mean_abs_dlogprob": float(d.mean()),
        "p99_abs_dlogprob": float(np.percentile(d, 99)),
        "greedy_diverge_at": diverge_at,
        "logprob_gap_at_divergence": gap_at_div,
        "threshold": threshold,
        "verdict": "PASS" if ok else "CHECK",
    }
