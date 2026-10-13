import numpy as np
import pytest

from vitals.core import metrics as M


def test_entropy_uniform_is_log_v():
    v = 128
    assert M.entropy(np.zeros(v)) == pytest.approx(np.log(v), rel=1e-9)


def test_entropy_peaked_is_near_zero():
    logits = np.full(64, -100.0)
    logits[3] = 100.0
    assert M.entropy(logits) < 1e-6


def test_top2_margin_bounds():
    logits = np.zeros(16)
    assert M.top2_margin(logits) == pytest.approx(0.0, abs=1e-12)
    logits[0] = 50.0
    assert M.top2_margin(logits) == pytest.approx(1.0, abs=1e-6)


def test_kl_zero_for_identical():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=256)
    assert M.kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-12)


def test_kl_positive_and_asymmetric():
    rng = np.random.default_rng(1)
    p, q = rng.normal(size=64), rng.normal(size=64)
    assert M.kl_divergence(p, q) > 0
    assert M.kl_divergence(p, q) != pytest.approx(M.kl_divergence(q, p))


def test_js_symmetric_and_bounded():
    rng = np.random.default_rng(2)
    p, q = rng.normal(size=64), rng.normal(size=64) + 5
    js_pq, js_qp = M.js_divergence(p, q), M.js_divergence(q, p)
    assert js_pq == pytest.approx(js_qp, rel=1e-9)
    assert 0 <= js_pq <= np.log(2) + 1e-9


def test_entropy_from_top_logprobs_matches_full_when_k_covers_all():
    logits = np.array([2.0, 1.0, 0.0, -1.0])
    lp = M.log_softmax(logits)
    top = {str(i): float(lp[i]) for i in range(4)}
    assert M.entropy_from_top_logprobs(top) == pytest.approx(M.entropy(logits), abs=1e-6)


def test_accumulator_forking_flags_after_warmup():
    acc = M.MetricsAccumulator(warmup=16)
    v = 64
    # deterministic entropy spread: dominance grows with i, so entropy strictly falls
    for i in range(32):
        logits = np.zeros(v)
        logits[0] = 1.0 + 0.2 * i
        acc.update(logits)
    # clearly below the running 80th percentile → not a fork
    peaked = np.zeros(v)
    peaked[0] = 12.0
    assert acc.update(peaked).fork is False
    # uniform = maximal entropy → fork
    assert acc.update(np.zeros(v)).fork is True


def test_accumulator_js_chain_and_reset():
    acc = M.MetricsAccumulator()
    a = acc.update(np.arange(8.0))
    assert a.js == 0.0
    b = acc.update(np.arange(8.0)[::-1].copy())
    assert b.js > 0
    acc.reset_step_chain()
    c = acc.update(np.arange(8.0))
    assert c.js == 0.0


def test_accumulator_warmup_zero_first_step_safe():
    acc = M.MetricsAccumulator(warmup=0)
    sm = acc.update(np.arange(16.0))  # must not percentile an empty history
    assert sm.fork is False
    assert acc.update(np.zeros(16)).fork in (True, False)


def test_accumulator_kl_ref():
    acc = M.MetricsAccumulator()
    logits = np.arange(16.0)
    sm = acc.update(logits, ref_logits=logits)
    assert sm.kl_ref == pytest.approx(0.0, abs=1e-12)
    sm2 = acc.update(logits, ref_logits=logits[::-1].copy())
    assert sm2.kl_ref > 0
