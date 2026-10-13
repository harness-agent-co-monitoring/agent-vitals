"""Per-head weights: weighted head mean on the host path, uniform fallback."""
import json

import numpy as np
import pytest

from vitals.engine import head_weights as hw


@pytest.fixture()
def wfile(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"3": [1, 0, 0, 1], "7": [0, 0, 0, 0]}))
    hw._cache.update(path=None, w=None)
    return str(p)


def test_pool_weighted_and_fallbacks(wfile):
    a = np.stack([np.full((2, 5), i, dtype=np.float32) for i in range(4)])   # heads 0..3 constant rows
    assert np.allclose(hw.pool(a, "kq_soft_max-3", wfile), np.full((2, 5), 1.5))   # (0+3)/2
    assert np.allclose(hw.pool(a, "kq_soft_max-7", wfile), np.full((2, 5), 1.5))   # all-zero -> uniform mean of 0..3
    assert np.allclose(hw.pool(a, "kq_soft_max-11", wfile), np.full((2, 5), 1.5))  # absent layer -> uniform
    assert np.allclose(hw.pool(a[:3], "kq_soft_max-3", wfile), np.full((2, 5), 1.0))  # length mismatch -> uniform
    assert hw.for_node("kq_soft_max-3", 4, wfile).tolist() == [1, 0, 0, 1]
    assert hw.layer_of(3) == 3 and hw.layer_of("3") == 3 and hw.layer_of("kq_soft_max-3") == 3 and hw.layer_of("x") is None
    assert np.allclose(hw.pool(a, 3, wfile), np.full((2, 5), 1.5))                   # int key (harvest dict)
    assert hw.pool(a[0], "kq_soft_max-3", wfile) is not None                       # 2-D passthrough
    assert hw.layer_dropped("kq_soft_max-7", wfile) and not hw.layer_dropped("kq_soft_max-3", wfile)
    assert not hw.layer_dropped("kq_soft_max-11", wfile)                            # absent layer: kept, uniform


def test_layer_head_mean_drops_zero_layers(wfile, monkeypatch):
    monkeypatch.setenv("VITALS_HEAD_WEIGHTS", wfile)
    hw._cache.update(path=None, w=None)
    from vitals.replay.pooling import layer_head_mean
    a3 = np.stack([np.full((2, 5), i, dtype=np.float32) for i in range(4)])
    a7 = np.full((4, 2, 5), 100.0, dtype=np.float32)
    rows = layer_head_mean({"kq_soft_max-3": a3, "kq_soft_max-7": a7})
    assert np.allclose(rows, 1.5)          # layer 7 (all-zero weights) left the layer mean


def test_no_env_means_uniform(monkeypatch):
    monkeypatch.delenv("VITALS_HEAD_WEIGHTS", raising=False)
    hw._cache.update(path=None, w=None)
    a = np.random.default_rng(0).random((4, 2, 5)).astype(np.float32)
    assert np.allclose(hw.pool(a, "kq_soft_max-3"), a.mean(axis=0))
