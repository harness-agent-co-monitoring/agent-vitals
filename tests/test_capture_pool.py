"""Unit tests for attention pooling over region token spans (no GPU)."""

import numpy as np

from vitals.core.schema import Region, RegionKind
from vitals.engine.capture import pool_attention


def _region(rid, s, e):
    return Region(id=rid, kind=RegionKind.USER, role="user", token_start=s, token_end=e)


def test_pool_shares_sum_and_split():
    # 10 kv cells: regions [0,4) and [4,7); generated bucket [7,10)
    regions = [_region("sys_0", 0, 4), _region("obs_0", 4, 7)]
    vec = np.zeros(10)
    vec[:4] = 0.05          # 0.2 to region 1
    vec[4:7] = 0.2          # 0.6 to region 2
    vec[7:10] = 0.2 / 3     # 0.2 to generated
    attn = {3: np.tile(vec, (8, 1))}  # (heads, n_kv), identical heads
    shares = pool_attention(attn, regions, n_kv=10)
    assert np.allclose(shares, [0.2, 0.6, 0.2], atol=1e-6)  # f32 accumulation
    assert abs(sum(shares) - 1.0) < 1e-6


def test_pool_uses_last_query_row_and_averages_layers():
    regions = [_region("sys_0", 0, 2)]
    a = np.zeros((8, 3, 4))     # (heads, T_q, n_kv): prefill-shaped
    a[:, -1, :2] = 0.5          # last row: all mass in the region
    b = np.zeros((8, 4))        # step-shaped: half mass in region, half after
    b[:, :2] = 0.25
    b[:, 2:] = 0.25
    shares = pool_attention({3: a, 11: b}, regions, n_kv=4)
    assert np.allclose(shares, [0.75, 0.25], atol=1e-9)  # mean over layers


def test_pool_none_when_spans_missing_or_empty():
    r = Region(id="sys_0", kind=RegionKind.SYSTEM, role="system")  # no spans
    assert pool_attention({3: np.ones((8, 4)) / 4}, [r], 4) is None
    assert pool_attention({}, [_region("sys_0", 0, 2)], 4) is None
    assert pool_attention({3: np.ones((8, 4)) / 4}, [], 4) is None
