"""Head-weight calibration recovers an informative head from synthetic data and beats the head mean; the
behavior map is monotone and interpolates."""

import numpy as np

from vitals.analysis.calibration import (
    apply_behavior_map,
    cross_validate,
    fit_behavior_map,
    fit_head_weights,
    score,
)


def _units(n=60, heads=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        R = rng.integers(4, 12)
        effect = rng.exponential(1.0, R) * rng.choice([-1, 1], R)
        reading = rng.dirichlet(np.ones(R), heads)
        reading[3] = np.abs(effect) / np.abs(effect).sum() * 0.9 + 0.1 / R                 # one informative head
        out.append({"reading": reading, "effect": effect})
    return out


def test_head_weights_find_the_informative_head():
    units = _units()
    w = fit_head_weights(units, ridge=0.1)
    assert int(np.argmax(w)) == 3
    assert score(units, w)["spearman"] > score(units, None)["spearman"] + 0.2
    cv = cross_validate(units, np.arange(len(units)) // 3, k=5)
    assert cv["calibrated"]["spearman"] > cv["read_map"]["spearman"]


def test_behavior_map_is_monotone():
    bm = fit_behavior_map([-2, -1, 0, 1, 2, 3], [-0.6, -0.1, -0.2, 0.1, 0.5, 0.4])
    assert all(a <= b for a, b in zip(bm["y"], bm["y"][1:], strict=False))
    assert apply_behavior_map(bm, -10) == bm["y"][0] and apply_behavior_map(bm, 10) == bm["y"][-1]
