"""Routing features on synthetic per-head readings."""

import numpy as np

from vitals.monitor.routing import (
    distraction_heads,
    is_anchor,
    role_similarity,
    routing_ratio,
    selected_reading,
)

SECS = ["system/policy", "user:1", "tool_result:get#1", "injection", "output"]


def test_anchor_and_ratio():
    assert [is_anchor(s) for s in SECS] == [True, True, False, False, False]
    at = np.array([[0.2, 0.2, 0.1, 0.0, 0.5], [0.05, 0.05, 0.1, 0.4, 0.4]])
    rho = routing_ratio(at, SECS, [3])
    assert rho[0] > 0.99 and abs(rho[1] - 0.2) < 1e-6
    assert np.allclose(selected_reading(at, [1]), at[1])


def test_role_similarity_and_heads():
    at = np.array([[0.2, 0.4, 0.0, 0.4], [0.3, 0.1, 0.0, 0.1], [0.1, 0.3, 0.3, 0.3]])
    tok = np.array([10, 4, 4, 4])
    assert role_similarity(at, tok, 3, 1) > 0.99 and role_similarity(at, tok, 2, 1) < 0.9
    clean = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    inj = np.array([[0.1, 0.1], [0.45, 0.45], [0.4, 0.4]])
    assert distraction_heads([(clean, [0, 1], inj, [0, 1])], k=1) == [0]
