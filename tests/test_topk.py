"""Top-K logprob capture: loop plumbing and store roundtrip."""

import math

import numpy as np
from test_loop import FakeBackend, _script

from vitals.core.schema import CaptureSpec, TokenMetrics
from vitals.core.store import TraceStore
from vitals.loop.decode import run_turn


def test_loop_captures_topk_when_asked():
    script = _script(4, v=16, seed=1)
    res = run_turn(FakeBackend(script), [], capture=CaptureSpec(top_logprobs=5))
    for tm, logits in zip(res.metrics, script, strict=True):
        assert len(tm.top_ids) == 5 and len(tm.top_lp) == 5
        # descending, and top-1 is the argmax (greedy fake backend)
        assert tm.top_lp == sorted(tm.top_lp, reverse=True)
        assert tm.top_ids[0] == int(np.argmax(logits)) == tm.token_id
        # surprisal of the sampled (=top-1) token matches its logprob
        assert math.isclose(tm.surprisal, -tm.top_lp[0], rel_tol=1e-9)
        # logprobs are a valid sub-distribution
        assert sum(math.exp(x) for x in tm.top_lp) <= 1.0 + 1e-9


def test_loop_topk_off_by_default():
    res = run_turn(FakeBackend(_script(3)), [], capture=CaptureSpec())
    for tm in res.metrics:
        assert tm.top_ids is None and tm.top_lp is None and tm.surprisal is None


def test_store_roundtrip_topk(tmp_path):
    store = TraceStore(tmp_path, flush_every=2)
    for i in range(4):
        store.append(TokenMetrics(
            episode_id="ep0", turn=0, step=i, token_id=i, ent=0.1, m2=0.9,
            js=0.0, fork=False, surprisal=0.5 + i,
            top_ids=[i, i + 1], top_lp=[-0.1, -2.3]))
    t = store.read_tokens("ep0")
    assert t.column("surprisal").to_pylist() == [0.5, 1.5, 2.5, 3.5]
    assert t.column("top_ids").to_pylist()[0] == [0, 1]
    assert [round(x, 4) for x in t.column("top_lp").to_pylist()[3]] == [-0.1, -2.3]


def test_store_roundtrip_topk_absent(tmp_path):
    store = TraceStore(tmp_path)
    store.append(TokenMetrics(episode_id="e", turn=0, step=0, token_id=1,
                              ent=0.1, m2=0.9, js=0.0, fork=False))
    t = store.read_tokens("e")
    assert t.column("top_ids").to_pylist() == [None]
    assert t.column("surprisal").to_pylist() == [None]
