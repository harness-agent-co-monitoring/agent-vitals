"""Loop semantics tested against a deterministic fake backend — the engine
plugs into an already-tested loop, not vice versa."""

import numpy as np
import pytest

from vitals.backends.base import Capabilities, StepOutput
from vitals.core.metrics import StepMetrics
from vitals.core.schema import AttentionMode, AttentionSpec, CaptureSpec, CaptureUnsupported
from vitals.loop.decode import run_turn


class FakeBackend:
    """Emits a fixed logit script; argmax sampling; done after the script ends."""

    name = "fake"

    def __init__(self, script: list[np.ndarray], caps: Capabilities | None = None):
        self.script = script
        self.caps = caps or Capabilities(full_logits=True, branch_probes=True)
        self.sampled: list[int] = []

    def capabilities(self) -> Capabilities:
        return self.caps

    def prefill(self, messages, capture):
        return {"i": 0}

    def step(self, ctx) -> StepOutput:
        return StepOutput(logits=self.script[ctx["i"]])

    def sample(self, ctx, step, route=None) -> int:
        tok = int(np.argmax(step.logits))
        self.sampled.append(tok)
        return tok

    def advance(self, ctx, token):
        return {"i": ctx["i"] + 1}

    def done(self, ctx) -> bool:
        return ctx["i"] >= len(self.script)


def _script(n=6, v=32, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=v) for _ in range(n)]


def test_loop_emits_one_metric_per_token_in_order():
    be = FakeBackend(_script())
    res = run_turn(be, [], episode_id="ep0")
    assert len(res.tokens) == 6
    assert res.tokens == be.sampled
    assert [tm.step for tm in res.metrics] == list(range(6))
    assert all(tm.episode_id == "ep0" for tm in res.metrics)


def test_empty_capture_no_policy_is_inert():
    """Golden-rule shape: the loop must not change what the backend samples."""
    script = _script()
    baseline = [int(np.argmax(x)) for x in script]
    res = run_turn(FakeBackend(script), [], capture=CaptureSpec())
    assert res.tokens == baseline


def test_max_tokens_stops_early():
    res = run_turn(FakeBackend(_script(10)), [], max_tokens=3)
    assert len(res.tokens) == 3


def test_capture_negotiation_refuses_loudly():
    be = FakeBackend(_script(), caps=Capabilities(full_logits=True))
    spec = CaptureSpec(attention=AttentionSpec(mode=AttentionMode.REGION_POOLED))
    with pytest.raises(CaptureUnsupported, match="attention capture"):
        run_turn(be, [], capture=spec)


def test_probes_run_when_due_and_land_in_metrics():
    class EveryOtherProbe:
        def due(self, ctx, m: StepMetrics) -> bool:
            return ctx["i"] % 2 == 0

        def run(self, backend, ctx) -> dict[str, float]:
            return {"commit:tool_call": 0.5}

    res = run_turn(FakeBackend(_script()), [], probes=EveryOtherProbe())
    flagged = [tm.probes for tm in res.metrics]
    assert flagged[0] == {"commit:tool_call": 0.5}
    assert flagged[1] == {}


def test_policy_ref_logits_feed_kl():
    class RefPolicy:
        def ref_logits(self, ctx):
            return np.zeros(32)

        def decide(self, m, events):
            return None

    res = run_turn(FakeBackend(_script()), [], policy=RefPolicy())
    assert all(tm.kl_ref is not None and tm.kl_ref > 0 for tm in res.metrics)
