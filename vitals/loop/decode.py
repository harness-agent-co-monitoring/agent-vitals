"""The decode loop — the choke point every feature hooks into.

Four taps: CAPTURE (backend.step), METRICS (accumulator), POLICY (route), and
BRANCH PROBES. Correctness rule 1: with an empty CaptureSpec, no policy, and no
probes, this loop's token stream is identical to the backend's native sampling
under the same seed — enforced by golden tests, engine-side by construction
(native sampler chain).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from vitals.backends.base import Backend, StepOutput, negotiate
from vitals.core.metrics import MetricsAccumulator, StepMetrics
from vitals.core.schema import CaptureSpec, HarnessEvent, TokenMetrics


class Policy(Protocol):
    """Tap 3. Must preserve output validity for harness users."""

    def ref_logits(self, ctx: Any) -> np.ndarray | None: ...

    def decide(self, m: StepMetrics, events: list[HarnessEvent]) -> Any | None: ...


class Probes(Protocol):
    """Tap 4. Side-effect-free by contract: branches are never written back."""

    def due(self, ctx: Any, m: StepMetrics) -> bool: ...

    def run(self, backend: Backend, ctx: Any) -> dict[str, float]: ...


class Sink(Protocol):
    def emit(self, tm: TokenMetrics) -> None: ...


class NullSink:
    def emit(self, tm: TokenMetrics) -> None:  # pragma: no cover - trivial
        pass


@dataclass
class TurnResult:
    episode_id: str
    turn: int
    tokens: list[int] = field(default_factory=list)
    metrics: list[TokenMetrics] = field(default_factory=list)
    events: list[HarnessEvent] = field(default_factory=list)


def run_turn(
    backend: Backend,
    messages: list[dict],
    *,
    episode_id: str = "episode",
    turn: int = 0,
    capture: CaptureSpec | None = None,
    policy: Policy | None = None,
    probes: Probes | None = None,
    sink: Sink | None = None,
    events: list[HarnessEvent] | None = None,
    max_tokens: int | None = None,
    accumulator: MetricsAccumulator | None = None,
    capture_sink=None,  # callable (step_idx, StepOutput) -> None; receives raw captures
    tools: list[dict] | None = None,
    chat_kwargs: dict | None = None,
    spans: list[dict] | None = None,
) -> TurnResult:
    capture = capture or CaptureSpec()
    sink = sink or NullSink()
    events = events or []
    negotiate(capture, backend.capabilities(), backend.name)

    acc = accumulator or MetricsAccumulator()
    kw: dict = {}
    if tools:
        kw["tools"] = tools
    if chat_kwargs:
        kw["chat_kwargs"] = chat_kwargs
    if spans:
        kw["spans"] = spans
    ctx = backend.prefill(messages, capture, **kw)
    result = TurnResult(episode_id=episode_id, turn=turn, events=list(events))

    step_idx = 0
    while not backend.done(ctx):
        if max_tokens is not None and step_idx >= max_tokens:
            break
        t0 = time.perf_counter()

        step: StepOutput = backend.step(ctx)                      # tap 1: CAPTURE
        if capture_sink is not None and (step.resid or step.attn):
            capture_sink(step_idx, step)

        ref = policy.ref_logits(ctx) if policy is not None else None
        m = acc.update(step.logits, ref_logits=ref)               # tap 2: METRICS

        route = policy.decide(m, result.events) if policy is not None else None  # tap 3: POLICY

        probe_values: dict[str, float] = {}
        if probes is not None and probes.due(ctx, m):              # tap 4: BRANCH PROBES
            probe_values = probes.run(backend, ctx)

        token = backend.sample(ctx, step, route)

        surprisal = top_ids = top_lp = None
        k = capture.top_logprobs
        if k and step.logits is not None:
            lg = np.asarray(step.logits, dtype=np.float64)
            lp = lg - lg.max()
            lp -= np.log(np.exp(lp).sum())
            surprisal = float(-lp[token])
            idx = np.argpartition(lp, -k)[-k:]
            idx = idx[np.argsort(lp[idx])[::-1]]
            top_ids = [int(i) for i in idx]
            top_lp = [float(lp[i]) for i in idx]

        tm = TokenMetrics(
            episode_id=episode_id,
            turn=turn,
            step=step_idx,
            token_id=token,
            ent=m.ent,
            m2=m.m2,
            js=m.js,
            fork=m.fork,
            kl_ref=m.kl_ref,
            attn_share=step.attn_share,
            probes=probe_values,
            t_ms=(time.perf_counter() - t0) * 1e3,
            surprisal=surprisal,
            top_ids=top_ids,
            top_lp=top_lp,
        )
        sink.emit(tm)
        result.metrics.append(tm)
        result.tokens.append(token)

        ctx = backend.advance(ctx, token)
        step_idx += 1

    return result
