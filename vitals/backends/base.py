"""Backend protocol + CaptureSpec negotiation.

Each backend advertises capabilities; unsatisfiable capture requests fail
loudly at session start with an explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from vitals.core.schema import AttentionMode, CaptureSpec, CaptureUnsupported


@dataclass(frozen=True)
class Capabilities:
    """What a backend can capture; ``negotiate`` refuses a capture request the backend cannot serve."""

    full_logits: bool = False
    top_logprobs: bool = False
    activations: bool = False
    attention_modes: tuple[AttentionMode, ...] = (AttentionMode.NONE,)
    branch_probes: bool = False
    interventions: bool = False
    note: str = ""


def negotiate(spec: CaptureSpec, caps: Capabilities, backend_name: str) -> None:
    """Refuse loudly, with the reason and the tier that would satisfy the request."""
    problems: list[str] = []
    if spec.wants_activations and not caps.activations:
        problems.append("activation capture")
    if spec.wants_attention and spec.attention.mode not in caps.attention_modes:
        problems.append(f"attention capture (mode={spec.attention.mode.value})")
    if spec.probes and not caps.branch_probes:
        problems.append(f"probes {spec.probes}")
    if problems:
        hint = caps.note or "the backend does not capture this"
        raise CaptureUnsupported(
            f"backend {backend_name!r} cannot satisfy: {', '.join(problems)}. {hint}"
        )


@dataclass
class StepOutput:
    """One decode step's raw observables (tap 1)."""

    logits: np.ndarray
    resid: dict[int, np.ndarray] = field(default_factory=dict)
    attn: dict[int, np.ndarray] = field(default_factory=dict)
    # per-region attention shares for THIS token (region order + trailing
    # generated-so-far bucket); None unless attention was captured and region
    # token spans resolved
    attn_share: list[float] | None = None


@runtime_checkable
class Backend(Protocol):
    """The decode loop's engine contract."""

    name: str

    def capabilities(self) -> Capabilities: ...

    def prefill(self, messages: list[dict], capture: CaptureSpec) -> Any:
        """Ingest history; resolve region token spans; return an opaque ctx."""
        ...

    def step(self, ctx: Any) -> StepOutput: ...

    def sample(self, ctx: Any, step: StepOutput, route: Any | None = None) -> int:
        """Native sampler chain; `route` may override sampler state (tap 3)."""
        ...

    def advance(self, ctx: Any, token: int) -> Any: ...

    def done(self, ctx: Any) -> bool: ...
