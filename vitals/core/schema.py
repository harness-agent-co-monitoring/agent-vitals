"""Versioned data contracts shared by the loop, the store, the server, and experiments.

Pydantic models carry validation for everything that crosses a boundary
(manifests, capture specs, summaries, events). The one hot-path type,
``TokenMetrics``, is a slotted dataclass instead — it is created once per
decode step and validated implicitly by the store's parquet schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from vitals import SCHEMA_VERSION


class RegionKind(str, Enum):  # noqa: UP042  (StrEnum would change str()/format() of members)
    SYSTEM = "system"
    TOOL_SCHEMA = "tool_schema"
    FEW_SHOT = "few_shot"
    USER = "user"
    OBSERVATION = "observation"  # off-policy injected content (tool results, command output)
    REASONING = "reasoning"      # model-generated span
    TOOL_CALL = "tool_call"


class EventKind(str, Enum):  # noqa: UP042
    OBSERVATION_INJECTED = "OBSERVATION_INJECTED"
    ERROR_BLOCK = "ERROR_BLOCK"
    TOOL_CALL_EMITTED = "TOOL_CALL_EMITTED"
    COMPACTION = "COMPACTION"
    TURN_START = "TURN_START"
    TURN_END = "TURN_END"


class Region(BaseModel):
    """A contiguous span of context with a single source.

    Token spans are filled in by the backend at prefill time (tokenizer-aware);
    char spans are known at segmentation time.
    """

    id: str
    kind: RegionKind
    role: str
    turn: int = 0
    char_start: int = 0
    char_end: int = 0
    token_start: int | None = None
    token_end: int | None = None
    is_error: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class HarnessEvent(BaseModel):
    """One harness event of an episode (``events.jsonl``): a turn boundary, an observation, a compaction."""

    kind: EventKind
    turn: int
    region_id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class TokenMetrics:
    """One decode step — maps 1:1 onto a parquet row."""

    episode_id: str
    turn: int
    step: int
    token_id: int
    ent: float
    m2: float
    js: float
    fork: bool
    kl_ref: float | None = None
    region_id: str = "gen"
    attn_share: list[float] | None = None
    probes: dict[str, float] = field(default_factory=dict)
    t_ms: float = 0.0
    # surprisal = -log p(sampled token); top_ids/top_lp = the top-K token ids and
    # logprobs of the step's full distribution (K = CaptureSpec.top_logprobs;
    # None when K=0)
    surprisal: float | None = None
    top_ids: list[int] | None = None
    top_lp: list[float] | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "turn": self.turn,
            "step": self.step,
            "token_id": self.token_id,
            "ent": self.ent,
            "m2": self.m2,
            "js": self.js,
            "fork": self.fork,
            "kl_ref": self.kl_ref,
            "region_id": self.region_id,
            "attn_share": self.attn_share,
            "probes": self.probes,
            "t_ms": self.t_ms,
            "surprisal": self.surprisal,
            "top_ids": self.top_ids,
            "top_lp": self.top_lp,
        }


class TurnSummary(BaseModel):
    schema_version: str = SCHEMA_VERSION
    turn: int
    n_tokens: int = 0
    mean_ent: float = 0.0
    fork_rate: float = 0.0
    taskspec_attn: float | None = None
    commit_step: int | None = None
    metabolization_window: int | None = None


class AttentionMode(str, Enum):  # noqa: UP042
    NONE = "none"
    REGION_POOLED = "region_pooled"  # requires diagnostic mode (fa off) in the llama.cpp tier
    PER_HEAD = "per_head"            # full per-head matrices (replay, research)


class ActivationSpec(BaseModel):
    layers: list[int] = Field(default_factory=list)
    dtype: str = "fp16"


class AttentionSpec(BaseModel):
    mode: AttentionMode = AttentionMode.NONE
    layers: list[int] | None = None  # None = all layers the backend can serve


class CaptureUnsupported(RuntimeError):
    """Raised at session start when a CaptureSpec exceeds the backend's tier."""


class CaptureSpec(BaseModel):
    """What the caller wants captured. Backends advertise what they can satisfy
    and refuse the rest loudly at session start."""

    tokens: bool = True
    top_logprobs: int = 0
    # score PROMPT tokens during prefill (ent/m2/surprisal per position); rows
    # land in the parquet under their prompt region ids (sys_0/user_0/obs_k/...)
    prompt_scores: bool = False
    activations: ActivationSpec = Field(default_factory=ActivationSpec)
    attention: AttentionSpec = Field(default_factory=AttentionSpec)
    probes: list[str] = Field(default_factory=list)

    @property
    def wants_activations(self) -> bool:
        return bool(self.activations.layers)

    @property
    def wants_attention(self) -> bool:
        return self.attention.mode is not AttentionMode.NONE

    @property
    def is_empty(self) -> bool:
        """Empty spec = golden-test mode: the loop must be observationally inert."""
        return not (self.wants_activations or self.wants_attention or self.probes
                    or self.top_logprobs or self.prompt_scores)


class EngineInfo(BaseModel):
    kind: str = "llamacpp"
    build: str | None = None         # e.g. "b9910 (f5525f7)"
    flags: dict[str, Any] = Field(default_factory=dict)  # fa mode, capture shim, ngl, ...


class ModelInfo(BaseModel):
    name: str
    path: str | None = None
    sha256: str | None = None
    quant: str | None = None         # e.g. "Q4_K_M", "bf16", "GPTQ-Int4"
    arch: str | None = None          # e.g. "qwen3_5"
    n_layers: int | None = None
    full_attn_layers: list[int] | None = None


class EpisodeManifest(BaseModel):
    """``manifest.json`` of an episode: id, creation time, model and engine, and free ``meta`` (task, reward and
    other fields a harness or ingester adds; ``vitals sections`` and the dashboard read ``meta.reward``)."""

    schema_version: str = SCHEMA_VERSION
    episode_id: str
    library_version: str = ""
    engine: EngineInfo = Field(default_factory=EngineInfo)
    model: ModelInfo | None = None
    ref_model: ModelInfo | None = None  # replay / dual-context reference condition
    sampler: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None
    capture: CaptureSpec = Field(default_factory=CaptureSpec)
    harness_template_hash: str | None = None
    created_at: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _stamp_library_version(self) -> EpisodeManifest:
        if not self.library_version:
            from vitals import __version__

            self.library_version = __version__
        return self
