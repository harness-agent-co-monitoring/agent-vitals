"""Native llama.cpp sampler chains from OpenAI-style params.

Sampling is delegated to llama.cpp so vitals output is token-identical with the
ecosystem by construction. Chain order mirrors llama.cpp's default sampler
sequence for the samplers we expose:
top_k -> top_p -> min_p -> temperature -> dist  (greedy when temperature <= 0).
"""

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass

from vitals.engine.binding import LlamaEngine, LlamaSamplerChainParams


@dataclass(frozen=True)
class SamplerConfig:
    """OpenAI-style sampling params. Defaults are pass-through (no truncation),
    NOT llama-server's opinionated defaults — explicit beats implicit for
    experiments. temperature<=0 means greedy."""

    temperature: float = 1.0
    top_k: int = 0          # 0 = disabled
    top_p: float = 1.0      # 1.0 = disabled
    min_p: float = 0.0      # 0.0 = disabled
    seed: int = 42

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature, "top_k": self.top_k,
            "top_p": self.top_p, "min_p": self.min_p, "seed": self.seed,
        }


class NativeSampler:
    """Owns one llama_sampler chain. Create per episode (the dist sampler's RNG
    advances across steps; recreating at prefill makes episodes reproducible)."""

    def __init__(self, engine: LlamaEngine, config: SamplerConfig):
        self.engine = engine
        self.config = config
        lib = engine.lib
        params = LlamaSamplerChainParams(no_perf=True)
        self.chain = lib.llama_sampler_chain_init(params)
        if not self.chain:
            raise RuntimeError("llama_sampler_chain_init failed")
        if config.is_greedy:
            lib.llama_sampler_chain_add(self.chain, lib.llama_sampler_init_greedy())
        else:
            if config.top_k > 0:
                lib.llama_sampler_chain_add(self.chain, lib.llama_sampler_init_top_k(config.top_k))
            if config.top_p < 1.0:
                lib.llama_sampler_chain_add(self.chain, lib.llama_sampler_init_top_p(config.top_p, 1))
            if config.min_p > 0.0:
                lib.llama_sampler_chain_add(self.chain, lib.llama_sampler_init_min_p(config.min_p, 1))
            lib.llama_sampler_chain_add(self.chain, lib.llama_sampler_init_temp(config.temperature))
            lib.llama_sampler_chain_add(self.chain, lib.llama_sampler_init_dist(C.c_uint32(config.seed)))

    def sample(self, output_idx: int = -1) -> int:
        """Sample from the context's logits at `output_idx` (native apply+accept)."""
        return int(self.engine.lib.llama_sampler_sample(self.chain, self.engine.ctx, output_idx))

    def reset(self) -> None:
        self.engine.lib.llama_sampler_reset(self.chain)

    def free(self) -> None:
        if getattr(self, "chain", None):
            self.engine.lib.llama_sampler_free(self.chain)
            self.chain = None
