"""Section-level interpretability for LLM agent harnesses.

Teacher-forced replay of logged episodes and live capture behind an
OpenAI-compatible server, over a llama.cpp engine (see ``vitals.engine``).
"""

__version__ = "0.1.0"

# Version of every stored artifact. Bump on any change to the trace schema.
# 0.2.0: adds per-token surprisal + top-K logprobs (surprisal/top_ids/top_lp, nullable).
SCHEMA_VERSION = "0.2.0"

_EXPORTS = {
    "LlamaCppBackend": "vitals.backends.llamacpp",
    "TraceStore": "vitals.core.store",
    "episode_readmap": "vitals.analysis.readmap",
    "record_turn": "vitals.replay",
    "replay_episode": "vitals.replay",
    "replay_store": "vitals.replay",
}

__all__ = ["SCHEMA_VERSION", "__version__", *_EXPORTS]


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'vitals' has no attribute {name!r}")
