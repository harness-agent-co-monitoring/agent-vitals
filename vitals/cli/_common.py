"""Arguments and helpers shared by the commands."""

from __future__ import annotations

import argparse

from vitals.core.paths import CACHE_KINDS


def csv(value: str | None) -> list[str]:
    """A comma-separated argument as a list, empty items dropped."""
    return [x for x in (value or "").split(",") if x]


def episodes(value: str | None) -> list[str] | None:
    """``--episodes``: the named episodes, or None for every episode of the store."""
    return csv(value) or None


def turns(value: str | None) -> list[int] | None:
    """``--turns``: ``all`` or empty means every turn (None), else comma-separated turn indices."""
    if not value or value == "all":
        return None
    return [int(t) for t in csv(value)]


def add_store(p: argparse.ArgumentParser, help: str = "trace store directory (episodes)") -> None:
    p.add_argument("--store", required=True, help=help)


def add_episodes(p: argparse.ArgumentParser, help: str = "comma-separated episode ids (default: all)") -> None:
    p.add_argument("--episodes", default="", help=help)


def add_model(p: argparse.ArgumentParser, *, n_ctx: int, gguf_help: str = "reference model GGUF (the model that "
              "produced the logged tokens)") -> None:
    """``--gguf``, ``--n-ctx`` (per sequence), ``--n-gpu-layers`` and ``--lib`` for a command that loads a model."""
    p.add_argument("--gguf", required=True, help=gguf_help)
    p.add_argument("--n-ctx", type=int, default=n_ctx, help=f"context window per sequence in tokens (default {n_ctx})")
    p.add_argument("--n-gpu-layers", type=int, default=999, help="layers offloaded to the GPU (default: all)")
    p.add_argument("--lib", default=None, help="libllama path (default: $VITALS_LLAMA_LIB, else the pinned build)")


def add_cache(p: argparse.ArgumentParser, kind: str, flag: str = "--out-dir", what: str = "cache directory") -> None:
    p.add_argument(flag, default=None, help=f"{what} (default: ${CACHE_KINDS[kind]}, else {kind}/ under the vitals "
                                            "cache, ~/.cache/vitals)")


def load_backend(args: argparse.Namespace, *, n_seq_max: int = 1, diagnostic_attention: bool = False):
    """The command's model, loaded with ``n_seq_max`` sequences of ``--n-ctx`` tokens each."""
    from vitals.backends.llamacpp import LlamaCppBackend

    return LlamaCppBackend(args.gguf, n_ctx=n_seq_max * args.n_ctx, n_seq_max=n_seq_max, n_gpu_layers=args.n_gpu_layers,
                           diagnostic_attention=diagnostic_attention, lib_path=args.lib)
