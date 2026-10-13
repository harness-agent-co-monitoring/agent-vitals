"""Log ingesters: benchmark logs and trajectories -> vitals episodes."""

from vitals.addons.metaharness.config import CFG


def detok_engine():
    """A CPU engine over the small same-vocabulary model, for tokenizing logged text."""
    from vitals.engine.binding import LlamaEngine

    return LlamaEngine(str(CFG.detok_gguf), n_ctx=512, n_gpu_layers=0)
