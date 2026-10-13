"""End to end on the 0.8B CI model: an explanation ranks sections, ablates the top ones, passes the no-op
check and states claims in probability terms."""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
VENDOR = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin"
GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))


@pytest.mark.skipif(not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS or not torch.cuda.is_available(),
                    reason="pinned engine build, 0.8B GGUF or CUDA not available")
def test_explain_decision():
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.explain import explain
    from vitals.probes.attribution import runner_up
    from vitals.probes.linear_reading import section_labels

    be = LlamaCppBackend(GGUFS[0], n_ctx=4096, n_seq_max=2)
    try:
        msgs = [{"role": "system", "content": "You are terse. Always answer with the colour blue. Be polite."},
                {"role": "user", "content": "What colour is the sky at noon? Answer in one word."}]
        ctx, prompt = be.render_turn(msgs, spans=[{"name": "rule", "text": "Always answer with the colour blue."},
                                                  {"name": "manners", "text": "Be polite."}])
        eng = be.engine
        eng.reset_context(capture_names=set())
        eng.decode(prompt, pos0=0, seq_id=0)
        z = eng.logits(-1)
        a = int(np.argmax(z))
        L = len(prompt)
        labels = section_labels(ctx.regions, L, 1)
        r = explain(be, prompt + [a], labels, (L - 1, a, runner_up(z, a)), span=(L - 1, L), top_k=2,
                    exclude=("output", "cue"))
    finally:
        be.close()
    assert r["checks"]["noop_delta"] == 0.0
    ab = [s for s in r["sections"] if s["ablated"]]
    assert len(ab) == 2 and all(0.0 <= s["p_without"] <= 1.0 for s in ab)
    assert r["claims"] and r["joint"]["delta"] is not None
    assert {s["section"] for s in r["sections"]} >= {"rule", "manners"}
