"""All-layer write shares: the per-section parts of every mixing layer's write add up to the layer's own
pre-projection tensor, and the shares of each layer sum to 1. Runs on the 0.8B CI model (attention and
Gated DeltaNet layers)."""

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
def test_write_shares_add_up_on_a_hybrid_model():
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.probes.linear_reading import section_labels
    from vitals.probes.write_share import WriteShareProbe

    be = LlamaCppBackend(GGUFS[0], n_ctx=4096, diagnostic_attention=True)
    try:
        msgs = [{"role": "system", "content": "You are terse. Always name colours in alphabetical order."},
                {"role": "user", "content": "Name three primary colours."}]
        ctx, prompt = be.render_turn(msgs, spans=[{"name": "rule", "text": "Always name colours in alphabetical order."}])
        eng = be.engine
        eng.reset_context(capture_names=set())
        eng.decode(prompt, pos0=0, seq_id=0)
        out = []
        for _ in range(24):
            t = int(np.argmax(eng.logits(-1)))
            if eng.is_eog(t):
                break
            out.append(t)
            eng.decode([t], pos0=len(prompt) + len(out) - 1, seq_id=0)
        labels = section_labels(ctx.regions, len(prompt), len(out))
        probe = WriteShareProbe(be, GGUFS[0], check=True, out_chunk=8)
        rows = probe.measure(prompt, out, labels)
    finally:
        be.close()
    by = {r["section"]: r for r in rows}
    assert "rule" in by and "output" in by
    r0 = rows[0]
    assert r0["check_rel_err_linear"] < 1e-4, r0          # the linear split is exact through norm and gate
    assert r0["check_rel_err_attention"] < 1e-3, r0       # recomputed attention vs the engine's kernel
    for key in ("write_all", "write_attention", "write_linear", "weight"):
        assert abs(sum(r[key] for r in rows) - 1) < 1e-4, key
    per_layer = np.array([r["per_layer"] for r in rows])  # [sections, layers]
    assert np.allclose(per_layer.sum(0), 1, atol=1e-4)
