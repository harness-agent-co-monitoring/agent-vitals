"""Signed section attribution: at every decision the linearized margin equals the engine's logit margin from
an independent pass, the per-section parts reproduce the engine's own tensors, and switching the recurrent
layers off leaves the attention layers' contributions unchanged. Runs on the 0.8B CI model."""

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
def test_section_attribution_matches_the_engine_margin():
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.probes.attribution import SectionAttribution, runner_up
    from vitals.probes.linear_reading import section_labels

    be = LlamaCppBackend(GGUFS[0], n_ctx=4096)     # flash attention on, as in production use
    try:
        msgs = [{"role": "system", "content": "You are terse. Always answer with the colour blue."},
                {"role": "user", "content": "What colour is the sky at noon? Answer in one sentence."}]
        ctx, prompt = be.render_turn(msgs, spans=[{"name": "rule", "text": "Always answer with the colour blue."}])
        eng = be.engine
        eng.reset_context(capture_names=set())
        eng.decode(prompt, pos0=0, seq_id=0)
        out, steps = [], []
        for _ in range(8):
            z = eng.logits(-1)
            t = int(np.argmax(z))
            steps.append((t, runner_up(z, t), float(z[t] - z[runner_up(z, t)])))
            out.append(t)
            eng.decode([t], pos0=len(prompt) + len(out) - 1, seq_id=0)
        tokens = prompt + out
        labels = section_labels(ctx.regions, len(prompt), len(out))
        L = len(prompt)
        decisions = {"first": (L - 1, steps[0][0], steps[0][1]), "fifth": (L + 3, steps[4][0], steps[4][1])}
        full = SectionAttribution(be, GGUFS[0], linear=True, check=True)
        res = full.attribute(tokens, labels, decisions)
        err = dict(full._err)
        att_only = SectionAttribution(be, GGUFS[0], linear=False)
        res2 = att_only.attribute(tokens, labels, decisions)
    finally:
        be.close()
    for k, i in (("first", 0), ("fifth", 4)):
        s = res["summary"][k]
        assert abs(s["margin"] - steps[i][2]) < 0.05, (k, s, steps[i])     # another batch shape: small numeric drift
        assert abs(s["margin_linear"] - s["margin"]) < 1e-2 * max(1.0, abs(s["margin"])), (k, s)
        assert abs(s["direct_total"] + s["remainder"] - s["margin_linear"]) < 1e-6
    assert np.sqrt(err["attention"][0] / err["attention"][1]) < 1e-2    # recomputed attention vs the flash kernel
    assert np.sqrt(err["linear"][0] / err["linear"][1]) < 1e-4
    rows = {(r["decision"], r["section"]): r for r in res["rows"]}
    rows2 = {(r["decision"], r["section"]): r for r in res2["rows"]}
    assert ("first", "rule") in rows and ("fifth", "output") in rows
    for key, r in rows.items():
        assert abs(r["direct_attention"] - rows2[key]["direct_attention"]) < 1e-3 * max(1.0, abs(r["direct_attention"]))
        assert rows2[key]["direct_linear"] == 0.0
