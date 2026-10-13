"""Section writes for factored probes: projected onto a logit-margin direction they reproduce the attention
rows of SectionAttribution, the parts match the engine's attention tensors, and the read weights sum to one.
Runs on the 0.8B CI model."""

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
def test_section_writes_match_attribution():
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.engine.weights import GGUFWeights
    from vitals.probes.attribution import SectionAttribution, runner_up
    from vitals.probes.linear_reading import section_labels
    from vitals.probes.section_writes import SectionWrites

    be = LlamaCppBackend(GGUFS[0], n_ctx=4096)
    try:
        msgs = [{"role": "system", "content": "You are terse. Always answer with the colour blue."},
                {"role": "user", "content": "What colour is the sky at noon? Answer in one sentence."}]
        ctx, prompt = be.render_turn(msgs, spans=[{"name": "rule", "text": "Always answer with the colour blue."}])
        eng = be.engine
        eng.reset_context(capture_names=set())
        eng.decode(prompt, pos0=0, seq_id=0)
        out, steps = [], []
        for _ in range(6):
            z = eng.logits(-1)
            t = int(np.argmax(z))
            steps.append((t, runner_up(z, t)))
            out.append(t)
            eng.decode([t], pos0=len(prompt) + len(out) - 1, seq_id=0)
        tokens = prompt + out
        labels = section_labels(ctx.regions, len(prompt), len(out))
        L = len(prompt)
        n_layers = be.node_map.n_layers
        att = SectionAttribution(be, GGUFS[0], linear=False)
        res = att.attribute(tokens, labels, {"d": (L + 2, steps[3][0], steps[3][1])})
        sw = SectionWrites(be, GGUFS[0], layers=[n_layers // 2 - 1, n_layers - 1], check=True)
        cap = sw.capture(tokens, labels, [L - 1, L + 2])
        err = dict(sw._err)
        w = GGUFWeights(GGUFS[0], device="cuda")
        gamma = w.get("output_norm.weight", torch.float32).cpu().numpy()
        U = w.rows("output.weight" if "output.weight" in w else "token_embd.weight", [steps[3][0], steps[3][1]]).cpu().numpy()
    finally:
        be.close()
    assert np.sqrt(err["attention"][0] / err["attention"][1]) < 1e-2
    p = cap["positions"][L + 2]
    h = p["h"][-1]
    d = gamma * (U[0] - U[1]) / np.sqrt((h * h).mean() + 1e-6)
    rows = {r["section"]: r["direct_attention"] for r in res["rows"]}
    for i, name in enumerate(cap["sections"]):
        assert abs(float(d @ p["writes"][-1, i]) - rows[name]) < 1e-3 * max(1.0, abs(rows[name])), name
    for q in cap["positions"].values():
        assert np.allclose(q["read"].sum(-1), 1.0, atol=1e-3)
        assert q["writes"].shape == (2, len(cap["sections"]), h.shape[0])


@pytest.mark.skipif(not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS or not torch.cuda.is_available(),
                    reason="pinned engine build, 0.8B GGUF or CUDA not available")
def test_section_writes_linear_match_attribution():
    """With the recurrent layers split, the writes reproduce SectionAttribution's attention + linear rows and
    the attention-only capture, the linear parts match the engine's own outputs, and a mid cutoff keeps only
    the layers at or below it. Both sides decode the same batches: llama.cpp's activations move slightly with
    the batch split, and the linear layers show it more than attention."""
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.engine.weights import GGUFWeights
    from vitals.probes.attribution import SectionAttribution, runner_up
    from vitals.probes.linear_reading import section_labels
    from vitals.probes.section_writes import SectionWrites

    be = LlamaCppBackend(GGUFS[0], n_ctx=4096)
    try:
        msgs = [{"role": "system", "content": "You are terse. Always answer with the colour blue."},
                {"role": "user", "content": "What colour is the sky at noon? Answer in one sentence."}]
        ctx, prompt = be.render_turn(msgs, spans=[{"name": "rule", "text": "Always answer with the colour blue."}])
        eng = be.engine
        eng.reset_context(capture_names=set())
        eng.decode(prompt, pos0=0, seq_id=0)
        out, steps = [], []
        for _ in range(6):
            z = eng.logits(-1)
            t = int(np.argmax(z))
            steps.append((t, runner_up(z, t)))
            out.append(t)
            eng.decode([t], pos0=len(prompt) + len(out) - 1, seq_id=0)
        tokens = prompt + out
        labels = section_labels(ctx.regions, len(prompt), len(out))
        L = len(prompt)
        n_layers = be.node_map.n_layers
        mid = n_layers // 2 - 1
        att = SectionAttribution(be, GGUFS[0], linear=True)
        res = att.attribute(tokens, labels, {"d": (L + 2, steps[3][0], steps[3][1])})
        sw = SectionWrites(be, GGUFS[0], layers=[mid, n_layers - 1], linear=True, check=True)
        cap = sw.capture(tokens, labels, [L + 2])
        n_lin_mid = sum(1 for x in sw._linear if x.layer <= mid)
        cap2 = sw.capture(tokens, labels, [L - 1, L + 2])                  # two positions: exactness only
        err = dict(sw._err)
        cap_attn = SectionWrites(be, GGUFS[0], layers=[mid, n_layers - 1]).capture(tokens, labels, [L + 2])
        wvec = np.random.default_rng(0).normal(size=sw.n_embd).astype(np.float32)
        capw = sw.capture(tokens, labels, [L + 2], project=wvec)
        w = GGUFWeights(GGUFS[0], device="cuda")
        gamma = w.get("output_norm.weight", torch.float32).cpu().numpy()
        U = w.rows("output.weight" if "output.weight" in w else "token_embd.weight", [steps[3][0], steps[3][1]]).cpu().numpy()
    finally:
        be.close()
    assert np.sqrt(err["linear"][0] / err["linear"][1]) < 1e-3
    assert np.sqrt(err["attention"][0] / err["attention"][1]) < 1e-3
    assert 0 < n_lin_mid < sum(1 for k in range(n_layers) if k not in be.node_map.full_attn_layers)
    p = cap["positions"][L + 2]
    h = p["h"][-1]
    d = gamma * (U[0] - U[1]) / np.sqrt((h * h).mean() + 1e-6)
    rows = {r["section"]: (r["direct_attention"], r["direct_linear"]) for r in res["rows"]}
    for i, name in enumerate(cap["sections"]):
        a, li = rows[name]
        assert abs(float(d @ p["writes"][-1, i]) - (a + li)) < 1e-3 * max(1.0, abs(a + li)), name
        assert abs(float(d @ p["writes_linear"][-1, i]) - li) < 1e-3 * max(1.0, abs(li)), name
    pa = cap_attn["positions"][L + 2]
    assert "writes_linear" not in pa
    assert np.allclose(p["writes"] - p["writes_linear"], pa["writes"], rtol=1e-3, atol=1e-3)
    assert np.allclose(p["read"], pa["read"], atol=1e-5)
    assert np.allclose(p["h"], pa["h"], atol=1e-4)
    pw = capw["positions"][L + 2]
    assert pw["contrib_pos"].shape == (2, L + 3)
    lab = np.array(labels[:L + 3])
    for i, name in enumerate(capw["sections"]):                       # per-position sums = attention writes . w
        for k in range(2):
            want = float((pw["writes"][k, i] - pw["writes_linear"][k, i]) @ wvec)
            assert abs(float(pw["contrib_pos"][k, lab == name].sum()) - want) < 1e-3 * max(1.0, abs(want)), name
    assert set(cap2["positions"]) == {L - 1, L + 2} and cap2["positions"][L - 1]["writes_linear"].shape == p["writes_linear"].shape
