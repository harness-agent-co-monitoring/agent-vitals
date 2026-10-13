import os
from glob import glob
from pathlib import Path

import numpy as np
import pytest

from vitals.engine import binding, nodes

VENDOR_LIB = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin/libllama.so"


def test_nodes_full_attn_pattern():
    m = nodes.for_model("qwen3_5", 64)
    assert len(m.full_attn_layers) == 16
    assert m.full_attn_layers[:3] == (3, 7, 11)
    assert m.residual_node(32) == "l_out-32"
    assert m.attn_node(3) == "kq_soft_max-3"


def test_nodes_refuse_attention_on_linear_layers():
    m = nodes.REGISTRY["qwen3_5:0.8b"]
    with pytest.raises(ValueError, match="linear-attention"):
        m.attn_node(0)


def test_nodes_unknown_arch_fails_loudly():
    with pytest.raises(KeyError, match="eval-callback"):
        nodes.for_model("mystery_arch", 12)


@pytest.mark.skipif(not VENDOR_LIB.exists(), reason="pinned engine not built")
def test_doctor_against_pinned_build():
    report = binding.doctor(VENDOR_LIB)
    assert report["ok"], report
    assert report["symbols_missing"] == []


def test_doctor_reports_missing_lib(monkeypatch, tmp_path):
    monkeypatch.setenv("VITALS_LLAMA_LIB", str(tmp_path / "nope.so"))
    report = binding.doctor()
    assert report["ok"] is False
    assert "error" in report
    monkeypatch.delenv("VITALS_LLAMA_LIB")
    monkeypatch.setattr(binding, "_DEFAULT_CANDIDATES", ["no/such/libllama.so"])
    report2 = binding.doctor()
    assert report2["ok"] is False


GGUFS_08B = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))


@pytest.mark.skipif(not VENDOR_LIB.exists() or not GGUFS_08B,
                    reason="pinned engine or 0.8B GGUF absent")
def test_score_tokens_matches_stepwise():
    """Chunked logits_all scoring == one-token-at-a-time scoring (and exercises
    the cross-chunk boundary via a tiny n_batch)."""
    eng = binding.LlamaEngine(GGUFS_08B[0], n_ctx=256, n_gpu_layers=0, n_batch=8)
    try:
        toks = eng.tokenize("The quick brown fox jumps over the lazy dog near the river bank.")
        assert len(toks) > 10  # spans multiple 8-token chunks
        got = eng.score_tokens(list(toks))

        eng.reset_context()
        want = []
        eng.decode([toks[0]])
        for t in toks[1:]:
            lg = eng.logits(-1).astype(np.float64)
            lg -= lg.max()
            want.append(float(lg[t] - np.log(np.exp(lg).sum())))
            eng.decode([t])
        assert got.shape == (len(toks) - 1,)
        # batched and single-token decodes take different GEMM paths; observed
        # deltas are ~1e-2 in logprob. The scoring LOGIC is what this verifies.
        assert np.allclose(got, np.asarray(want), atol=0.05), (got - np.asarray(want))

        from vitals.core.metrics import nll_summary
        s = nll_summary(got)
        assert s["n"] == len(toks) - 1 and s["min20_nll"] >= s["mean_nll"]
    finally:
        eng.close()


@pytest.mark.skipif(not VENDOR_LIB.exists() or not GGUFS_08B,
                    reason="pinned engine or 0.8B GGUF absent")
def test_layer_mean_capture_matches_host_path():
    """"layer_mean" capture: the rows pooled over heads and layers on the device
    equal the host path (head-mean copies + numpy layer mean), nothing is left
    on the host, and logits_rows equals the per-row logits of the same decode."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("device pooling needs CUDA")
    from vitals.engine.nodes import for_model
    from vitals.replay.pooling import layer_head_mean

    eng = binding.LlamaEngine(GGUFS_08B[0], n_ctx=512, flash_attn=binding.FLASH_ATTN_DISABLED,
                              capture_names={"placeholder"})
    try:
        nm = for_model(eng.arch, eng.n_layer)
        names = {nm.attn_node(lay) for lay in nm.full_attn_layers}
        toks = eng.tokenize("The quick brown fox jumps over the lazy dog near the river bank.")
        head, tail = toks[:-4], toks[-4:]

        eng.reset_context(capture_names=names, capture_modes=dict.fromkeys(names, "head_mean"))
        eng.decode(head)
        eng.decode(tail, logits_all=True)
        cap = eng.collect_captured()
        if not isinstance(binding.LlamaEngine._gpu_pool_state, dict):
            pytest.skip(f"device pooling unavailable: {binding.LlamaEngine._gpu_pool_reason}")
        assert set(cap) == names
        ref = layer_head_mean({n: cap[n][..., :eng.n_past] for n in names})
        ref_logits = np.stack([eng.logits(j) for j in range(4)])
        assert np.array_equal(eng.logits_rows(4), ref_logits)

        eng.reset_context(capture_names=names, capture_modes=dict.fromkeys(names, "layer_mean"))
        eng.decode(head)
        eng.decode(tail, logits_all=True)
        got = eng.take_layer_mean(eng.n_past)
        assert got is not None and tuple(got.shape) == ref.shape == (4, eng.n_past)
        assert np.allclose(got.cpu().numpy(), ref, atol=1e-5)
        assert eng.collect_captured() == {}          # nothing crossed to the host
        assert eng.take_layer_mean() is None          # the accumulator was cleared
    finally:
        eng.close()
