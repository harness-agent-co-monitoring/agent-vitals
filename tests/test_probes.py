"""Probe-package integration on the 0.8B (CPU): fork read, calibrations."""

import os
from glob import glob
from pathlib import Path

import numpy as np
import pytest

VENDOR = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin"
GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF"
             "/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))
pytestmark = pytest.mark.skipif(
    not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS,
    reason="pinned engine build or 0.8B GGUF not available")


def test_retrieval_heads_and_position_bias_smoke():
    from vitals.probes import position_bias, retrieval_heads

    rh = retrieval_heads(GGUFS[0], n_probes=2, n_ctx=1024, n_filler=15,
                         layers=[3, 7], engine_kwargs={"n_gpu_layers": 0})
    assert rh["ranked_heads"], "no heads scored"
    assert rh["ranked_heads"][0]["needle_mass"] >= rh["ranked_heads"][-1]["needle_mass"]

    pb = position_bias(GGUFS[0], depths=[0.2, 0.8], n_ctx=1024, n_filler=20,
                       layers=[3], engine_kwargs={"n_gpu_layers": 0})
    assert len(pb["curve"]) == 2
    assert all(c["share"] >= 0 and c["ratio"] >= 0 for c in pb["curve"])
    assert {"sink_mass", "local_mass", "control_mass"} <= set(rh["ranked_heads"][0])
    assert rh["answer_acc"] is not None

    from vitals.probes import induction_heads
    ih = induction_heads(GGUFS[0], n_probes=1, seq_len=16, n_ctx=512, layers=[3],
                         engine_kwargs={"n_gpu_layers": 0})
    assert ih["ranked_heads"] and 0.0 <= ih["ranked_heads"][0]["induction"] <= 1.0


class _FakeEng:
    """Whitespace tokenizer whose tokens carry their leading space, like BPE."""

    def tokenize(self, text, add_special=True, parse_special=True):
        import re
        return [m.group(0) for m in re.finditer(r"\s*\S+", text)]

    def detokenize(self, toks):
        return "".join(toks)


def test_token_span_and_control_span():
    from vitals.probes.calibrate import _control_span, _filler, _token_span
    eng = _FakeEng()
    text = "Note: a b. The access code for locker K12 is 54321. Note: c d."
    ts, te = _token_span(eng, text, "54321")
    toks = eng.tokenize(text)
    assert "".join(toks[ts:te]).strip().startswith("54321")
    # the span starts at the token that covers the first character of the needle
    assert text.find("54321") >= sum(len(t) for t in toks[:ts])
    assert _token_span(eng, text, "absent") is None
    rng = np.random.default_rng(0)
    assert len(_filler(rng, 7).split("Note:")) == 8
    cs, ce = _control_span(rng, ts=40, te=45, n_tok=200)
    assert ce - cs == 5 and 4 <= cs and ce <= 40 - 4
    cs, ce = _control_span(rng, ts=6, te=11, n_tok=200)      # no room before: after the needle
    assert ce - cs == 5 and cs >= 11 + 4
