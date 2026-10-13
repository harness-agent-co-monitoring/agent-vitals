"""Section ablation and per-head reading on the 0.8B CI model: a no-op replacement changes nothing, the
original margin matches an independent decode, localization reports groups then members, and per-head
reading sums to one per head and averages back to the write-share probe's head-mean attention weight; the
per-position reading over the span sums by section to the same values."""

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
needs_model = pytest.mark.skipif(not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS or not torch.cuda.is_available(),
                                 reason="pinned engine build, 0.8B GGUF or CUDA not available")


def test_spans_from_labels():
    from vitals.probes.ablation import spans_from_labels

    assert spans_from_labels(["a", "a", "b", "a", "c", "c"]) == {"a": [(0, 2), (3, 4)], "b": [(2, 3)], "c": [(4, 6)]}


@needs_model
def test_ablation_and_head_reading():
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.probes.ablation import SectionAblation, spans_from_labels
    from vitals.probes.attribution import runner_up
    from vitals.probes.head_reading import HeadReading, as_read_map
    from vitals.probes.linear_reading import section_labels
    from vitals.probes.write_share import WriteShareProbe

    be = LlamaCppBackend(GGUFS[0], n_ctx=4096, n_seq_max=2)
    try:
        msgs = [{"role": "system", "content": "You are terse. Always answer with the colour blue. Never mention green."},
                {"role": "user", "content": "What colour is the sky at noon? Answer in one sentence."}]
        ctx, prompt = be.render_turn(msgs, spans=[{"name": "rule", "text": "Always answer with the colour blue."}])
        eng = be.engine
        eng.reset_context(capture_names=set())
        eng.decode(prompt, pos0=0, seq_id=0)
        out, steps = [], []
        for _ in range(6):
            z = eng.logits(-1)
            t = int(np.argmax(z))
            steps.append((t, runner_up(z, t), float(z[t] - z[runner_up(z, t)])))
            out.append(t)
            eng.decode([t], pos0=len(prompt) + len(out) - 1, seq_id=0)
        L = len(prompt)
        tokens = prompt + out
        labels = section_labels(ctx.regions, L, len(out))
        spans = spans_from_labels(labels)
        rule = spans["rule"]
        rule_toks = [tokens[s:e] for s, e in rule]
        abl = SectionAblation(be, seed=1)
        decision = (L + 2, steps[3][0], steps[3][1])
        res = abl.effects(tokens, {"d": decision}, {"rule": rule, "noop": rule}, replacements={"noop": rule_toks})
        groups = {"system": {k: v for k, v in spans.items() if k == "rule" or k.startswith("region:sys")},
                  "user": {k: v for k, v in spans.items() if k.startswith("region:user")}}
        rows = abl.localize(tokens, decision, groups, top_groups=2)
        hr = HeadReading(be).measure(tokens, labels, span=(L - 1, L + len(out) - 1), decisions=[L + 2], positions=True)
        ws = WriteShareProbe(be, GGUFS[0], check=False).measure(prompt, out, labels)
    finally:
        be.close()
    assert res["delta"]["noop"]["d"] == 0.0
    assert abs(res["base"]["d"] - steps[3][2]) < 0.05
    assert res["delta"]["rule"]["d"] != 0.0
    assert [r["level"] for r in rows][:2] == ["group", "group"] and any(r["level"] == "section" for r in rows)
    assert np.allclose(hr["at"][L + 2].sum(-1), 1.0, atol=1e-3)
    assert np.allclose(hr["span"].sum(-1), 1.0, atol=1e-3)
    weight = {r["section"]: r["weight"] for r in ws}
    rm = as_read_map(hr["span"])
    for i, name in enumerate(hr["sections"]):
        assert abs(rm[i] - weight[name]) < 2e-3, name
    # per-position reading over the span adds up by section to the head-mean reading
    pos = hr["span_pos"]
    assert np.isclose(pos.sum(), 1.0, atol=1e-3)
    lab = np.asarray(labels[:len(pos)])
    for i, name in enumerate(hr["sections"]):
        assert abs(pos[lab == name].sum() - rm[i]) < 1e-4, name


@needs_model
def test_head_reading_residuals_match_section_writes():
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.probes.head_reading import HeadReading
    from vitals.probes.linear_reading import section_labels
    from vitals.probes.section_writes import SectionWrites

    be = LlamaCppBackend(GGUFS[0], n_ctx=4096)
    try:
        msgs = [{"role": "system", "content": "You are terse. Always answer with the colour blue."},
                {"role": "user", "content": "What colour is the sky at noon?"}]
        ctx, prompt = be.render_turn(msgs, spans=[{"name": "rule", "text": "Always answer with the colour blue."}])
        L = len(prompt)
        labels = section_labels(ctx.regions, L, 0)
        last = be.node_map.n_layers - 1
        hr = HeadReading(be, residual_layers=[last]).measure(prompt, labels, span=(L - 1, L), decisions=[L - 1])
        sw = SectionWrites(be, GGUFS[0], layers=[last]).capture(prompt, labels, [L - 1])
    finally:
        be.close()
    assert hr["section_residual"].shape == (1, len(hr["sections"]), sw["positions"][L - 1]["h"].shape[-1])
    assert np.allclose(hr["residual_at"][L - 1][0], sw["positions"][L - 1]["h"][0], atol=1e-3)
    rule = hr["sections"].index("rule")
    assert np.abs(hr["section_residual"][0, rule]).sum() > 0
