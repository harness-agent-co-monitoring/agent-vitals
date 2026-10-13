"""Opt-in CPU integration test with a real GGUF reference model.

Set VITALS_TEST_GGUF and VITALS_LLAMA_LIB to enable. This is a functional
smoke test, not a benchmark sensitivity result.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("VITALS_TEST_GGUF") and os.environ.get("VITALS_LLAMA_LIB")),
    reason="set VITALS_TEST_GGUF and VITALS_LLAMA_LIB for the real-model diagnostic test",
)


def test_exact_peak_capture_and_sensitivity_on_cpu(tmp_path, monkeypatch):
    from vitals.analysis.readmap import episode_readmap
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.core.schema import TokenMetrics
    from vitals.core.store import TraceStore
    from vitals.probes.sensitivity import sensitivity_from_store
    from vitals.replay.episode import replay_episode

    monkeypatch.setenv("VITALS_REPLAY_HEAD_MEAN", "0")
    be = LlamaCppBackend(os.environ["VITALS_TEST_GGUF"], lib_path=os.environ["VITALS_LLAMA_LIB"],
                         n_gpu_layers=0, n_ctx=512, diagnostic_attention=True)
    try:
        rule = ("Choose the label red. Follow this rule when answering the question. "
                "Return only the label, with no explanation or additional punctuation. "
                "Review the required label carefully before you answer. ")
        messages = [{"role": "raw", "content": rule + "Question: Which label should you return? Answer:"}]
        spans = [{"name": "rule", "text": rule}]
        ctx, prompt = be.render_turn(messages, spans=spans)
        region = next(r for r in ctx.regions if r.id == "span:rule")
        be.engine.reset_context(capture_names=set())
        be.engine.decode(prompt)
        output = []
        for _ in range(8):
            token = int(np.argmax(be.engine.logits(-1)))
            output.append(token)
            be.engine.decode([token])
        store, eid = TraceStore(tmp_path), "ep-real"
        store.write_messages(eid, 0, messages)
        for step, token in enumerate(output):
            store.append(TokenMetrics(episode_id=eid, turn=0, step=step, token_id=token,
                                      ent=0, m2=0, js=0, fork=False, region_id="gen"))
        store.flush(eid)
        result = replay_episode(be, tmp_path, eid, spans=spans, section_diagnostics=True, chunk=4)
        assert result["turns"] == 1
        rows = episode_readmap(tmp_path / eid, diagnostics=True)
        assert len(rows) == 1
        d = rows[0]["diagnostics"]
        assert d["status"] == "complete"
        assert d["peak_status"] == "complete"
        assert d["focus_status"] == "complete"
        assert 0 <= d["focus_js_bits"] <= d["focus_weight_entropy_bits"] + 1e-8
        for weight in ("equal_step", "mass_weighted"):
            assert d[f"peak_step_{weight}"] + 1e-8 >= d[f"peak_pooled_{weight}"]
        assert d["peak_pooled_mass_weighted"] == pytest.approx(rows[0]["peak"], abs=2e-3)
        original = prompt[region.token_start:region.token_end]
        shuffled = np.random.default_rng(8).permutation(original).tolist()
        sensitivity = sensitivity_from_store(
            be, tmp_path, eid, turn=0, section="rule", steps=[0, 4, 7],
            replacements={"noop": {"token_ids": original},
                          "reverse": {"token_ids": original[::-1]},
                          "shuffle": {"token_ids": shuffled}},
        )
        assert sensitivity["variants"][0]["mean_js_nats"] < 1e-8
        assert all(0 <= v["mean_js_nats"] <= np.log(2) + 1e-8 for v in sensitivity["variants"])
        # Probe rebuilds state, so replay must still work when the backend is reused.
        assert replay_episode(be, tmp_path, eid, force=True,
                              section_diagnostics=True)["turns"] == 1
    finally:
        be.close()
