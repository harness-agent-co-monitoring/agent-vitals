"""Conditional sensitivity must preserve positions and rebuild prefix state."""

from types import SimpleNamespace

import numpy as np
import pytest

from vitals.probes.sensitivity import distribution_js, section_sensitivity


class Engine:
    n_vocab = 8
    n_ctx_per_seq = 128
    _ctx_opts = {"n_batch": 2}

    def __init__(self):
        self.histories = []

    def reset_context(self, **kwargs):
        assert kwargs == {"capture_names": set(), "capture_modes": {}}
        self.tokens = []
        self.histories.append(self.tokens)

    def decode(self, tokens):
        self.tokens.extend(tokens)

    def logits(self, index):
        assert index == -1
        values = np.zeros(self.n_vocab)
        # Depend on both the changed prompt token and the forced history.
        values[self.tokens[1]] = 2
        values[self.tokens[-1]] += 1
        return values


def test_full_distribution_js_properties():
    p, q = np.log([.8, .2]), np.log([.2, .8])
    assert distribution_js(p, p) == pytest.approx(0)
    assert distribution_js(p, q) == pytest.approx(.192744757)
    assert distribution_js(p, q) == pytest.approx(distribution_js(q, p))
    assert distribution_js([1000, -1000], [-1000, 1000]) == pytest.approx(np.log(2))


def test_fixed_prefix_noop_and_multiple_replacements():
    be = SimpleNamespace(engine=Engine(), _replay_kept=("old",), _kv_tokens=[0, 1], _kv_plan_sig=("old",))
    prompt, output = [0, 1, 2, 3], [4, 5, 6]
    result = section_sensitivity(be, prompt, output, start=1, end=2,
                                 replacements={"noop": [1], "alternative": [7], "other": [2]}, steps=[2, 0, 1])
    assert result["steps"] == [0, 1, 2]
    assert result["variants"][0]["mean_js_nats"] == pytest.approx(0)
    assert result["variants"][1]["mean_js_nats"] > 0
    assert result["variants"][0]["changed_tokens"] == 0
    assert result["variants"][1]["changed_tokens"] == 1
    assert be._replay_kept is None
    assert be._kv_tokens == [] and be._kv_plan_sig is None
    assert be.engine.histories == [prompt + output[:2], prompt + output[:2],
                                    [0, 7, 2, 3, 4, 5], [0, 2, 2, 3, 4, 5]]
    for variant in result["variants"]:
        assert [v["token_id"] for v in variant["positions"]] == output


@pytest.mark.parametrize("replacements,steps", [({"long": [1, 2]}, [0]), ({"bad": [8]}, [0]),
                                                ({"bad": [1.5]}, [0]), ({"ok": [1]}, [3]),
                                                ({"ok": [1]}, [-1]), ({"ok": [1]}, [])])
def test_invalid_replacements_or_steps_fail_before_inference(replacements, steps):
    be = SimpleNamespace(engine=Engine())
    with pytest.raises(ValueError):
        section_sensitivity(be, [0, 1, 2, 3], [4, 5, 6], start=1, end=2,
                            replacements=replacements, steps=steps)
    assert be.engine.histories == []


def test_store_prompt_verification_and_token_replacements(tmp_path):
    from vitals.core.schema import TokenMetrics
    from vitals.core.store import TraceStore
    from vitals.probes.sensitivity import sensitivity_from_store

    store = TraceStore(tmp_path)
    eid = "ep-probe"
    store.write_messages(eid, 0, [{"role": "raw", "content": "example"}])
    from vitals.core.schema import Region, RegionKind

    region = Region(id="span:rule", kind=RegionKind.SYSTEM, role="system", token_start=1, token_end=2)
    store.write_regions(eid, 0, [region])
    store.append_attn_tokens(eid, 0, [0, 1, 2, 3], ["a", "rule", "b", "c"], [region], [.25] * 4)
    for t, tok in enumerate([4, 5, 6]):
        store.append(TokenMetrics(episode_id=eid, turn=0, step=t, token_id=tok,
                                  ent=1, m2=.1, js=0, fork=False, region_id="gen"))
    store.flush(eid)
    be = SimpleNamespace(engine=Engine(), render_turn=lambda *a, **k: (None, [0, 1, 2, 3]))
    result = sensitivity_from_store(be, tmp_path, eid, turn=0, section="rule",
                                    replacements={"replace": {"token_ids": [7]}}, steps=[0, 2])
    assert result["variants"][0]["mean_js_nats"] > 0
    be.render_turn = lambda *a, **k: (None, [0, 7, 2, 3])
    with pytest.raises(ValueError, match="rendered prompt differs"):
        sensitivity_from_store(be, tmp_path, eid, turn=0, section="rule",
                               replacements={"replace": {"token_ids": [7]}}, steps=[0, 2])
