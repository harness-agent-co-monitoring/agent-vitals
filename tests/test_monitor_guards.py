"""Monitoring edge cases: missing confidence values, tool-call text that is not an argument object, nested keys,
webhook retries, explanation cache keys, ledger reuse, ablation of decisions before the fork, and capture parts
from a one-token ubatch."""

import types

import numpy as np

from vitals.monitor.checks import thresholds
from vitals.monitor.turns import locate_calls


def test_thresholds_ignore_missing_confidence():
    args = [{"name": f"a{i}", "value": i, "min_lp": -0.01 * i} for i in range(40)]
    args.append({"name": "x", "value": 1, "min_lp": float("nan")})
    th = thresholds({"ep": [{"turn": 0, "calls": [{"name": "t", "args": args}]}]})
    assert th["n"] == 40 and np.isfinite(th["warn"]) and np.isfinite(th["alert"])


def test_locate_calls_skips_non_object_json_and_reads_top_level_keys():
    assert locate_calls("<tool_call>[1, 2]</tool_call>") == []
    text = '<tool_call>{"name": "f", "arguments": {"x": {"id": 1}, "id": 2}}</tool_call>'
    (call,) = locate_calls(text)
    spans = {a["name"]: text[slice(*a["span"])] for a in call["args"]}
    assert spans == {"x": '{"id": 1}', "id": "2"}


def test_webhook_retries_failures_and_resends_on_escalation():
    from vitals.monitor.pipeline import notify

    flag = {"id": "f1", "episode": "e", "turn": 1, "check": "c", "severity": "warn", "message": "m"}
    state = {"notified": []}
    calls = []
    assert notify(state, [flag], "http://hook", "warn", post=lambda *_: calls.append(1) or False)["failed"] == 1
    assert state["notified"] == []                                     # not delivered: tried again next pass
    assert notify(state, [flag], "http://hook", "warn", post=lambda *_: True)["sent"] == 1
    assert notify(state, [flag], "http://hook", "warn", post=lambda *_: True)["sent"] == 0
    assert notify(state, [{**flag, "severity": "alert"}], "http://hook", "warn", post=lambda *_: True)["sent"] == 1
    low = {**flag, "id": "f2", "severity": "info"}
    assert notify(state, [low], "http://hook", "warn", post=lambda *_: True)["sent"] == 0


def test_explanation_key_follows_the_probe():
    from vitals.monitor.measures import Normal
    from vitals.monitor.pipeline import explanation_key
    from vitals.monitor.probes import LinearProbe

    p = LinearProbe("b", 3, np.zeros(4), np.ones(4), np.ones(4), 0.0, threshold=1.0)
    k1 = explanation_key(["f1"], [p], None)
    p.threshold = 2.0
    assert explanation_key(["f1"], [p], None) != k1
    store_normal = {"decision": Normal.fit([{"key": "a", "lift": 1.0}] * 5)}
    assert explanation_key(["f1"], [p], store_normal) != explanation_key(["f1"], [p], None)


def test_ledger_reuse_needs_a_matching_capture():
    from vitals.monitor.ledger import covers

    have = {"version": 1, "model": {"name": "m.gguf", "size": 10}, "layers": [3, 7], "writes": True, "linear": False}
    want = {"model": {"name": "m.gguf", "size": 10}, "layers": [7], "writes": True, "linear": False}
    assert covers(have, want)
    assert not covers(have, {**want, "layers": [7, 11]})
    assert not covers(have, {**want, "linear": True})
    assert not covers(have, {**want, "model": {"name": "other.gguf", "size": 10}})
    assert covers({**have, "writes": True}, {**want, "writes": False})
    assert covers({**have, "model": None}, want)                     # older ledgers did not record the model


class _FakeEngine:
    """Logits that depend on the tokens seen so far on each sequence, so a replaced span changes later margins."""

    def __init__(self, nb=4):
        self._ctx_opts = {"n_seq_max": 2, "n_batch": nb}
        self.seqs = {0: [], 1: []}
        self.last = None

    def reset_context(self, **_):
        self.seqs = {0: [], 1: []}

    def decode(self, toks, pos0, seq_id):
        s = self.seqs[seq_id]
        assert len(s) == pos0, "decode must continue the sequence"
        s.extend(toks)
        self.last = seq_id

    def logits(self, _i):
        z = np.zeros(8)
        z[1] = float(sum(self.seqs[self.last]))
        return z

    def branch_copy(self, dst, src):
        self.seqs[dst] = list(self.seqs[src])

    def branch_drop(self, seq):
        self.seqs[seq] = []

    def tokenize(self, text, **_):
        return [0] * len(text.split())


def test_ablation_of_decisions_before_the_fork():
    from vitals.probes.ablation import SectionAblation

    eng = _FakeEngine(nb=4)
    abl = SectionAblation(types.SimpleNamespace(engine=eng))
    toks = [1] * 16
    res = abl.effects(toks, {"early": (3, 1, 0), "late": (14, 1, 0)}, {"tail": [(9, 11)]},
                      replacements={"tail": [[0, 0]]})
    assert res["delta"]["tail"]["early"] == 0.0                      # the span lies after the early decision
    assert res["delta"]["tail"]["late"] == 2.0                       # two ones replaced by zeros before it


def test_one_token_ubatch_keeps_every_row():
    from vitals.engine.binding import LlamaEngine

    fake = types.SimpleNamespace(_captured_parts={"res": [np.ones((512, 8)), np.full(8, 2.0)],
                                                  "att": [np.ones((4, 512, 512)), np.ones((4, 513))]})
    out = LlamaEngine.collect_captured(fake)
    assert out["res"].shape == (513, 8) and out["res"][-1, 0] == 2.0
    assert out["att"].shape == (4, 513, 513)
