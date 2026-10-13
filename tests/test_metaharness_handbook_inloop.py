"""HANDBOOK in-loop replay/digest helpers (no model, no GPU)."""
import json

import pandas as pd

from vitals.addons.metaharness.digest.inloop_handbook import _aggregate, _read_once
from vitals.addons.metaharness.ingest.handbook import declared_system_spans
from vitals.addons.metaharness.replay.inloop_handbook import agent_of, load_declaration, pick_trials


def _trial(root, name, score, with_log=True):
    d = root / name
    (d / "agent" / "completions").mkdir(parents=True)
    if with_log:
        (d / "agent" / "completions" / "c.json").write_text("{}")
    (d / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": score}}}))
    return d


def test_pick_trials_is_score_stratified_and_deterministic(tmp_path):
    for i, s in enumerate([0.9, 0.1, 0.5, 0.7, 0.3, 0.8]):
        _trial(tmp_path, f"task{i}__x", s)
    _trial(tmp_path, "task9__nolog", 1.0, with_log=False)      # no completion logs: never picked
    picked = pick_trials(tmp_path, 4)
    names = [t.name for t, _ in picked]
    assert names == ["task0__x", "task5__x", "task1__x", "task4__x"]   # two strongest, two weakest
    assert pick_trials(tmp_path, 4) == picked


def test_agent_of_and_declaration_lookup(tmp_path):
    assert agent_of(tmp_path / "evolve-quote_rule-t2") == "quote_rule"
    assert agent_of(tmp_path / "openhands-baseline-t2") == "openhands-baseline"
    (tmp_path / "baseline_openhands.py").write_text(
        "def prompt_sections(p):\n    return [('a', p[:5]), ('b', p[5:])]\n"
        "def feedback_sections(o):\n    return [('nudge', 'REMINDER')] if 'REMINDER' in o else None\n")
    ps, fb = load_declaration(tmp_path, "openhands-baseline")
    assert callable(ps) and ps("hello world")[0] == ("a", "hello")
    assert fb("x REMINDER y") == [("nudge", "REMINDER")]
    assert load_declaration(tmp_path, "missing") is None


def test_declared_spans_must_be_verbatim():
    p = "Today is Monday. Be careful. Read the files."
    ok = declared_system_spans(p, lambda s: [("date", "Today is Monday."), ("care", "Be careful.")])
    assert ok == [("sys:date", "Today is Monday."), ("sys:care", "Be careful.")]
    assert declared_system_spans(p, lambda s: [("x", "not in prompt"), ("y", "Be careful.")]) is None
    assert declared_system_spans(p, lambda s: [("only", "Today is Monday.")]) is None


def test_read_once_and_aggregate():
    rows = []
    for turn, kind, lift in [(1, "read", 1.8), (3, "write", 0.2), (5, "exec", 0.3)]:
        rows.append({"episode_id": "e1", "turn": turn, "section": "doc:2/1. Rules", "group": "handbook", "role": "handbook",
                     "kind": kind, "lift": lift, "peak": 3.0, "share": 0.1, "span_tok": 100, "consult_ratio": 1.0,
                     "consult_contrast": 1.0, "turn_id": f"e1:{turn}"})
    rows.append({"episode_id": "e1", "turn": 1, "section": "sys:begin", "group": "system", "role": "system", "kind": "read",
                 "lift": 10.0, "peak": float("nan"), "share": 0.01, "span_tok": 11, "consult_ratio": 1.5,
                 "consult_contrast": 2.0, "turn_id": "e1:1"})
    df = pd.DataFrame(rows)
    assert bool(_read_once(df)["e1"]) is True
    agg = _aggregate(df)
    assert set(agg.role) == {"handbook", "system"}
    assert "sys:begin" in set(agg.section) and abs(agg[agg.section == "sys:begin"].consult_ratio.iloc[0] - 1.5) < 1e-6


def test_feedback_spans_are_labeled_inside_tool_results():
    from vitals.addons.metaharness.ingest.handbook import build_spans
    msgs = [{"role": "system", "content": "Today is Monday. Be careful. Read files."},
            {"role": "user", "content": "Handle my inbox please, everything unread today."},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "name": "syntara__listFiles", "content": "[Tool executed.]{\"files\": [\"a.pdf\"]}\nREMINDER: quote the rule before writing."}]

    def fb(o):
        return [("reminder", "REMINDER: quote the rule before writing.")] if "REMINDER" in o else None

    spans = build_spans(msgs, [], None, fb)
    names = [s["name"] for s in spans]
    assert "fb:3:reminder" in names and "obs:3:listFiles" in names and "instruction" in names
    assert next(s for s in spans if s["name"] == "fb:3:reminder")["msg_index"] == 3


def test_load_tools_uses_logged_names_and_order(tmp_path):
    from vitals.addons.metaharness.ingest.handbook import load_tools, logged_tool_names
    cached = [{"type": "function", "function": {"name": n, "parameters": {}}} for n in ("a", "b", "think")]
    cj = tmp_path / "tools.json"
    cj.write_text(json.dumps(cached))
    logged = [{"title": "Tool B", "mcp_tool": {"name": "b"}}, {"title": "think"}, {"title": "A", "mcp_tool": {"name": "a"}}]
    assert logged_tool_names(logged) == ["b", "think", "a"]
    trial = tmp_path / "t"
    (trial / "agent").mkdir(parents=True)
    out = load_tools(trial, [{"tools": logged}], cj)
    assert [x["function"]["name"] for x in out] == ["b", "think", "a"]
    (trial / "agent" / "tools_openai.json").write_text(json.dumps(cached))
    assert [x["function"]["name"] for x in load_tools(trial, [{"tools": logged}], cj)] == ["a", "b", "think"]
