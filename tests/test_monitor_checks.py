"""Checks and alarms on synthetic turn records: percentile thresholds, value sources, severity raised for values
taken only from tool results, values found nowhere in the context, free text left alone."""

from vitals.monitor.checks import check_turn, source_of, thresholds


def _arg(name, value, lp, found):
    return {"name": name, "value": value, "steps": [1], "min_lp": lp, "found_in": found,
            "read_while_writing": [("tool_result:get_user#1", 0.2)]}


def _records():
    args = [_arg(f"a{i}", f"v{i:04d}", -0.01 * i, ["user:1"]) for i in range(100)]
    return {"ep": [{"turn": 3, "calls": [{"name": "book", "args": args}]}]}


def test_source_of():
    assert source_of(["system/policy", "user:2"]) == "user"
    assert source_of(["tool_result:get_user#1", "obs:3"]) == "tool_result"
    assert source_of(["system/policy"]) == "harness"
    assert source_of(["self:2", "tool_result:get_user#1"]) == "tool_result"
    assert source_of(["self:2", "region:gen_3"]) == "none"
    assert source_of([]) == "none" and source_of(None) is None


def test_thresholds_follow_the_budget():
    th = thresholds(_records(), budget=0.10, alert_budget=0.02)
    assert th["n"] == 100 and -0.92 < th["warn"] < -0.88 and th["alert"] < th["warn"]
    assert thresholds({"ep": []})["warn"] is None


def test_uncertain_argument_and_severity():
    th = {"warn": -0.5, "alert": -2.0}
    rec = {"turn": 4, "calls": [{"name": "book", "args": [
        _arg("payment_id", "gift_card_1", -3.0, ["tool_result:get_user#1"]),      # below the alert threshold
        _arg("cabin", "economy", -1.0, ["user:1"]),                               # below the warn threshold
        _arg("rid", "ABC123", -0.1, ["user:1"]),                                  # confident: nothing
    ]}]}
    alarms = check_turn("ep", rec, th)
    by = {a["arg"]: a for a in alarms}
    assert set(by) == {"payment_id", "cabin"}
    assert by["payment_id"]["severity"] == "alert" and by["cabin"]["severity"] == "warn"
    assert by["payment_id"]["evidence"]["source"] == "tool_result" and by["cabin"]["evidence"]["source"] == "user"
    assert by["payment_id"]["section"] == "tool_result:get_user#1"
    assert "low confidence" in by["payment_id"]["message"]


def test_value_not_in_context():
    tools = [{"function": {"name": "book", "parameters": {"properties": {"rid": {"type": "string"}, "n": {"type": "integer"},
                                                                         "reason": {"type": "string"}}}}}]
    rec = {"turn": 1, "calls": [{"name": "book", "args": [
        _arg("rid", "ZZZ999", -0.1, []), _arg("n", 1234, -0.1, []), _arg("date", "2024-05-24", -0.1, []),
        _arg("reason", "the user changed plans and asked to cancel", -0.1, [])]}]}
    alarms = check_turn("ep", rec, {"warn": None}, tools)
    assert [(a["check"], a["arg"]) for a in alarms] == [("value_not_in_context", "rid")]
