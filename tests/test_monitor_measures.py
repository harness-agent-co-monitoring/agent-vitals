"""Section measures: section keys, read lift and read z at the decision and over the reasoning, probe lift, the
section normal and z, read and probe peaks with their hottest line."""

from __future__ import annotations

import numpy as np

from vitals.monitor.measures import (
    READ_FLOOR,
    Normal,
    probe_peak,
    read_normals,
    read_rows,
    read_z,
    section_keys,
    section_peak,
    split_measures,
    split_report,
)
from vitals.monitor.probes import LinearProbe


def test_section_keys():
    secs = ["region:sys_0", "rule", "tool:get_order", "user:1", "self:1", "tool_result:get_order#1", "user:2",
            "tool_result:get_order#2", "tool_result:cancel#1", "output", "obs[3]", "obs[4]"]
    assert section_keys(secs) == ["template", "rule", "tool:get_order", "user:earlier", "self:latest",
                                  "tool_result:get_order:earlier", "user:latest", "tool_result:get_order:latest",
                                  "tool_result:cancel:latest", "reasoning", "obs:earlier", "obs:latest"]


def test_probe_lift_is_signed_and_length_normalized():
    secs, toks = ["rule", "policy", "output"], [10, 90, 20]
    rows = {r["section"]: r for r in split_measures([1.0, -1.0, 5.0], secs, toks)}
    # context = rule + policy: sum |c| = 2, L = 100 -> uniform per token 0.02
    assert np.isclose(rows["rule"]["lift"], (1.0 / 10) / 0.02)
    assert np.isclose(rows["policy"]["lift"], (-1.0 / 90) / 0.02)
    assert rows["output"]["lift"] is None and rows["output"]["key"] == "reasoning"
    assert np.isclose(sum(abs(r["share"]) for r in rows.values()), 1.0)


def test_normal_and_z():
    rng = np.random.default_rng(0)
    ref = [{"key": "rule", "lift": float(v)} for v in rng.normal(5.0, 1.0, 200)]
    ref += [{"key": "flat", "lift": 2.0} for _ in range(50)] + [{"key": "rare", "lift": 1.0}]
    n = Normal.fit(ref)
    assert abs(n.median["rule"] - 5.0) < 0.2 and abs(n.spread["rule"] - 1.0) < 0.2
    assert n.spread["flat"] > 0                       # floored, so a constant section gives finite z
    assert n.z("rare", 3.0) is None                   # too few reference decisions
    assert abs(n.z("rule", 8.0) - 3.0) < 0.6
    assert Normal.from_dict(n.to_dict()).z("rule", 8.0) == n.z("rule", 8.0)


def test_split_report_adds_up_and_orders_by_z():
    rng = np.random.default_rng(1)
    d = 16
    probe = LinearProbe("p", 3, np.zeros(d), np.ones(d), rng.normal(size=d), 0.0)
    writes = rng.normal(size=(3, d))
    h = writes.sum(0) + rng.normal(size=d)
    rep = split_report(probe, h, writes, ["rule", "user:1", "output"], [10, 30, 5])
    assert np.isclose(rep["section_total"] + rep["remainder"], float(h @ probe.w_eff))
    rows = {r["section"]: r for r in rep["rows"]}
    normal = Normal({"rule": rows["rule"]["lift"] - 10.0, "user:latest": rows["user:1"]["lift"]},
                    {"rule": 1.0, "user:latest": 1.0}, {"rule": 9, "user:latest": 9})
    rep = split_report(probe, h, writes, ["rule", "user:1", "output"], [10, 30, 5], normal=normal)
    assert rep["rows"][0]["section"] == "rule" and np.isclose(rep["rows"][0]["z"], 10.0)
    assert rep["rows"][-1]["z"] is None               # the turn's own reasoning has no lift, so no z


def test_probe_peak_finds_the_line():
    lines = ["Always be polite and helpful to every customer you talk to today.\n",
             "Do NOT modify the tests under any circumstances whatsoever.\n",
             "Answer briefly and use plain language in every reply you write.\n"]
    text, labels, c = [], [], []
    for li, line in enumerate(lines):
        for w in line.split(" "):
            tok = w if w.endswith("\n") else w + " "
            text.append(tok)
            labels.append("rule")
            c.append(-1.0 if li == 1 and w.strip(" \n") in {"NOT", "modify", "tests"} else -0.01)
    text += ["x "] * 10
    labels += ["user:1"] * 10
    c += [0.01] * 10
    out = probe_peak(np.array(c), labels, text, prompt_text="".join(lines))
    assert "rule" in out and "user:1" not in out      # user:1 is under 32 tokens
    assert out["rule"]["peak"] < 0
    assert out["rule"]["hot_line"].startswith("Do NOT modify the tests")


def test_read_rows_lift_at_both_footings():
    secs, toks = ["rule", "user:1", "output"], [10, 90, 20]
    at = np.array([[0.3, 0.3, 0.4], [0.1, 0.5, 0.4]])                # two heads, per-section shares
    span = np.array([0.4, 0.2, 0.4])
    rows = {r["section"]: r for r in read_rows(secs, toks, at, span)}
    # head mean at the decision: rule 0.2, user 0.4; context mass 0.6, L = 100
    assert np.isclose(rows["rule"]["read"], 0.2) and np.isclose(rows["rule"]["read_lift"], 0.2 / 0.6 * 100 / 10)
    assert np.isclose(rows["rule"]["reasoning_lift"], 0.4 / 0.6 * 100 / 10)
    assert np.isclose(rows["rule"]["log_reasoning_lift"], np.log2(rows["rule"]["reasoning_lift"]))
    assert rows["output"]["read_lift"] is None and rows["output"]["key"] == "reasoning"
    # lift weighted by tokens over the context averages to one
    assert np.isclose(sum(r["read_lift"] * r["tokens"] for r in rows.values() if r["read_lift"]) / 100, 1.0)
    assert read_rows(secs, toks, at)[0]["reasoning_lift"] is None   # no reasoning reading: no reasoning lift


def test_read_normals_floor_and_z():
    rng = np.random.default_rng(0)
    secs, toks = ["rule", "policy", "user:1", "output"], [40, 400, 60, 30]
    per = []
    for _ in range(40):
        at = np.array([0.05, 0.3, 0.25, 0.4]) * np.exp(rng.normal(scale=[0.05, 0.6, 0.6, 0.1]))
        per.append(read_rows(secs, toks, at / at.sum(), at / at.sum()))
    n = read_normals(per)
    assert set(n) == {"decision", "reasoning"}
    assert n["decision"].spread["rule"] == READ_FLOOR                 # a steady section is held at the floor
    assert n["decision"].spread["policy"] > READ_FLOOR - 0.5
    hot = np.array([0.4, 0.2, 0.1, 0.3])                               # the rule read about 10x more than usual
    rows = {r["section"]: r for r in read_z(read_rows(secs, toks, hot, hot), n)}
    assert rows["rule"]["read_z_decision"] > 2.5 and rows["rule"]["read_z_reasoning"] > 2.5
    assert rows["output"]["read_z_decision"] is None
    assert read_normals([read_rows(secs, toks, hot)])["decision"].z("rule", 1.0) is None   # under min_ref decisions


def test_read_peak_is_the_section_peak_of_attention():
    lines = ["Always be polite and helpful to every customer you talk to today.\n",
             "Before taking any action that updates the booking database, obtain explicit confirmation.\n",
             "Answer briefly and use plain language in every reply you write.\n"]
    text, labels, a = [], [], []
    for li, line in enumerate(lines):
        for w in line.split(" "):
            text.append(w if w.endswith("\n") else w + " ")
            labels.append("policy")
            a.append(0.05 if li == 1 and w.strip(" \n") in {"explicit", "confirmation.", "obtain"} else 0.001)
    out = section_peak(np.array(a), labels, text, prompt_text="".join(lines))
    assert out["policy"]["peak"] > 0 and out["policy"]["hot_line"].startswith("Before taking any action")
    norm = sum(a) / len(a)
    assert np.isclose(out["policy"]["peak"], np.mean(sorted(a)[-8:]) / norm)
    assert probe_peak is section_peak
