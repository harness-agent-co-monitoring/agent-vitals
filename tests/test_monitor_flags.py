"""Explain on flag and its outputs: both views in every explanation (read rows picked by reasoning lift, read z and
peak; read share over the probe's layers and read z for the behavior beside the probe rows), the label-free read
normals from a ledger, the webhook, the dashboard's alarm API."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from vitals.monitor.flags import explain_decision, probe_view, read_view
from vitals.monitor.measures import Normal, normals_to_dict, read_normals, read_rows
from vitals.monitor.pipeline import flag_id, post_webhook, webhook_payload
from vitals.monitor.probes import LinearProbe

SECTIONS = ["region:sys_0", "rule", "tool:get", "user:1", "self:1", "user:2", "output"]


def decision(rng, d=16, rule_push=2.0, v=None):
    """A ledger decision (the fields ledger_episode writes) where the rule writes along v."""
    R = len(SECTIONS)
    writes = rng.normal(scale=0.2, size=(1, R, d))
    if v is not None:
        writes[0, 1] += rule_push * v
    at = np.abs(rng.normal(size=(8, R)))
    at[:, 3] *= 4                                                   # the agent reads user:1 most
    at /= at.sum(1, keepdims=True)
    return {"episode": "ep-0", "turn": 3, "sections": SECTIONS, "tokens": np.array([5, 20, 60, 30, 40, 25, 15]),
            "at": at, "span": at, "residual_layers": np.array([7]), "residual_at": writes[:, :, :].sum(1),
            "writes": writes, "meta": {"position": 120, "a_text": "<tool_call>", "b_text": "I"}}


def test_explanation_carries_both_views():
    rng = np.random.default_rng(0)
    d = 16
    v = rng.normal(size=d)
    v /= np.linalg.norm(v)
    probe = LinearProbe("bound_by_rule", 7, np.zeros(d), np.ones(d), v, -1.0, threshold=0.0,
                        meta={"alert_threshold": 5.0})
    dec = decision(rng, d, 3.0, v)
    normal = Normal({"rule": 0.0, "user:latest": 0.0, "user:earlier": 0.0, "tool:get": 0.0, "self:latest": 0.0},
                    dict.fromkeys(["rule", "user:latest", "user:earlier", "tool:get", "self:latest"], 1.0),
                    dict.fromkeys(["rule", "user:latest", "user:earlier", "tool:get", "self:latest"], 10))
    rv = read_view(dec)
    assert rv["rows"][0]["section"] == "user:1" and all(r["key"] not in ("template", "reasoning") for r in rv["rows"])
    pv = probe_view(dec, probe, normal)
    assert pv["severity"] == "warn" and pv["rows"][0]["section"] == "rule" and "read" in pv["rows"][0]
    ex = explain_decision(dec, [probe], normals=[normal],
                          alarms=[{"message": "Low-confidence argument.", "check": "uncertain_argument"}])
    assert ex["read_map"]["rows"] and ex["probes"][0]["probe"] == "bound_by_rule"
    text = " ".join(ex["claims"])
    assert "Read map" in text and "Probe 'bound_by_rule' fires" in text and "'rule'" in text and "Low-confidence" in text
    json.dumps(ex, default=str)                                     # serializable as written to the cache


def test_read_view_picks_rows_by_lift_z_and_peak():
    rng = np.random.default_rng(1)
    dec = decision(rng)
    span = np.full((8, len(SECTIONS)), 0.02)
    span[:, 5] = 0.6                                                # over the reasoning the agent reads user:2 most
    span[:, 6] = 0.3
    dec["span"] = span / span.sum(1, keepdims=True)
    dec["meta"]["reason_peaks"] = {"tool:get": {"peak": 40.0, "hot_line": "get(order_id) returns the order"}}
    # normals where the rule is usually read far less than here: it is the most unusual
    ref = []
    for _ in range(20):
        sp = dec["span"].mean(0).copy()
        sp[1] /= 16
        ref.append(read_rows(SECTIONS, dec["tokens"], dec["at"], sp / sp.sum()))
    rv = read_view(dec, top=2, normals=read_normals(ref))
    assert rv["lift_by"] == "reasoning_lift" and rv["z_by"] == "read_z_reasoning" and rv["peak_by"] == "reason_peak"
    assert rv["best"] == {"lift": "user:2", "z": "rule", "peak": "tool:get"}
    picked = {r["section"]: r["picked_by"] for r in rv["rows"]}
    assert set(picked) == {"user:2", "rule", "tool:get"} and picked["tool:get"] == ["peak"]
    assert rv["rows"][0]["section"] == "user:2"                      # ordered by read lift over the reasoning
    ex = explain_decision(dec, [], read_normals=read_normals(ref))
    text = " ".join(ex["claims"])
    assert "before acting the agent reads 'user:2'" in text and "'rule' is the most unusual" in text
    assert "get(order_id) returns the order" in text


def test_probe_view_reads_the_probe_layers_and_behavior_z():
    rng = np.random.default_rng(2)
    d = 16
    v = rng.normal(size=d)
    v /= np.linalg.norm(v)
    dec = decision(rng, d, 3.0, v)
    dec["heads"] = np.array([(3, 0), (3, 1), (7, 0), (7, 1), (11, 0), (11, 1), (15, 0), (15, 1)])
    dec["at"][4:, 1] += 0.5                                         # heads above layer 7 read the rule a lot
    dec["at"] /= dec["at"].sum(1, keepdims=True)
    neg = [read_rows(SECTIONS, dec["tokens"], dec["at"][:4] * [1, 0.2, 1, 1, 1, 1, 1], None) for _ in range(6)]
    probe = LinearProbe("bound_by_rule", 7, np.zeros(d), np.ones(d), v, -1.0, threshold=0.0,
                        meta={"read_normal": normals_to_dict(read_normals(neg))})
    pv = probe_view(dec, probe)
    row = next(r for r in pv["rows"] if r["section"] == "rule")
    assert np.isclose(row["read_layers"], dec["at"][:4, 1].mean())          # heads at layers 3 and 7 only
    assert np.isclose(row["read"], dec["at"][:, 1].mean())
    assert row["read_z_decision"] is not None and row["read_z_reasoning"] is None   # this normal has no reasoning
    assert pv["read_normal"] is True


def test_ledger_read_normals(tmp_path):
    import pandas as pd

    from vitals.monitor.ledger import ledger_read_normals

    assert ledger_read_normals(tmp_path) is None
    rng = np.random.default_rng(3)
    rows = []
    for e in range(8):
        at = np.abs(rng.normal(size=len(SECTIONS)))
        at /= at.sum()
        rows += [{"episode": f"ep{e}", "turn": 1, "position": 99, "section": s, "tokens": t, "read_at": a,
                  "read_reasoning": a, "reason_peak": None} for s, t, a in zip(SECTIONS, [5, 20, 60, 30, 40, 25, 15], at, strict=True)]
    pd.DataFrame(rows).to_parquet(tmp_path / "ledger.parquet", index=False)
    n = ledger_read_normals(tmp_path)
    assert set(n) == {"decision", "reasoning"} and n["decision"].n["rule"] == 8 and "reasoning" not in n["decision"].n


def test_webhook_delivers_and_survives_failure():
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    flag = {"episode": "ep-0", "turn": 3, "check": "probe:x", "severity": "warn", "message": "m"}
    assert post_webhook(f"http://127.0.0.1:{srv.server_address[1]}/hook", webhook_payload("s", flag, None))
    srv.shutdown()
    assert got and got[0]["check"] == "probe:x" and got[0]["source"] == "agent-vitals"
    assert not post_webhook("http://127.0.0.1:9/none", {"x": 1}, timeout=0.5, retries=0)
    assert flag_id(flag) == flag_id(dict(flag))


def test_dashboard_alarm_api(tmp_path):
    import pytest

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from test_dashboard import make_store

    from vitals.dashboard.app import create_app
    from vitals.dashboard.data import Dashboard
    from vitals.monitor.checks import alarm_cache

    store = make_store(tmp_path / "store", [0.4])
    cache = alarm_cache(tmp_path / "alarms", store.root)
    (cache / "explanations").mkdir(parents=True)
    flag = {"id": "f1", "episode": "ep-0", "turn": 1, "check": "probe:x", "kind": "probe", "severity": "warn",
            "message": "Probe 'x' scores +1.00.", "explained": True}
    (cache / "flags.jsonl").write_text(json.dumps(flag) + "\n")
    ex = {"episode": "ep-0", "turn": 1, "read_map": {"rows": [], "reasoning_share": 0.2}, "probes": [], "claims": ["c"]}
    (cache / "explanations" / "ep-0.jsonl").write_text(json.dumps(ex) + "\n")
    (cache / "_pipeline_state.json").write_text(json.dumps({"notified": ["f1"], "last_run": "2026-10-12T14:00:00"}))
    dash = Dashboard([tmp_path / "store"], alarm_dir=tmp_path / "alarms")
    client = TestClient(create_app(dash))
    sid = client.get("/api/sources").json()["stores"][0]["id"]
    res = client.get(f"/api/stores/{sid}/alarms").json()
    assert res["pipeline"] and res["flags"][0]["id"] == "f1" and res["last_run"].startswith("2026")
    assert client.get(f"/api/stores/{sid}/episodes/ep-0/flags").json()[0]["claims"] == ["c"]
    assert client.get(f"/api/stores/{sid}/episodes/nope/flags").status_code == 404
