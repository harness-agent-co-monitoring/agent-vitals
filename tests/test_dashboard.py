import json
import time

import numpy as np
import pytest

from vitals.analysis.readmap import cached_readmap, episode_readmap
from vitals.core.schema import TokenMetrics
from vitals.core.store import TraceStore

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vitals.dashboard.app import create_app  # noqa: E402
from vitals.dashboard.data import Dashboard  # noqa: E402


def add_turn(store, eid, turn, policy_mass, sections=("policy", "rules")):
    """One replayed turn: a 48-token prompt with 16-token sections, attention split as given."""
    regs = [{"id": "sys_0", "token_start": 0, "token_end": 48}]
    regs += [{"id": f"span:{s}", "token_start": 16 * i, "token_end": 16 * (i + 1)} for i, s in enumerate(sections)]
    for step in range(8):
        store.append(TokenMetrics(episode_id=eid, turn=turn, step=step, token_id=2, ent=1 + step % 3, m2=.1, js=0,
                                  fork=False, region_id="gen", attn_share=[.5] * len(regs) + [.5]))
    store.flush(eid)
    rest = (1 - policy_mass) / 32
    vec = np.r_[np.full(16, policy_mass / 16), np.full(32, rest)]
    with (store.root / eid / "regions.jsonl").open("a") as f:
        f.write(json.dumps({"turn": turn, "regions": regs}) + "\n")
    store.append_attn_received(eid, turn, vec)
    store.append_attn_tokens(eid, turn, range(48), ["w"] * 48, regs, vec)


def make_store(path, masses, sections=("policy", "rules")):
    store = TraceStore(path)
    for i, m in enumerate(masses):
        for t in range(3):
            add_turn(store, f"ep-{i}", t, m, sections)
    return store


def monitor_when_ready(client, **params):
    """Requests compute nothing; the background worker fills the read-map cache while we poll."""
    for _ in range(100):
        res = client.get("/api/monitor", params=params).json()
        if all(res[k]["progress"]["done"] == res[k]["progress"]["total"] for k in ("target", "reference")):
            return res
        time.sleep(0.05)
    raise TimeoutError("read maps were not computed")


def test_cached_readmap_matches_and_follows_new_turns(tmp_path):
    store = make_store(tmp_path / "store", [0.4])
    ep = tmp_path / "store" / "ep-0"
    cache = tmp_path / "cache"
    assert cached_readmap(ep, cache) == episode_readmap(ep)
    assert not (ep / "readmap.parquet").exists() and len(list(cache.iterdir())) == 1
    add_turn(store, "ep-0", 3, 0.4)
    assert {r["turn"] for r in cached_readmap(ep, cache)} == {0, 1, 2, 3}


def test_monitor_flags_reading_shifts_and_structural_changes(tmp_path):
    make_store(tmp_path / "ref", [0.30, 0.32, 0.34, 0.31, 0.33])
    make_store(tmp_path / "target", [0.32, 0.9])
    make_store(tmp_path / "renamed", [0.32], sections=("policy", "guidelines"))
    dash = Dashboard([tmp_path], cache_dir=tmp_path / "cache")
    ids = {s.path.name: sid for sid, s in dash.stores.items()}
    client = TestClient(create_app(dash))

    res = monitor_when_ready(client, target=ids["target"], ref=ids["ref"], threshold=3)
    alarms = {(a["episode"], a["section"], a["kind"]) for a in res["alarms"]}
    assert ("ep-1", "policy", "over-read") in alarms
    assert not any(a["episode"] == "ep-0" for a in res["alarms"])

    res = monitor_when_ready(client, target=ids["renamed"], ref=ids["ref"])
    kinds = {(a["section"], a["kind"]) for a in res["alarms"]}
    assert {("rules", "missing"), ("guidelines", "new")} <= kinds

    rm = client.get(f"/api/stores/{ids['target']}/episodes/ep-1/readmap").json()
    assert [s["name"] for s in rm["sections"]] == ["policy", "rules"] and len(rm["turns"]) == 3


def test_change_feed_reports_growing_episodes(tmp_path):
    store = make_store(tmp_path / "store", [0.4])
    dash = Dashboard([tmp_path / "store"], cache_dir=tmp_path / "cache")
    v = dash.version
    assert dash.changes(v)["changed"] == []
    add_turn(store, "ep-0", 3, 0.4)
    dash.rescan(sources=False)
    assert dash.changes(v)["changed"] == [{"store": next(iter(dash.stores)), "episode": "ep-0"}]


def test_evolution_names_candidates_and_merges_instances(tmp_path):
    import pandas as pd

    logs, jobs = tmp_path / "logs" / "run1", tmp_path / "jobs" / "run1"
    logs.mkdir(parents=True)
    (logs / "evolution_summary.jsonl").write_text(json.dumps(
        {"iteration": 1, "agent": "terse", "avg_pass_rate": 0.6, "base_system": "baseline", "per_task": {"t1": 1.0}}) + "\n")
    digest = {"evolve-terse-t2": [("rules", 10, 2.0), ("demo:a#1", 10, 1.0), ("demo:a#7@3", 30, 3.0)],
              "baseline-t2": [("rules", 10, 1.0)], "smoke-terse": [("rules", 10, 9.0)]}
    for job, rows in digest.items():
        (jobs / job).mkdir(parents=True)
        pd.DataFrame([{"section": s, "footprint_tok": t, "lift": v, "mass_share": 0.1} for s, t, v in rows]).to_parquet(
            jobs / job / "vitals_digest.parquet")
    dash = Dashboard([logs], cache_dir=tmp_path / "cache")
    res = TestClient(create_app(dash)).get(f"/api/runs/{next(iter(dash.runs))}").json()

    assert [(c["name"], c.get("baseline", False)) for c in res["candidates"]] == [("baseline", True), ("terse", False)]
    assert res["candidates"][1]["per_task"] == {"t1": 1.0} and res["run"]["score_scale"] == 1
    cells = {(r[0], r[1]): dict(zip(res["matrix"]["fields"], r, strict=True)) for r in res["matrix"]["cells"]}
    assert set(cells) == {("baseline", "rules"), ("terse", "rules"), ("terse", "demo:a")}
    assert cells[("terse", "demo:a")]["lift"] == 2.5 and cells[("terse", "demo:a")]["instances"] == 2
