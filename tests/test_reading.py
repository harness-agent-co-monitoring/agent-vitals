"""vitals.analysis.reading on a synthetic episode with known geometry."""

import json

import numpy as np
import pytest

from vitals.analysis.reading import episode_coupling, episode_turns, needed_but_unread
from vitals.core.schema import TokenMetrics
from vitals.core.store import TraceStore


@pytest.fixture()
def ep(tmp_path):
    """8 turns; entropy ramps up while harness reading ramps down (raw
    abandonment by construction). Two span sections: 'pin' loses lift with
    turn, 'other' holds steady."""
    store = TraceStore(tmp_path)
    eid = "ep-test"
    rng = np.random.default_rng(0)
    for t in range(8):
        L = 100 + 50 * t
        u = 0.2 + 0.1 * t
        for s in range(20):
            store.append(TokenMetrics(
                episode_id=eid, turn=t, step=s, token_id=1,
                ent=float(u + rng.normal(0, 0.01)), m2=0.5, js=0.0,
                fork=s < 4, region_id="gen"))
        store.flush(eid)
        vec = np.full(L, 1.0 / L)
        vec[:2] = 0.05                            # attention sink at the front
        vec[10:30] = (8 - t) * 0.4 / L            # harness body decays w/ turn
        pin_lift = 8.0 if t < 4 else 0.2          # 'pin' span dies under load
        vec[12:16] = pin_lift / L
        vec[30:40] = 2.0 / L                      # 'other' span: steady 2x uniform
        d = tmp_path / eid
        d.mkdir(exist_ok=True)
        with (d / "attn_received.jsonl").open("a") as f:
            f.write(json.dumps({"turn": t, "vec": [float(x) for x in vec]}) + "\n")
        regions = [
            {"id": "sys_0", "token_start": 0, "token_end": 30},
            {"id": "user_0", "token_start": 30, "token_end": 60},
            {"id": "span:pin", "token_start": 12, "token_end": 16},
            {"id": "span:other", "token_start": 30, "token_end": 40},
        ]
        with (d / "regions.jsonl").open("a") as f:
            f.write(json.dumps({"turn": t, "regions": regions}) + "\n")
    return tmp_path / eid


def test_turn_table_and_sink_exclusion(ep):
    turns = episode_turns(ep)
    assert len(turns) == 8
    # sink positions excluded: W_H must be far below the 0.05 sink values
    assert all(r["W_H"] < 0.05 for r in turns)
    assert turns[0]["L"] == 100 and turns[7]["L"] == 450
    assert set(turns[0]["sections"]) == {"pin", "other"}


def test_coupling_signs_and_adrift(ep):
    c = episode_coupling(ep)
    assert c["n_turns"] == 8
    assert c["chi"] is not None and c["chi"] < 0, "raw abandonment by construction"
    assert c["chi_spans"] is not None and c["chi_spans"] < 0, \
        "pin lift decays while need rises; 'other' flat -> mean span lift falls"
    assert c["regime"] == "abandonment"
    late = {r["turn"] for r in c["turns"]} - set(range(4))
    assert set(c["adrift_turns"]) <= late and c["adrift_turns"], \
        "adrift turns are the late high-need low-read ones"


def test_needed_but_unread_ranks_dead_pin_first(ep):
    nbu = needed_but_unread(ep)
    assert [d["section"] for d in nbu][0] == "pin"
    assert nbu[0]["need_weighted_lift"] < nbu[1]["need_weighted_lift"]


def test_consult_contrast_tail_form():
    from vitals.analysis.reading import consult_contrast, consult_ratio
    rng = np.random.default_rng(0)
    ents = rng.uniform(0, 2, 400)
    shares = 0.05 + 0.10 * (ents > np.quantile(ents, 0.9))     # read 3x harder only at the top decile
    cr, cc = consult_ratio(ents, shares), consult_contrast(ents, shares)
    assert cc == pytest.approx(3.0)                              # the contrast sees the extremes
    assert 1.0 < cr < 1.5                                        # the ratio compresses them
    assert consult_contrast(ents[:20], shares[:20]) is None      # too few steps per side
