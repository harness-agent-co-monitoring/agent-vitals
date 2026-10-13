"""Counterexamples and invariants for optional section diagnostics."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vitals.analysis.concentration import SectionConcentration, read_concentration, write_concentration
from vitals.analysis.diagnostics import attention_diagnostics, episode_attention_diagnostics
from vitals.analysis.reading import consult_contrast, consult_ratio
from vitals.analysis.readmap import episode_readmap
from vitals.core.schema import TokenMetrics
from vitals.core.store import TraceStore


def diagnostics(h, a, c):
    return attention_diagnostics(h, a, c, length=64, section_length=32, min_tokens=2, min_side=1)


def test_context_return_without_relative_selection():
    h, c = np.array([1., 3.]), np.array([.1, .5])
    a = .2 * c
    d = diagnostics(h, a, c)
    assert consult_ratio(h, a, min_tokens=2) == pytest.approx(4 / 3)
    assert d["context_consult_ratio"] == pytest.approx(4 / 3)
    assert d["relative_consult_ratio"] == pytest.approx(1)
    assert d["relative_consult_contrast"] == pytest.approx(1)


def test_decomposition_reconstructs_both_original_measures():
    rng = np.random.default_rng(17)
    h, c, r = rng.uniform(.01, 1, (3, 320))
    a = c * r
    d = diagnostics(h, a, c)
    assert d["context_consult_ratio"] * d["relative_consult_ratio"] == pytest.approx(consult_ratio(h, a))
    assert d["context_consult_contrast"] * d["relative_consult_contrast"] == pytest.approx(consult_contrast(h, a))


def test_zero_covariance_does_not_mean_no_relationship():
    h = np.tile([1., 2., 3.], 10)
    a = np.tile([.2, .1, .2], 10)
    assert consult_ratio(h, a) == pytest.approx(1)
    assert len(set(a)) > 1


def test_both_lift_averages_and_zero_context_policy():
    d = diagnostics([1, 3], [.08, .18], [.1, .9])
    assert d["lift_equal_step"] == pytest.approx(1)
    assert d["lift_mass_weighted"] == pytest.approx(.52)
    d = diagnostics([1, 3], [0, .18], [0, .9])
    assert d["lift_equal_step"] is None
    assert d["lift_mass_weighted"] == pytest.approx(.4)
    assert d["n_context_steps"] == 1
    d = diagnostics([0, 0], [0, 0], [0, 0])
    assert d["context_consult_ratio"] is None
    assert d["relative_consult_ratio"] is None
    assert d["lift_mass_weighted"] is None


@pytest.mark.parametrize("h,a,c", [([1], [1, 2], [1]), ([1, np.nan], [.1, .2], [.5, .5]),
                                    ([1, 2], [.6, .2], [.5, .5]), ([1, 2], [.1, .2], [-.1, .5])])
def test_invalid_inputs_fail_explicitly(h, a, c):
    with pytest.raises(ValueError):
        diagnostics(h, a, c)


def concentration(rows):
    acc = SectionConcentration(["word"] * len(rows[0]),
                               [{"id": "span:memory", "token_start": 0, "token_end": len(rows[0])}])
    for row in rows:
        acc.update(row)
    return acc.finish(0)[0]


def test_diffuse_fixed_and_moving_hotspots():
    diffuse = np.full((4, 32), 1 / 32)
    fixed = np.zeros((4, 32))
    fixed[:, :8] = 1 / 8
    moving = np.zeros((4, 32))
    for i in range(4):
        moving[i, i * 8:(i + 1) * 8] = 1 / 8
    for rows, pooled, step, gap in [(diffuse, 1, 1, 0), (fixed, 4, 4, 0), (moving, 1, 4, .75)]:
        d = concentration(rows)
        for weight in ("equal_step", "mass_weighted"):
            assert d[f"peak_pooled_{weight}"] == pytest.approx(pooled)
            assert d[f"peak_step_{weight}"] == pytest.approx(step)
            assert d[f"peak_gap_{weight}"] == pytest.approx(gap)
        assert d["focus_js_bits"] == pytest.approx(2 if gap else 0, abs=1e-12)
        assert d["focus_pooled_entropy_bits"] == pytest.approx(3 if pooled == 4 else 5)
        assert d["focus_step_entropy_bits"] == pytest.approx(5 if step == 1 else 3)


def test_focus_js_matches_dense_weighted_formula_and_permutation():
    rng = np.random.default_rng(49)
    raw = rng.dirichlet(np.ones(45), size=31) * rng.uniform(.01, 1, (31, 1))
    regions = [{"id": "span:subset", "token_start": 5, "token_end": 24}]
    def capture(rows):
        acc = SectionConcentration(["word"] * 45, regions)
        for r in rows:
            acc.update(r)
        return acc.finish(0)[0]
    d = capture(raw)
    a = raw[:, 5:24].sum(axis=1)
    r, omega = raw[:, 5:24] / a[:, None], a / a.sum()
    def entropy(p):
        return -(p[p > 0] * np.log2(p[p > 0])).sum()
    pooled = entropy(np.average(r, axis=0, weights=omega))
    steps = np.dot(omega, [entropy(x) for x in r])
    assert d["focus_js_bits"] == pytest.approx(pooled - steps)
    assert d["focus_weight_entropy_bits"] == pytest.approx(entropy(omega))
    assert capture(raw[::-1])["focus_js_bits"] == pytest.approx(d["focus_js_bits"])
    assert capture(raw * .03)["focus_js_bits"] == pytest.approx(d["focus_js_bits"])
    assert d["focus_status"] == "complete"
    assert d["peak_status"] == "ineligible"


def test_focus_js_uses_all_tokens_and_handles_zero_mass():
    regions = [{"id": "span:punctuation", "token_start": 0, "token_end": 2}]
    acc = SectionConcentration(["!", "?"], regions)
    acc.update([.9, 0])
    acc.update([0, .1])
    acc.update([0, 0])
    d = acc.finish(0)[0]
    assert d["focus_js_bits"] == pytest.approx(-.9*np.log2(.9)-.1*np.log2(.1))
    assert d["focus_step_entropy_bits"] == pytest.approx(0)
    assert d["focus_n_attended_steps"] == 2
    zero = concentration(np.zeros((2, 32)))
    assert zero["focus_status"] == "no_attention"
    assert zero["focus_js_bits"] is None
    assert concentration(np.full((1, 32), 1/32))["focus_js_bits"] == pytest.approx(0, abs=1e-12)


def test_focus_js_retains_tiny_section_mass():
    acc = SectionConcentration(["word"] * 3,
                               [{"id": "span:tiny", "token_start": 1, "token_end": 3}])
    acc.update([.9, 1e-25, 0])
    acc.update([.9, 0, 1e-25])
    d = acc.finish(0)[0]
    assert d["focus_n_attended_steps"] == 2
    assert d["focus_js_bits"] == pytest.approx(1)


def test_concentration_matches_dense_formula_with_variable_mass():
    rng = np.random.default_rng(2)
    x = rng.dirichlet(np.ones(40), size=20)
    c = rng.uniform(.01, 1, size=20)
    d = concentration(x * c[:, None])
    for weight, weights in [("equal_step", np.ones(20) / 20), ("mass_weighted", c / c.sum())]:
        pooled = 40 * np.sort(np.average(x, axis=0, weights=weights))[-8:].mean()
        step = 40 * np.average(np.sort(x, axis=1)[:, -8:].mean(axis=1), weights=weights)
        assert d[f"peak_pooled_{weight}"] == pytest.approx(pooled)
        assert d[f"peak_step_{weight}"] == pytest.approx(step)
        assert step >= pooled


def test_concentration_content_mask_short_sections_and_zero_mass():
    regions = [{"id": "span:long", "token_start": 0, "token_end": 40},
               {"id": "span:short", "token_start": 1, "token_end": 12}]
    acc = SectionConcentration(["!"] + ["word"] * 39, regions)
    row = np.full(40, .2 / 39)
    row[0] = .8
    acc.update(row)
    values = acc.finish(0)
    assert values[0]["peak_pooled_mass_weighted"] == pytest.approx(40 * .2 / 39)
    assert values[1]["peak_status"] == "ineligible"
    assert values[1]["peak_step_mass_weighted"] is None
    acc.update(np.zeros(40))
    assert acc.finish(0)[0]["peak_step_equal_step"] is None
    assert acc.finish(0)[0]["peak_step_mass_weighted"] == pytest.approx(40 * .2 / 39)
    zero = concentration(np.zeros((2, 32)))
    assert zero["peak_gap_mass_weighted"] is None


def make_episode(tmp_path):
    store, eid = TraceStore(tmp_path), "ep-diagnostic"
    ep = tmp_path / eid
    regs = [{"id": "sys_0", "token_start": 0, "token_end": 48},
            {"id": "span:memory", "token_start": 0, "token_end": 32}]
    for step in range(32):
        c = .1 if step < 16 else .5
        store.append(TokenMetrics(episode_id=eid, turn=0, step=step, token_id=2,
                                  ent=1 if step < 16 else 3, m2=.1, js=0, fork=False,
                                  region_id="gen", attn_share=[.9 * c, .2 * c, 1 - c]))
    store.flush(eid)
    vec = np.r_[np.full(32, .06 / 32), np.full(32, .24 / 32)]
    (ep / "regions.jsonl").write_text(json.dumps({"turn": 0, "regions": regs}) + "\n")
    store.append_attn_received(eid, 0, vec)
    store.append_attn_tokens(eid, 0, range(64), ["word"] * 64, regs, vec)
    store.write_messages(eid, 0, [{"role": "raw", "content": "example"}])
    return ep


def test_report_preserves_existing_fields_and_includes_unassigned_context(tmp_path):
    ep = make_episode(tmp_path)
    original = episode_readmap(ep)
    extended = episode_readmap(ep, diagnostics=True)
    assert [{k: v for k, v in r.items() if k != "diagnostics"} for r in extended] == original
    d = extended[0]["diagnostics"]
    assert d["status"] == "complete"
    assert d["context_attention"] == pytest.approx(.3)
    assert d["relative_consult_ratio"] == pytest.approx(1)
    assert d["lift_mass_weighted"] == pytest.approx(original[0]["lift"], abs=1e-3)
    assert d["peak_status"] == "not_captured"
    assert d["peak_step_mass_weighted"] is None
    assert d["focus_status"] == "not_captured"
    json.dumps(extended, allow_nan=False)


@pytest.mark.parametrize("legacy", [False, True])
def test_incomplete_or_legacy_vectors_are_not_imputed(tmp_path, legacy):
    ep = make_episode(tmp_path)
    tk = pd.read_parquet(ep / "tokens-00000.parquet")
    tk.at[0, "attn_share"] = list(tk.at[0, "attn_share"][:-1]) if legacy else None
    d = episode_attention_diagnostics(ep, tokens=tk)[0, "memory"]
    assert d["status"] == "missing_complete_attention"
    assert d["context_consult_ratio"] is None


def test_concentration_turn_replacement_and_report_metadata(tmp_path):
    ep = make_episode(tmp_path)
    acc = SectionConcentration(["word"] * 64, [{"id": "span:memory", "token_start": 0, "token_end": 32}])
    for _ in range(32):
        acc.update(np.full(64, .3 / 64))
    write_concentration(ep, 0, acc.finish(0))
    assert episode_readmap(ep, diagnostics=True)[0]["diagnostics"]["peak_step_mass_weighted"] == pytest.approx(1)
    write_concentration(ep, 1, acc.finish(1))
    write_concentration(ep, 0, acc.finish(0))
    assert len(read_concentration(ep)) == 2
    assert episode_readmap(ep, diagnostics=True)[0]["diagnostics"]["focus_js_bits"] == pytest.approx(0, abs=1e-12)
    write_concentration(ep, 0, [], sidecar=True)
    assert len(read_concentration(ep)) == 2
    stale = acc.finish(0)
    stale[0]["n_gen"] = 31
    write_concentration(ep, 0, stale)
    assert episode_readmap(ep, diagnostics=True)[0]["diagnostics"]["peak_status"] == "capture_mismatch"
    assert episode_readmap(ep, diagnostics=True)[0]["diagnostics"]["focus_status"] == "capture_mismatch"
    write_concentration(ep, 0, [])
    assert set(read_concentration(ep)) == {(1, "memory")}


def test_missing_output_step_is_not_silently_dropped(tmp_path):
    ep = make_episode(tmp_path)
    tk = pd.read_parquet(ep / "tokens-00000.parquet").drop(index=5)
    d = episode_attention_diagnostics(ep, tokens=tk)[0, "memory"]
    assert d["status"] == "missing_output_steps"
    assert d["lift_equal_step"] is None


def test_cli_reports_diagnostics(tmp_path, capsys):
    from vitals.cli import main

    make_episode(tmp_path)
    assert main(["report", "--store", str(tmp_path), "--diagnostics", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["diagnostics"]["relative_consult_ratio"] == pytest.approx(1)
    assert main(["report", "--store", str(tmp_path), "--diagnostics"]) == 0
    assert "relative CR" in capsys.readouterr().out


def test_replay_accumulates_exact_peaks_and_invalidates_old_capture(tmp_path, monkeypatch):
    from vitals.core.schema import Region, RegionKind
    from vitals.replay.episode import replay_episode

    ep = make_episode(tmp_path)
    monkeypatch.setenv("VITALS_REPLAY_HEAD_MEAN", "0")

    class Engine:
        n_past = 0
        n_ctx_per_seq = 1024
        _ctx_opts = {"n_seq_max": 1, "n_batch": 64}

        def reset_context(self, **kwargs):
            self.n_past = 0

        def set_capture(self, *args, **kwargs):
            pass

        def decode(self, tokens, **kwargs):
            start = self.n_past
            self.n_past += len(tokens)
            self.rows = np.zeros((len(tokens), self.n_past))
            for j, pos in enumerate(range(start, self.n_past)):
                group = (pos - 63) % 4
                self.rows[j, group * 8:(group + 1) * 8] = (1 if pos < 64 else .6) / 8
                if pos >= 64:
                    self.rows[j, pos] = .4

    class Session:
        def __init__(self, *args):
            pass

        def harvest(self, engine):
            return {}, {"rows": engine.rows}

    monkeypatch.setattr("vitals.engine.capture.plan_capture", lambda *a: SimpleNamespace(names=set(), modes={}))
    monkeypatch.setattr("vitals.engine.capture.CaptureSession", Session)
    monkeypatch.setattr("vitals.replay.episode.layer_head_mean", lambda a: a["rows"])
    region = Region(id="span:memory", kind=RegionKind.SYSTEM, role="system", token_start=0, token_end=32)
    backend = SimpleNamespace(engine=Engine(), node_map=None, detokenize=lambda t: "word",
                              render_turn=lambda *a, **k: (SimpleNamespace(regions=[region]), list(range(64))))
    # An old attention channel without diagnostics must be replayed even without --force.
    result = replay_episode(backend, tmp_path, ep.name, channels=("attention",), section_diagnostics=True,
                            spans=[{"name": "memory", "text": "example"}], sparse_attn_k=2, chunk=4)
    assert result["turns"] == 1
    d = read_concentration(ep)[0, "memory"]
    assert d["peak_step_mass_weighted"] == pytest.approx(8)
    assert d["peak_gap_equal_step"] == pytest.approx(.75)
    assert len(pd.read_parquet(ep / "attn_sparse.parquet").pos.iloc[0]) == 2
    assert replay_episode(backend, tmp_path, ep.name, channels=("attention",),
                          section_diagnostics=True)["turns"] == 0
    # An older peak-only capture must be refreshed rather than silently
    # reported as a complete JS capture.
    from vitals.analysis.concentration import FOCUS_FIELDS

    legacy = {k: v for k, v in d.items() if k not in FOCUS_FIELDS and k != "focus_status"}
    legacy["version"] = 1
    write_concentration(ep, 0, [legacy])
    assert replay_episode(backend, tmp_path, ep.name, channels=("attention",),
                          section_diagnostics=True, sparse_attn_k=0)["turns"] == 1
    assert read_concentration(ep)[0, "memory"]["focus_status"] == "complete"
    replay_episode(backend, tmp_path, ep.name, channels=("attention",), force=True, sparse_attn_k=0)
    assert read_concentration(ep) == {}


def test_legacy_capture_stays_readable_without_fabricating_focus(tmp_path):
    ep = make_episode(tmp_path)
    legacy = pd.DataFrame([dict(version=1, turn=0, section="memory", token_start=0,
                                token_end=32, L=64, n_gen=32, peak_k=8, peak_status="complete",
                                peak_step_mass_weighted=1.)])
    legacy.to_parquet(ep / "section_diagnostics.parquet", index=False)
    d = episode_readmap(ep, diagnostics=True)[0]["diagnostics"]
    assert d["peak_step_mass_weighted"] == 1.
    assert d["focus_js_bits"] is None
    assert d["focus_status"] == "not_captured"
    assert d["concentration_version"] == 1
