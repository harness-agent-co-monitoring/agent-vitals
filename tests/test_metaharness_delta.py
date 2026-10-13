"""digest.delta on two synthetic stores: hunk classes, lifts, family pooling,
parent resolution by overlap, outcome pairing."""

import json

import numpy as np
import pandas as pd
import pytest

from vitals.addons.metaharness.digest import delta


def _episode(store, name, example, ids, attn, spans, reward, turns=1, extra=None):
    """One episode: prompt token ids + a cue of 2 tokens; `attn` is the
    per-position attention (prompt part); later turns append `extra` tokens
    and keep the prefix."""
    ep = store / f"ep-{name}-{example:04d}-c0" if isinstance(example, int) else store / f"ep-{name}-{example}"
    ep.mkdir(parents=True)
    rows, regs = [], []
    for t in range(turns):
        tail = list(extra or []) * t
        toks = list(ids) + [900, 901] + tail
        vec = list(attn) + [0.0, 0.0] + [0.01] * len(tail)
        for i, (tok, a) in enumerate(zip(toks, vec, strict=True)):
            rows.append({"turn": t, "idx": i, "token_id": tok, "token_str": str(tok), "region_id": "user_0", "attn": a})
        regions = [{"id": "user_0", "turn": t, "token_start": 0, "token_end": len(ids)}]
        regions += [{"id": f"span:{n}", "turn": t, "token_start": s, "token_end": e} for n, s, e in spans]
        regs.append({"turn": t, "regions": regions})
    pd.DataFrame(rows).to_parquet(ep / "attn_tokens.parquet", index=False)
    (ep / "regions.jsonl").write_text("\n".join(json.dumps(r) for r in regs) + "\n")
    (ep / "manifest.json").write_text(json.dumps({"meta": {"example_idx": example, "reward": reward, "call_idx": 0, "n_calls": 1}}))


@pytest.fixture()
def vitals_dir(tmp_path):
    """parent: [intro(4) | body A(6) | query(2)]; child replaces body A by
    body B(6) and keeps the rest verbatim. Attention: parent reads A at 0.02
    /token, child reads B at 0.005/token; intro/query identical."""
    intro, query = [1, 2, 3, 4], [50, 51]
    A, B = [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]
    a_p = [0.03] * 4 + [0.02] * 6 + [0.10] * 2
    a_c = [0.03] * 4 + [0.005] * 6 + [0.10] * 2
    spans = [("intro", 0, 4), ("body", 4, 10), ("query", 10, 12)]
    sp = tmp_path / "parent--ds" / "store"
    sc = tmp_path / "child--ds" / "store"
    so = tmp_path / "other--ds" / "store"    # unrelated design: nothing shared
    for ex in range(3):
        _episode(sp, "parent", ex, intro + A + query, a_p, spans, reward=1.0 if ex < 2 else 0.0)
        _episode(sc, "child", ex, intro + B + query, a_c, spans, reward=1.0)
        _episode(so, "other", ex, [70 + i for i in range(12)], a_p, spans, reward=0.0)
    return tmp_path


def test_hunk_classes_and_lifts(vitals_dir):
    md, H = delta.system_delta(vitals_dir, "child", "parent", "ds")
    assert set(H["class"]) == {"retained", "removed", "added"}
    ret_p = H[(H["class"] == "retained") & (H.side == "parent")]
    assert ret_p.n.sum() == 3 * 6                     # intro + query per example
    rm = H[H["class"] == "removed"]
    ad = H[H["class"] == "added"]
    assert rm.n.sum() == 18 and ad.n.sum() == 18
    assert set(rm.section) == {"body"} and set(ad.section) == {"body"}
    # digest-convention lift = mean attn * L, L = 14 positions (12 prompt + cue)
    # lift is against uniform reading of the context: / the vector's total (0.44 on the parent)
    _, cv, _, _, _ = delta._prompt(next((vitals_dir / "child--ds" / "store").glob("ep-*")))
    assert rm.lift.iloc[0] == pytest.approx(0.02 * 14 / 0.44)
    assert ad.lift.iloc[0] == pytest.approx(0.005 * 14 / float(cv[0].sum()))
    # rel: over the prompt's own mean; parent prompt mean = (0.12+0.12+0.2)/12
    assert rm.rel.iloc[0] == pytest.approx(0.02 / (0.44 / 12))
    assert "| removed (parent only) | 6 |" in md and "| added (child only) | 6 |" in md
    assert "3 paired examples" in md
    # outcomes paired on every episode: parent 2/3 -> child 3/3, example 2 flipped
    assert "parent 2/3 → child 3/3; ✗→✓ 1 (2)" in md


def test_retained_paired_delta_is_zero(vitals_dir):
    _, H = delta.system_delta(vitals_dir, "child", "parent", "ds")
    ret = H[H["class"] == "retained"]
    p = ret[ret.side == "parent"].sort_values(["example", "twin"]).lift.to_numpy()
    c = ret[ret.side == "child"].sort_values(["example", "twin"]).lift.to_numpy()
    # identical attention on identical tokens: equal up to each side's context share
    _, pv, _, _, _ = delta._prompt(next((vitals_dir / "parent--ds" / "store").glob("ep-*")))
    _, cv, _, _, _ = delta._prompt(next((vitals_dir / "child--ds" / "store").glob("ep-*")))
    np.testing.assert_allclose(p * float(pv[0].sum()), c * float(cv[0].sum()))


def test_family_pooling():
    assert delta._family("demo:盗窃#31") == "demo:*"
    assert delta._family("hallmarks:诈骗") == "hallmarks:*"
    assert delta._family("directive:2") == "directive:2"
    assert delta._family("label_inventory") == "label_inventory"
    assert delta._family("fb:noop_repair") == "fb:noop_repair"


def test_parent_resolution(vitals_dir):
    parent, ov, how = delta.resolve_parent(vitals_dir, "child", "ds")
    assert parent == "parent" and how.startswith("overlap")
    assert ov == pytest.approx(0.5)                   # 6 of 12 prompt tokens shared
    # declared hint wins, even as free text with decoration
    parent, _, how = delta.resolve_parent(vitals_dir, "child", "ds", hint="other (frontier, 80.0)")
    assert parent == "other" and how == "declared"
    # siblings excluded
    parent, _, _ = delta.resolve_parent(vitals_dir, "child", "ds", exclude=["parent"])
    assert parent == "other"


def test_tb_prefix_pooling(tmp_path):
    """Multi-turn: hunk lift pools over the turns that keep the prefix; a turn
    whose prefix was rewritten is skipped."""
    ids, attn = [1, 2, 3, 4], [0.1, 0.1, 0.2, 0.2]
    spans = [("scaffold", 0, 4)]
    sp = tmp_path / "p--tb2" / "store"
    sc = tmp_path / "c--tb2" / "store"
    _episode(sp, "p", "task", ids, attn, spans, 1.0, turns=3, extra=[77, 78])
    _episode(sc, "c", "task", [1, 2, 3, 9], attn, spans, 0.0, turns=3, extra=[77, 78])
    ids_, vecs, Ls, _, end = delta._prompt(next(sp.glob("ep-*")))
    assert end == 4 and sorted(vecs) == [0, 1, 2] and Ls == {0: 6, 1: 8, 2: 10}
    md, H = delta.system_delta(tmp_path, "c", "p", "tb2")
    ret = H[(H["class"] == "retained") & (H.side == "parent")]
    assert ret.n.sum() == 3
    # lift = mean(attn over [0,3)) * L averaged over turns = (0.4/3) * mean(6, 8, 10)
    assert ret.lift.iloc[0] == pytest.approx(np.mean([(0.4 / 3) * Ls[t] / float(vecs[t].sum()) for t in (0, 1, 2)]))
    assert "1 paired tasks" in md
