"""Residual probes localized and confirmed: on the 0.8B CI model, a no-op ablation reproduces the probe score
exactly, the section decomposition adds up, and the base score matches an independent capture."""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import numpy as np
import pytest

from vitals.monitor.probes import LinearProbe, auroc, decompose, probe_normal, train_probe

torch = pytest.importorskip("torch")
VENDOR = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin"
GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))


def test_probe_fit_save_load_and_decompose(tmp_path):
    rng = np.random.default_rng(0)
    H = rng.normal(size=(80, 16))
    y = (H[:, 3] > 0).astype(int)
    p = LinearProbe.fit("toy", 5, H, y, C=1.0)
    assert ((p.score(H) > 0) == y).mean() > 0.9
    p.save(tmp_path / "p.json")
    q = LinearProbe.load(tmp_path / "p.json")
    assert np.allclose(q.score(H), p.score(H))
    writes = rng.normal(size=(4, 16))
    d = decompose(p, H[0], writes, ["a", "b", "c", "d"])
    assert abs(d["section_total"] + d["remainder"] - float(H[0] @ p.w_eff)) < 1e-9
    assert [c for _, c in d["sections"]] == sorted((c for _, c in d["sections"]), reverse=True)


@pytest.mark.skipif(not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS or not torch.cuda.is_available(),
                    reason="pinned engine build, 0.8B GGUF or CUDA not available")
def test_ablation_scores_and_explain_flag():
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.monitor.probes import ablation_scores, explain_flag
    from vitals.probes.ablation import spans_from_labels
    from vitals.probes.linear_reading import section_labels
    from vitals.probes.section_writes import SectionWrites

    be = LlamaCppBackend(GGUFS[0], n_ctx=4096, n_seq_max=2)
    try:
        msgs = [{"role": "system", "content": "You are terse. Always answer with the colour blue. Be polite."},
                {"role": "user", "content": "What colour is the sky at noon? Answer in one word."}]
        ctx, prompt = be.render_turn(msgs, spans=[{"name": "rule", "text": "Always answer with the colour blue."}])
        L = len(prompt)
        labels = section_labels(ctx.regions, L, 0)
        layer = be.node_map.n_layers // 2
        n_embd = be.engine.n_embd
        rng = np.random.default_rng(1)
        probe = LinearProbe("toy", layer, np.zeros(n_embd), np.ones(n_embd), rng.normal(size=n_embd) / np.sqrt(n_embd), 0.0, 0.0)
        spans = spans_from_labels(labels[:L - 1])
        rule = spans["rule"]
        res = ablation_scores(be, probe, prompt, L - 1, {"rule": rule, "noop": rule},
                              replacements={"noop": [prompt[s:e] for s, e in rule]})
        h = SectionWrites(be, GGUFS[0], layers=[layer]).capture(prompt, labels, [L - 1])["positions"][L - 1]["h"][0]
        ex = explain_flag(be, GGUFS[0], probe, prompt, labels, L - 1, confirm_top=1, exclude=("cue",))
    finally:
        be.close()
    assert res["without"]["noop"] == res["base"]
    assert res["without"]["rule"] != res["base"]
    assert abs(res["base"] - float(probe.score(h)[0])) < 0.05 * max(1.0, abs(res["base"]))
    assert ex["contributions"] and len(ex["confirmed"]) == 1 and ex["claims"]


def test_train_probe_threshold_and_normal():
    """The label comes from what one section ("rule") writes along a hidden direction; the trained probe carries a
    threshold near the false-alarm budget and a section normal, and the rule stands out by z on positives."""
    rng = np.random.default_rng(0)
    d, n = 32, 400
    v = rng.normal(size=d)
    sections = ["rule", "user:1", "user:2", "tool:get", "output"]
    tokens = [20, 40, 60, 80, 10]
    H, splits, y = [], [], []
    for i in range(n):
        lab = i % 2
        W = rng.normal(scale=0.3, size=(5, d))
        W[0] += (1.5 if lab else 0.0) * v / np.linalg.norm(v)
        H.append(W.sum(0) + rng.normal(scale=0.3, size=d))
        splits.append((W, sections, tokens))
        y.append(lab)
    groups = np.arange(n) // 8
    probe, oof = train_probe("toy", 3, np.array(H), np.array(y), groups, splits=splits, budget=0.1)
    assert auroc(y, oof) > 0.8 and probe.meta["oof_auroc"] > 0.8
    neg = oof[np.array(y) == 0]
    assert abs(np.mean(neg >= probe.threshold) - 0.1) < 0.05
    normal = probe_normal(probe)
    assert {"rule", "user:earlier", "user:latest", "tool:get"} <= set(normal.median)
    from vitals.monitor.measures import split_report
    tops = [split_report(probe, H[i], *splits[i], normal=normal)["rows"][0]["section"] for i in range(1, 40, 2)]
    assert np.mean([t == "rule" for t in tops]) > 0.7


def test_probe_cli_train_and_score(tmp_path, capsys):
    """vitals probe train / score on a synthetic ledger (the files ledger_episode writes)."""
    import json

    import pandas as pd

    from vitals.cli import main
    from vitals.monitor.ledger import ledger_dir

    rng = np.random.default_rng(0)
    d, layers = 24, np.array([3, 7])
    v = rng.normal(size=d)
    store = tmp_path / "store"
    store.mkdir()
    root = ledger_dir(tmp_path / "ledger", store)
    sections = np.array(["region:sys_0", "rule", "user:1", "tool_result:get#1", "output"])
    rows = []
    for e in range(12):
        (root / f"ep{e}").mkdir(parents=True)
        for t in range(6):
            lab = int((e + t) % 3 == 0)
            W = rng.normal(scale=0.3, size=(2, 5, d))
            W[:, 1] += 3.0 * lab * v / np.linalg.norm(v)
            res = W.sum(1) + rng.normal(scale=0.3, size=(2, d))
            at = np.abs(rng.normal(size=(4, 5)))
            at[:, 1] *= 1 + 3 * lab                                  # the rule is read more when it binds
            at /= at.sum(1, keepdims=True)
            np.savez(root / f"ep{e}" / f"{t}.npz", sections=sections, tokens=np.array([5, 20, 30, 40, 10]),
                     residual_layers=layers, residual_at=res.astype(np.float16), writes=W.astype(np.float16),
                     at=at.astype(np.float16), span=at.astype(np.float16),
                     meta=np.array(json.dumps({"episode": f"ep{e}", "turn": t})))
            rows.append({"episode": f"ep{e}", "turn": t, "label": lab})
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
    out = tmp_path / "probe.json"
    assert main(["probe", "train", "--store", str(store), "--out-dir", str(tmp_path / "ledger"), "--labels",
                 str(tmp_path / "labels.csv"), "--name", "toy", "--layer", "7", "--out", str(out)]) == 0
    probe = LinearProbe.load(out)
    assert probe.meta["oof_auroc"] > 0.8 and "rule" in probe.meta["normal"]["median"]
    assert set(probe.meta["read_normal"]) == {"decision", "reasoning"} and "rule" in probe.meta["read_normal"]["decision"]["median"]
    assert main(["probe", "score", "--store", str(store), "--out-dir", str(tmp_path / "ledger"), "--probe", str(out)]) == 0
    recs = [json.loads(x) for x in (root / "probe_toy.jsonl").read_text().splitlines()]
    assert len(recs) == 72 and any(r["flagged"] for r in recs)
    flagged_pos = [r for r in recs if r["flagged"] and (int(r["episode"][2:]) + r["turn"]) % 3 == 0]
    assert np.mean([r["top"][0]["section"] == "rule" for r in flagged_pos]) > 0.7
    # read z for the behavior beside the probe rows: the rule is read more than in decisions without it
    assert np.median([r["top"][0]["read_z_decision"] for r in flagged_pos if r["top"][0]["section"] == "rule"]) > 1
    assert "flagged" in capsys.readouterr().out
