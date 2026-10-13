"""Linear-attention reading: the exact per-section split of delta-rule memories and its replay integration.

The math checks run on any device. The replay check serves a two-turn episode on the 0.8B CI model (a
Gated DeltaNet hybrid), replays it with linear reading and requires the per-section parts to add back to
the engine's own linear-layer output.
"""

from __future__ import annotations

import os
import shutil
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from vitals.engine.nodes import LinearMixerSpec

torch = pytest.importorskip("torch")

from vitals.engine.linear_memory import (  # noqa: E402  (needs torch)
    LayerInputs,
    advance,
    fp32_exact,
    reference_outputs,
    segments,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VENDOR = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin"
GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))


def _inputs(T, H, d, decay, seed=0, strength=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*s):
        return torch.randn(*s, generator=g).to(DEV)

    q = r(T, H, d)
    q = q / q.norm(dim=-1, keepdim=True) * d ** -0.5
    k = r(T, H, d)
    k = k / k.norm(dim=-1, keepdim=True)
    fade = (-torch.rand(T, H, generator=g) * 2 if decay == "head" else -torch.rand(T, H, d, generator=g) * 5) * strength
    return LayerInputs(q=q, k=k, v=r(T, H, d), g=fade.to(DEV), beta=torch.rand(T, H, generator=g).to(DEV))


@pytest.mark.parametrize("decay", ["head", "channel"])
def test_chunked_update_matches_the_token_loop(decay):
    T, H, d, R = 150, 6, 16, 4
    x = _inputs(T, H, d, decay)
    labels = torch.tensor([0] * 30 + [1] * 7 + [2] * 50 + [0] * 3 + [3] * 60, device=DEV)
    S0 = torch.randn(R, H, d, d, device=DEV) * 0.1
    want = torch.zeros(T, dtype=torch.bool, device=DEV)
    want[100:] = True
    with fp32_exact():
        ref_out, ref_S = reference_outputs(S0, x, labels, R, decay)
        S = S0.clone()
        res = advance(S, x, labels, R, decay, want=want, chunk=16)
    assert torch.allclose(S, ref_S, rtol=0, atol=1e-5 * ref_S.abs().max())
    full = ref_out.sum(1)
    num = torch.einsum("trhd,thd->tr", ref_out, full)[100:]
    assert torch.allclose(res.num, num, rtol=0, atol=1e-5 * num.abs().max())
    assert torch.allclose(res.num.sum(1) / res.den, torch.ones(len(num), device=DEV), atol=1e-5)  # shares sum to 1


@pytest.mark.parametrize("decay", ["head", "channel"])
def test_parts_add_up_to_the_unsplit_memory(decay):
    T, H, d, R = 90, 4, 8, 3
    x = _inputs(T, H, d, decay, seed=1)
    labels = torch.tensor([0] * 20 + [1] * 40 + [2] * 30, device=DEV)
    with fp32_exact():
        parts, S_parts = reference_outputs(torch.zeros(R, H, d, d, device=DEV), x, labels, R, decay)
        whole, S_whole = reference_outputs(torch.zeros(1, H, d, d, device=DEV), x,
                                           torch.zeros(T, dtype=torch.long, device=DEV), 1, decay)
    assert torch.allclose(parts.sum(1), whole[:, 0], atol=1e-5)
    assert torch.allclose(S_parts.sum(0), S_whole[0], atol=1e-5)


def test_segments_and_retained_never_exceeds_one():
    assert segments(torch.tensor([2, 2, 0, 0, 0, 1, 2])) == [(0, 2), (2, 5), (5, 6), (6, 7)]
    T, H, d = 80, 4, 8
    x = _inputs(T, H, d, "channel", seed=2, strength=0.01)   # mild fades: the memory must not vanish
    labels = torch.tensor([0] * 20 + [1] * 60, device=DEV)
    S = torch.zeros(2, H, d, d, device=DEV)
    with fp32_exact():
        res = advance(S, x, labels, 2, "channel", chunk=16)
    written = res.seg_energy[0]                          # slot 0 right after its last token
    now = float(torch.linalg.vector_norm(S[0]) ** 2)     # after 60 more tokens of slot 1
    assert 0 < now <= written * (1 + 1e-5)               # fading and erasing only shrink a memory


def _spec(n_layers=1, decay="head"):
    return LinearMixerSpec(kind="gdn", layers=tuple(range(n_layers)), n_k_heads=2, n_v_heads=2, d_k=4, d_v=4,
                           decay=decay, g_node="g-{i}", beta_node="b-{i}", out_node="o-{i}", qkv_node="qkv-{i}")


def test_section_labels_partition_and_folding():
    from vitals.probes.linear_reading import OLDER_REGIONS, LinearReading, _Track

    lr = LinearReading.__new__(LinearReading)
    lr.group_sep = "/"

    def reg(rid, s, e):
        return SimpleNamespace(id=rid, token_start=s, token_end=e, meta={})

    regions = [reg("sys_0", 0, 10), reg("user_0", 10, 20), reg("cue_0", 20, 22),
               reg("span:policy/a", 2, 8), reg("span:policy/a/b", 4, 6), reg("span:task", 12, 15)]
    lab = lr.section_labels(regions, 22, 3)
    assert lab[:2] == ["region:sys_0"] * 2 and lab[2:8] == ["policy"] * 6       # grouped by "/"
    assert lab[12:15] == ["task"] * 3 and lab[15] == "region:user_0" and lab[-3:] == ["output"] * 3

    trk = _Track(_spec(), max_slots=3)
    trk.mem.S = torch.zeros(3, 2, 4, 4, device=DEV)
    for i, name in enumerate(["region:a", "region:b", "span"]):
        trk.mem.S[trk.slot(name, clock=i)] = float(i + 1)
    total = trk.mem.S[:trk.mem.n_slots].sum(0).clone()
    r = trk.slot("region:c", clock=9)                    # full: the two old regions fold into one bucket
    assert OLDER_REGIONS in trk.ids and trk.ids["region:a"] == trk.ids["region:b"] == trk.ids[OLDER_REGIONS]
    assert trk.names[r] == "region:c" and float(trk.mem.S[r].abs().sum()) == 0
    assert torch.equal(trk.mem.S[:trk.mem.n_slots].sum(0), total)             # folding is exact


@pytest.mark.skipif(not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS or DEV != "cuda",
                    reason="pinned engine build, 0.8B GGUF or CUDA not available")
def test_replay_linear_reading_is_exact(tmp_path):
    from fastapi.testclient import TestClient

    from vitals.replay.episode import replay_store
    from vitals.server.app import build_state, create_app

    store = tmp_path / "store"
    state = build_state(GGUFS[0], store_dir=str(store), n_ctx=4096, store_messages=True)
    try:
        client = TestClient(create_app(state))
        msgs = [{"role": "system", "content": "Answer briefly. Always name the colour first."},
                {"role": "user", "content": "Name three primary colors."}]
        for _ in range(2):
            r = client.post("/v1/chat/completions", json={
                "model": "m", "temperature": 0.0, "max_tokens": 32, "messages": msgs,
                "vitals": {"episode_id": "ep-lin"}})
            assert r.status_code == 200
            msgs = msgs + [{"role": "assistant", "content": r.json()["choices"][0]["message"]["content"] or "ok"},
                           {"role": "user", "content": "And three secondary colors?"}]
        state.store.flush("ep-lin")
    finally:
        state.backend.close()

    out = replay_store(GGUFS[0], store, n_ctx=4096, channels=("logits",), linear=True,
                       linear_opts={"check": True})
    assert out[0]["turns"] == 2 and "linear" in out[0]["channels"], out
    ep = store / "ep-lin"
    lin = pd.read_parquet(ep / "linear_reading.parquet")
    steps = pd.read_parquet(ep / "linear_steps.parquet")
    toks = pd.concat([pd.read_parquet(p) for p in sorted(ep.glob("tokens-*.parquet"))])
    for t, g in lin.groupby("turn"):
        assert g.check_rel_err.iloc[0] < 1e-4, f"turn {t}: parts do not add up to the engine output"
        assert abs(g.mem_share.sum() - 1) < 1e-4 and abs(g.read.sum() - 1) < 1e-4
        prompt = g[g.kind != "output"]
        assert (prompt.retained.dropna() <= 1 + 1e-4).all()
        n_gen = int((toks[(toks.turn == t)].region_id == "gen").sum())
        assert (steps.turn == t).sum() == n_gen
    shutil.rmtree(store)
