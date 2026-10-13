"""Teacher-forced attention replay vs live capture on the same episode.

Serves one episode with full diagnostic-attention capture (the ground truth),
then replays it in sidecar mode and checks the replayed received-attention
vector and per-token region shares match the live-captured ones. Forced
tokens = the exact logged trajectory, so agreement is limited only by kernel
noise (GEMM prefill path vs stepwise decode)."""

import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

VENDOR = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin"
GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))
needs_model = pytest.mark.skipif(
    not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS,
    reason="pinned engine build or 0.8B GGUF not available",
)


@pytest.fixture(scope="module")
def captured_store(tmp_path_factory):
    from fastapi.testclient import TestClient

    from vitals.server.app import build_state, create_app

    store_dir = tmp_path_factory.mktemp("replay-store")
    state = build_state(GGUFS[0], store_dir=str(store_dir), n_ctx=2048,
                        diagnostic_attention=True, capture_attention=True,
                        store_messages=True)
    try:
        client = TestClient(create_app(state))
        r = client.post("/v1/chat/completions", json={
            "model": "m", "temperature": 0.0, "max_tokens": 48,
            "messages": [{"role": "user", "content": "Name three primary colors."}],
            "vitals": {"episode_id": "ep-replaytest"},
        })
        assert r.status_code == 200
        state.store.flush("ep-replaytest")
    finally:
        state.backend.close()
    return store_dir


@needs_model
def test_replay_matches_live_capture(captured_store):
    from vitals.replay.episode import replay_store

    out = replay_store(GGUFS[0], captured_store, episodes=["ep-replaytest"],
                       sidecar=True, n_ctx=2048)
    assert out and out[0]["turns"] == 1, out

    ep = Path(captured_store) / "ep-replaytest"
    live = {json.loads(ln)["turn"]: json.loads(ln)["vec"]
            for ln in (ep / "attn_received.jsonl").read_text().splitlines()}
    rep = {json.loads(ln)["turn"]: json.loads(ln)["vec"]
           for ln in (ep / "replay_attn_received.jsonl").read_text().splitlines()}
    lv, rv = np.asarray(live[0]), np.asarray(rep[0])
    assert lv.shape == rv.shape
    cos = float(lv @ rv / (np.linalg.norm(lv) * np.linalg.norm(rv)))
    assert cos > 0.99, f"received-attention cosine {cos}"
    assert np.abs(lv - rv).max() < 0.01, "per-position mass drift beyond kernel noise"

    d = pd.concat([pd.read_parquet(p) for p in sorted(ep.glob("tokens-*.parquet"))])
    g = d[d.region_id == "gen"].sort_values("step")
    live_shares = np.asarray([list(s) for s in g.attn_share])
    rep_rec = json.loads((ep / "replay_attn_share.jsonl").read_text().splitlines()[0])
    rep_shares = np.asarray(rep_rec["shares"])
    assert rep_shares.shape == live_shares.shape
    assert np.abs(rep_shares - live_shares).max() < 0.02, \
        "per-token region shares diverge beyond kernel noise"


@needs_model
def test_replay_fills_missing_channels(captured_store, tmp_path):
    """On a store WITHOUT live attention, replay writes the real channels."""
    import shutil

    from vitals.replay.episode import replay_store

    lean = tmp_path / "lean"
    shutil.copytree(Path(captured_store) / "ep-replaytest", lean / "ep-replaytest")
    (lean / "ep-replaytest" / "attn_received.jsonl").unlink()
    # blank the live attn_share column to simulate a lean-captured store
    import pyarrow as pa
    import pyarrow.parquet as pq
    for p in sorted((lean / "ep-replaytest").glob("tokens-*.parquet")):
        tab = pq.read_table(p)
        meta = tab.schema.metadata
        ci = tab.schema.get_field_index("attn_share")
        tab = tab.set_column(ci, tab.schema.field(ci),
                             pa.array([None] * len(tab), type=pa.list_(pa.float32())))
        pq.write_table(tab.replace_schema_metadata(meta), p)

    out = replay_store(GGUFS[0], lean, n_ctx=2048)
    assert out[0]["turns"] == 1 and out[0]["rows_updated"] > 0
    assert (lean / "ep-replaytest" / "attn_received.jsonl").exists()
    d = pd.concat([pd.read_parquet(p)
                   for p in sorted((lean / "ep-replaytest").glob("tokens-*.parquet"))])
    g = d[d.region_id == "gen"].sort_values("step")
    assert g.attn_share.map(lambda s: s is not None and len(s) > 0).all(), \
        "replay must fill attn_share for every generated token"
    # logits channel: top-K present, surprisal consistent with forcing --
    # each token should appear in its own top-K unless truly improbable
    assert g.top_ids.notna().all() and g.ent.notna().all()
    hit = [int(r.token_id) in list(r.top_ids) for _, r in g.iloc[1:].iterrows()]
    assert np.mean(hit) > 0.9, "forced greedy tokens should be in their own top-K"
    assert (g.ent > 0).any(), "entropy must be real, not placeholder"


def test_sparse_row_splits_prompt_and_generated():
    """sparse_row keeps top-K over the prompt AND over the generated
    positions (as generated-step indices), with exact mass totals."""
    from vitals.replay.pooling import SPARSE_K_GEN, sparse_row
    n_prompt = 6
    row = np.array([0.30, 0.02, 0.01, 0.05, 0.00, 0.02,   # prompt: 0.40
                    0.25, 0.05, 0.30, 0.00], dtype=np.float32)   # generated: 0.60
    step, pos, w, pm, gm, gpos, gw = sparse_row(row, n_prompt, 3, step=7, k_gen=3)
    assert step == 7
    assert list(pos) == [0, 3, 1] or list(pos) == [0, 3, 5]      # ties at 0.02
    assert w[0] == pytest.approx(0.30) and pm == pytest.approx(0.40) and gm == pytest.approx(0.60)
    assert list(gpos) == [2, 0, 1] and list(gw) == pytest.approx([0.30, 0.25, 0.05])  # step indices, zero dropped
    # a token with no generated history yet
    _, _, _, _, gm0, gpos0, gw0 = sparse_row(row[:n_prompt], n_prompt, 3, step=0, k_gen=3)
    # default generated-side K is the larger SPARSE_K_GEN
    assert SPARSE_K_GEN >= 256 and len(sparse_row(row, n_prompt, 3, step=7)[5]) == 3
    assert gm0 == 0.0 and len(gpos0) == 0 and len(gw0) == 0


def test_write_attn_sparse_columns(tmp_path):
    from vitals.replay.pooling import sparse_row
    from vitals.replay.writers import write_attn_sparse_turns
    row = np.array([0.5, 0.1, 0.2, 0.2], dtype=np.float32)
    rows = [sparse_row(row, 2, 4, step=0), sparse_row(row, 2, 4, step=1)]
    write_attn_sparse_turns(tmp_path, {0: rows})
    df = pd.read_parquet(tmp_path / "attn_sparse.parquet")
    assert list(df.columns) == ["turn", "step", "pos", "w", "prompt_mass", "gen_mass", "gpos", "gw"]
    assert df.prompt_mass.iloc[0] == pytest.approx(0.6) and df.gen_mass.iloc[0] == pytest.approx(0.4)
    assert list(df.gpos.iloc[0]) == [0, 1] or list(df.gpos.iloc[0]) == [1, 0]


def test_chunk_logit_stats_matches_per_token():
    """The batched (torch, float64) statistics equal the per-row numpy form."""
    pytest.importorskip("torch")
    from vitals.replay.pooling import chunk_logit_stats, logit_stats

    rng = np.random.default_rng(0)
    rows = (rng.standard_normal((5, 1000)) * 4).astype(np.float32)
    toks = rng.integers(0, 1000, 5)
    got = chunk_logit_stats(rows, toks, 8)
    assert len(got) == 5
    for r, t, g in zip(rows, toks, got, strict=True):
        ref = logit_stats(r, int(t), 8)
        assert g["ent"] == pytest.approx(ref["ent"], rel=1e-9)
        assert g["surprisal"] == pytest.approx(ref["surprisal"], rel=1e-9)
        assert g["m2"] == pytest.approx(ref["m2"], abs=1e-12)
        assert g["top_ids"] == ref["top_ids"]
        assert np.allclose(g["top_lp"], ref["top_lp"], rtol=1e-9)


def test_pool_chunk_matches_per_row():
    """Device-side pooling of a chunk equals pool_row / sparse_row per row."""
    torch = pytest.importorskip("torch")
    from vitals.replay.pooling import pool_chunk, pool_row, sparse_row

    rng = np.random.default_rng(1)
    n_prompt, width, q = 40, 52, 6
    rows = rng.random((q, width)).astype(np.float32)
    rows[:, 45:] = 0.0            # masked-future cells are exact zeros
    rows[2, 10:20] = 0.0          # zero prompt cells must drop out of the sparse rows
    spans = [(0, 5), (5, 40), (None, None), (12, 30)]
    shares, recv, sparse = pool_chunk(torch.from_numpy(rows), spans, n_prompt, 3, step0=4)
    assert len(shares) == len(sparse) == q
    for i in range(q):
        assert np.allclose(shares[i], pool_row(rows[i], spans, gen_start=n_prompt), atol=1e-6)
        s, pos, w, pm, gm, gpos, gw = sparse_row(rows[i], n_prompt, 3, step=4 + i)
        ss, spos, sw, spm, sgm, sgpos, sgw = sparse[i]
        assert ss == s and spos.tolist() == pos.tolist() and sgpos.tolist() == gpos.tolist()
        assert np.allclose(sw, w) and np.allclose(sgw, gw)
        assert spm == pytest.approx(pm, rel=1e-6) and sgm == pytest.approx(gm, rel=1e-6)
        assert spos.dtype == np.int32 and sw.dtype == np.float32
    assert np.allclose(recv.cpu().numpy(), rows[:, :n_prompt].astype(np.float64).sum(0))


def test_bulk_writers_replace_listed_turns_only(tmp_path):
    """The per-episode writers rewrite each listed turn's rows and keep the rest."""
    from vitals.core.store import write_attn_tokens, write_attn_tokens_turns
    from vitals.replay.writers import write_attn_sparse_turns

    regs = [{"id": "user_0", "token_start": 0, "token_end": 3}, {"id": "span:a", "token_start": 1, "token_end": 3}]
    write_attn_tokens(tmp_path, 0, [1, 2, 3], ["a", "b", "c"], regs, [0.1, 0.2, 0.3])
    write_attn_tokens(tmp_path, 1, [4, 5, 6], ["d", "e", "f"], regs, [0.4, 0.5, 0.6])
    write_attn_tokens_turns(tmp_path, {1: ([7, 8, 9], ["g", "h", "i"], regs, [0.7, 0.8, 0.9]),
                                       2: ([1, 1, 1], ["x", "x", "x"], regs, [0.0, 0.0, 0.0])})
    at = pd.read_parquet(tmp_path / "attn_tokens.parquet").sort_values(["turn", "idx"])
    assert at.turn.tolist() == [0] * 3 + [1] * 3 + [2] * 3
    assert at[at.turn == 1].token_id.tolist() == [7, 8, 9] and at[at.turn == 0].token_id.tolist() == [1, 2, 3]
    assert at.region_id.tolist()[:3] == ["user_0", "span:a", "span:a"]

    def row(step):
        return (step, np.array([0], np.int32), np.array([1.0], np.float32), 1.0, 0.0,
                np.zeros(0, np.int32), np.zeros(0, np.float32))
    write_attn_sparse_turns(tmp_path, {0: [row(0), row(1)]})
    write_attn_sparse_turns(tmp_path, {0: [row(0)], 3: [row(0), row(1), row(2)]})
    sp = pd.read_parquet(tmp_path / "attn_sparse.parquet").sort_values(["turn", "step"])
    assert sp.turn.tolist() == [0, 3, 3, 3] and sp.step.tolist() == [0, 0, 1, 2]
