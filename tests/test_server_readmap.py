"""vitals serve with attention capture: the store carries the same attention
artifacts replay writes, and /report returns the read map per declared section.
Runs against the 0.8B GGUF; auto-skips without engine/model."""

import os
from glob import glob
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin"
GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))
pytestmark = pytest.mark.skipif(
    not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS,
    reason="pinned engine build or 0.8B GGUF not available",
)

RULES = ("Rules: answer in one short sentence. Never mention the moon. "
         "Always end your answer with the word done.")
TASK = "Question: what is the largest planet of the solar system?"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from fastapi.testclient import TestClient

    from vitals.server.app import build_state, create_app

    store_dir = tmp_path_factory.mktemp("traces")
    state = build_state(GGUFS[0], store_dir=str(store_dir), n_ctx=2048,
                        diagnostic_attention=True, capture_attention=True, store_messages=True)
    client = TestClient(create_app(state))
    yield client, state, store_dir
    state.backend.close()


def test_report_readmap_per_declared_section(server):
    client, _, store_dir = server
    body = {"model": "m", "temperature": 0, "max_tokens": 32,
            "messages": [{"role": "system", "content": RULES}, {"role": "user", "content": TASK}],
            "vitals": {"spans": [{"name": "rules", "text": RULES}, {"name": "task", "text": TASK}]}}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    ep = r.headers["X-Vitals-Episode"]

    # the artifacts replay writes are present from live capture too
    assert (store_dir / ep / "attn_received.jsonl").exists()
    assert (store_dir / ep / "attn_tokens.parquet").exists()
    import pandas as pd
    at = pd.read_parquet(store_dir / ep / "attn_tokens.parquet")
    assert set(at.region_id.dropna()) >= {"span:rules", "span:task"}
    assert at.attn.sum() > 0

    rep = client.get(f"/v1/vitals/episodes/{ep}/report").json()
    rows = {r["section"]: r for r in rep["readmap"]}
    assert set(rows) == {"rules", "task"}
    for row in rows.values():
        assert row["turn"] == 0 and row["tokens"] > 0 and row["lift"] > 0
        assert row["L"] > row["tokens"]
        # a section under 32 tokens reports no peak; longer ones report one
        assert (row["peak"] is None) == (row["tokens"] < 32)
        assert row["consult_ratio"] is None or row["consult_ratio"] > 0
    # the section shares of one prompt cannot exceed the whole context
    assert sum(r["lift"] * r["tokens"] / r["L"] for r in rows.values()) <= 1.0 + 1e-6
