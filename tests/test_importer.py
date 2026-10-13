"""Generic OpenAI-log import: both line shapes, grouping by session id and by prefix chaining, re-import
idempotence, continuation across imports. Pure grouping runs without a model; recording needs the 0.8B GGUF."""

from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path

import pytest

from vitals.replay.importer import INDEX, group, normalize

VENDOR = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin"
GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))

SYS = {"role": "system", "content": "You are a booking agent. Always confirm before acting."}


def u(t):
    return {"role": "user", "content": t}


def a(t):
    return {"role": "assistant", "content": t}


def log_lines():
    """Two sessions without ids (A, B) interleaved, one with a session id, both line shapes, one bad line."""
    A1 = [SYS, u("Cancel my booking ABC.")]
    B1 = [SYS, u("What is my baggage allowance?")]
    A2 = A1 + [a("Please confirm: cancel ABC?"), u("Yes.")]
    return [
        {"request": {"model": "m", "messages": A1}, "response": {"choices": [{"message": a("Please confirm: cancel ABC?")}]}},
        {"messages": B1, "response": "Two bags."},
        {"request": {"body": {"messages": A2}}, "response": {"choices": [{"message": {"role": "assistant", "content": "Done.",
                                                                                         "reasoning": "User said yes."}}]}},
        {"session_id": "s-1", "messages": [SYS, u("Hi")], "assistant": a("Hello.")},
        {"nonsense": True},
    ]


def test_normalize_and_group_without_model():
    calls = [c for c in map(normalize, log_lines()) if c]
    assert len(calls) == 4 and calls[2]["assistant"]["reasoning_content"] == "User said yes."
    index = {"episodes": {}}
    eids = [e for e, _ in group(calls, index, "log")]
    assert eids[0] == eids[2] != eids[1]               # A2 continues A1; B starts its own episode
    assert eids[3] == "s-1"
    # a later call continuing B joins B, even in a separate grouping pass
    B2 = [SYS, u("What is my baggage allowance?"), a("Two bags."), u("And the weight?")]
    more = group([normalize({"messages": B2, "response": "23 kg each."})], index, "log")
    assert more[0][0] == eids[1]


@pytest.mark.skipif(not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS, reason="pinned engine build or 0.8B GGUF missing")
def test_import_log_records_turns(tmp_path):
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.core.store import TraceStore, episode_dirs
    from vitals.replay.importer import import_log

    log = tmp_path / "calls.jsonl"
    log.write_text("".join(json.dumps(x) + "\n" for x in log_lines()))
    store = tmp_path / "store"
    be = LlamaCppBackend(GGUFS[0], n_ctx=4096)
    try:
        res = import_log(log, store, be)
        again = import_log(log, store, be)
        more = tmp_path / "more.jsonl"
        B2 = [SYS, u("What is my baggage allowance?"), a("Two bags."), u("And the weight?")]
        more.write_text(json.dumps({"messages": B2, "response": "23 kg each."}) + "\n")
        res2 = import_log(more, store, be)
    finally:
        be.close()
    assert res["imported"] == 4 and res["skipped_invalid"] == 1 and not res["errors"]
    assert [n["turn"] for n in res["notes"]] == [1]       # the reply with reasoning needed a retry
    assert again["imported"] == 0
    assert res2["imported"] == 1 and list(res2["episodes"].values()) == [1]
    eps = {p.name for p in episode_dirs(store)}
    assert eps == {"log-0000", "log-0001", "s-1"}
    st = TraceStore(store)
    msgs = st.read_messages("log-0001")
    assert [m["turn"] for m in msgs] == [0, 1] and msgs[1]["messages"][-1]["content"] == "And the weight?"
    turns = sorted(set(st.read_tokens("log-0001").column("turn").to_pylist()))
    assert turns == [0, 1]
    assert json.loads((store / INDEX).read_text())["episodes"]["log-0001"]["turns"] == 2


@pytest.mark.skipif(not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS,
                    reason="pinned engine build or 0.8B GGUF missing")
def test_replay_finds_imported_episodes_and_writes_token_attention(tmp_path):
    """Imported episodes are not named ep-*; a plain replay_store must still replay them, with attn_tokens.parquet
    (what the ledger, peaks and explanations read), and a second replay must find nothing left to do."""
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.replay import replay_store
    from vitals.replay.importer import import_log

    log = tmp_path / "calls.jsonl"
    log.write_text("".join(json.dumps(x) + "\n" for x in log_lines()))
    store = tmp_path / "store"
    be = LlamaCppBackend(GGUFS[0], n_ctx=4096, n_gpu_layers=0)
    try:
        import_log(log, store, be)
    finally:
        be.close()
    first = replay_store(GGUFS[0], store, n_ctx=4096, log=lambda *_: None)
    assert {r["episode"] for r in first} == {"log-0000", "log-0001", "s-1"}
    for ep in ("log-0000", "log-0001", "s-1"):
        assert (store / ep / "attn_tokens.parquet").exists() and (store / ep / "attn_received.jsonl").exists()
    second = replay_store(GGUFS[0], store, n_ctx=4096, log=lambda *_: None)
    assert all(not r.get("rows_updated") for r in second)
