"""vitals serve integration: OpenAI surface, episode tracking, store wiring,
sidecar endpoints. Runs against the 0.8B GGUF; auto-skips without engine/model."""

import json
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


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from fastapi.testclient import TestClient

    from vitals.server.app import build_state, create_app

    store_dir = tmp_path_factory.mktemp("traces")
    state = build_state(GGUFS[0], store_dir=str(store_dir), n_ctx=2048)
    client = TestClient(create_app(state))
    yield client, state, store_dir
    state.backend.close()


def _chat(client, messages, **kw):
    body = {"model": "m", "messages": messages, "temperature": 0, "max_tokens": 24, **kw}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    return r


def test_openai_surface(server):
    client, _, _ = server
    r = _chat(client, [{"role": "user", "content": "Say OK and nothing else."}])
    j = r.json()
    assert j["object"] == "chat.completion"
    msg = j["choices"][0]["message"]
    assert msg["role"] == "assistant" and isinstance(msg["content"], str) and msg["content"]
    u = j["usage"]
    assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert r.headers["X-Vitals-Episode"].startswith("ep-")
    assert client.get("/health").json()["ok"] is True
    assert client.get("/v1/models").json()["data"][0]["id"]


def test_multi_turn_episode_and_sidecar(server):
    client, _, _ = server
    msgs = [{"role": "user", "content": "Name one planet."}]
    r1 = _chat(client, msgs)
    ep = r1.headers["X-Vitals-Episode"]
    reply = r1.json()["choices"][0]["message"]["content"]

    msgs2 = msgs + [{"role": "assistant", "content": reply},
                    {"role": "tool", "content": "Traceback (most recent call last): boom"},
                    {"role": "user", "content": "And another?"}]
    r2 = _chat(client, msgs2, **{"metadata": {"episode_id": ep}})
    assert r2.headers["X-Vitals-Episode"] == ep

    tok = client.get(f"/v1/vitals/episodes/{ep}").json()
    turns = {t["turn"] for t in tok["tokens"]}
    assert turns == {0, 1} and tok["total"] > 0

    rep = client.get(f"/v1/vitals/episodes/{ep}/report").json()
    assert rep["manifest"]["engine"]["build"].startswith("b9910")
    assert len(rep["turns"]) == 2
    assert all(t["n_tokens"] > 0 and t["mean_ent"] >= 0 for t in rep["turns"])
    kinds = [e["kind"] for e in rep["events"]]
    assert "OBSERVATION_INJECTED" in kinds and "ERROR_BLOCK" in kinds


def test_fingerprint_continuation_without_header(server):
    client, _, _ = server
    msgs = [{"role": "user", "content": "Count from 1 to 3."}]
    r1 = _chat(client, msgs)
    ep = r1.headers["X-Vitals-Episode"]
    reply = r1.json()["choices"][0]["message"]["content"]
    r2 = _chat(client, msgs + [{"role": "assistant", "content": reply},
                               {"role": "user", "content": "Now to 5."}])
    assert r2.headers["X-Vitals-Episode"] == ep


def test_shared_system_prompt_episodes_do_not_merge(server):
    """Independent tasks sharing a system prompt must stay separate, and each task's continuation must attach to its OWN opener (not
    funnel into the first-created episode)."""
    client, _, _ = server
    sys = {"role": "system", "content": "You are a terse coding agent."}
    r_a = _chat(client, [sys, {"role": "user", "content": "Task A: print apple."}])
    r_b = _chat(client, [sys, {"role": "user", "content": "Task B: print banana."}])
    ep_a, ep_b = r_a.headers["X-Vitals-Episode"], r_b.headers["X-Vitals-Episode"]
    assert ep_a != ep_b, "distinct tasks must not share an episode"

    reply_b = r_b.json()["choices"][0]["message"]["content"]
    r_b2 = _chat(client, [sys, {"role": "user", "content": "Task B: print banana."},
                          {"role": "assistant", "content": reply_b},
                          {"role": "user", "content": "Now uppercase it."}])
    assert r_b2.headers["X-Vitals-Episode"] == ep_b, "continuation must attach to its own opener"


def test_store_messages_persists_history_for_replay(tmp_path_factory):
    from fastapi.testclient import TestClient

    from vitals.server.app import build_state, create_app

    store_dir = tmp_path_factory.mktemp("msgstore")
    state = build_state(GGUFS[0], store_dir=str(store_dir), n_ctx=2048, store_messages=True)
    try:
        client = TestClient(create_app(state))
        h = {"X-Vitals-Episode": "replay-ep"}
        msgs = [{"role": "user", "content": "Name a color."}]
        client.post("/v1/chat/completions",
                    json={"model": "m", "messages": msgs, "temperature": 0, "max_tokens": 8}, headers=h)
        stored = state.store.read_messages("replay-ep")
        assert stored and stored[0]["messages"][0]["content"] == "Name a color."
    finally:
        state.backend.close()


def test_explicit_episode_id_header_bypasses_heuristic(server):
    client, _, _ = server
    h = {"X-Vitals-Episode": "batch-run-t07"}
    r = client.post("/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}],
                          "temperature": 0, "max_tokens": 8}, headers=h)
    assert r.headers["X-Vitals-Episode"] == "batch-run-t07"


def test_capture_to_zarr_via_request_block(server):
    client, _, store_dir = server
    r = _chat(client, [{"role": "user", "content": "What color is the sky?"}],
              vitals={"capture": {"activations": {"layers": [5]}}})
    ep = r.headers["X-Vitals-Episode"]
    import zarr

    grp = zarr.open_group(str(store_dir / ep / "activations.zarr"), mode="r")
    arr = grp["layer5/turn0"][:]
    assert arr.ndim == 2 and arr.shape[1] == 1024 and arr.shape[0] > 0


def test_streaming_sse_shape_matches_nonstream_content(server):
    client, _, _ = server
    msgs = [{"role": "user", "content": "Say exactly: ok"}]
    plain = _chat(client, msgs).json()["choices"][0]["message"]["content"]

    r = client.post("/v1/chat/completions",
                    json={"model": "m", "messages": msgs, "temperature": 0, "max_tokens": 24,
                          "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert content == plain
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_tools_render_and_response_shape(server):
    client, _, _ = server
    tools = [{"type": "function", "function": {
        "name": "get_time", "description": "returns the current time",
        "parameters": {"type": "object", "properties": {}}}}]
    r = _chat(client, [{"role": "user", "content": "hello"}], tools=tools)
    j = r.json()
    msg = j["choices"][0]["message"]
    # 0.8B may or may not call the tool; both shapes must be valid
    if j["choices"][0]["finish_reason"] == "tool_calls":
        assert msg["tool_calls"][0]["function"]["name"]
    else:
        assert isinstance(msg["content"], str)


def test_bad_capture_refused_with_explanation(server):
    client, _, _ = server
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
            "vitals": {"capture": {"attention": {"mode": "per_head", "layers": [3]}}}}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 400
    assert "diagnostic" in r.json()["detail"]


def test_reasoning_split_and_template_kwargs_stored(tmp_path_factory):
    """A thinking prompt: the reply's reasoning goes to reasoning_content, not content; the standard
    chat_template_kwargs field reaches the template and is stored with the messages for replay."""
    from fastapi.testclient import TestClient

    from vitals.server.app import build_state, create_app

    store_dir = tmp_path_factory.mktemp("think-store")
    state = build_state(GGUFS[0], store_dir=str(store_dir), n_ctx=2048, store_messages=True)
    try:
        client = TestClient(create_app(state))
        r = client.post("/v1/chat/completions", json={
            "model": "m", "temperature": 0.0, "max_tokens": 160,
            "messages": [{"role": "user", "content": "What is 2+3? Answer with the number."}],
            "chat_template_kwargs": {"enable_thinking": True},
            "vitals": {"episode_id": "ep-think"}})
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert "</think>" not in (msg["content"] or "") and "<think>" not in (msg["content"] or "")
        assert msg.get("reasoning_content")
        state.store.flush("ep-think")
        rec = json.loads((store_dir / "ep-think" / "messages.jsonl").read_text().splitlines()[0])
        assert rec["chat_kwargs"] == {"enable_thinking": True}
    finally:
        state.backend.close()

