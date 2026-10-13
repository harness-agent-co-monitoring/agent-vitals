"""OpenAI-compatible server with the /v1/vitals sidecar.

Unmodified harnesses point at /v1/chat/completions and notice nothing; metrics
never touch the completions payload. Episode identity: X-Vitals-Episode header,
body metadata.episode_id, or the message-prefix fingerprint (which also detects
compaction). Generation completes before the response is sent (stream=true gets
it as SSE chunks); the engine serves one request at a time.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import ValidationError

from vitals import __version__
from vitals.analysis.episode import turn_summaries
from vitals.analysis.readmap import episode_readmap
from vitals.backends.llamacpp import LlamaCppBackend
from vitals.core.schema import (
    CaptureSpec,
    CaptureUnsupported,
    EngineInfo,
    EpisodeManifest,
    ModelInfo,
    TokenMetrics,
)
from vitals.core.store import TraceStore, check_episode_id
from vitals.engine import PINNED_COMMIT, PINNED_TAG
from vitals.engine.sampler import SamplerConfig
from vitals.loop.decode import run_turn
from vitals.server.session import EpisodeTracker
from vitals.server.toolcalls import parse_assistant_output

log = logging.getLogger(__name__)


@dataclass
class ServerState:
    backend: LlamaCppBackend
    store: TraceStore
    model_name: str
    model_path: str = ""
    default_capture: CaptureSpec = field(default_factory=CaptureSpec)
    store_messages: bool = False
    # server-side declared sections attached to every request, for harnesses
    # whose client cannot pass them per call
    default_spans: list[dict] = field(default_factory=list)
    # model-shipped sampling defaults from GGUF metadata (general.sampling.*);
    # used as the fallback when a request omits sampling params
    model_sampling: dict = field(default_factory=dict)
    tracker: EpisodeTracker = field(default_factory=EpisodeTracker)
    lock: threading.Lock = field(default_factory=threading.Lock)


class _StoreSink:
    def __init__(self, store: TraceStore):
        self.store = store

    def emit(self, tm: TokenMetrics) -> None:
        self.store.append(tm)


def _stream_response(payload: dict, episode_id: str):
    """OpenAI-shaped SSE stream of an already completed generation, for clients
    that require stream=true."""
    from fastapi.responses import StreamingResponse

    base = {k: payload[k] for k in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"
    choice = payload["choices"][0]

    def chunks():
        def sse(delta: dict, finish=None) -> str:
            c = dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": finish}])
            return f"data: {json.dumps(c)}\n\n"

        yield sse({"role": "assistant"})
        content = choice["message"].get("content") or ""
        for i in range(0, len(content), 64):
            yield sse({"content": content[i : i + 64]})
        for j, tc in enumerate(choice["message"].get("tool_calls") or []):
            yield sse({"tool_calls": [dict(tc, index=j)]})
        yield sse({}, finish=choice["finish_reason"])
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        chunks(), media_type="text/event-stream", headers={"X-Vitals-Episode": episode_id}
    )


def _sampler_from_request(body: dict, model_defaults: dict | None = None) -> SamplerConfig:
    """Client-specified sampling wins; otherwise the model's shipped defaults
    (GGUF general.sampling.*); otherwise the built-in fallback."""
    md = model_defaults or {}

    def pick(key, model_key, fallback, cast):
        v = body.get(key)
        if v is not None:
            return cast(v)
        if model_key in md:
            return cast(md[model_key])
        return cast(fallback)

    # OpenAI semantics: no client seed => nondeterministic. A fixed default would
    # collapse same-prompt resampling into identical episodes. The drawn seed is
    # recorded in the episode manifest, so any run stays reproducible by record.
    seed = body.get("seed")
    return SamplerConfig(
        temperature=pick("temperature", "temp", 1.0, float),
        top_p=pick("top_p", "top_p", 1.0, float),
        top_k=pick("top_k", "top_k", 0, int),
        min_p=pick("min_p", "min_p", 0.0, float),
        seed=int(seed) if seed is not None else secrets.randbits(31),
    )


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="vitals", version=__version__)
    eng = state.backend.engine

    def _manifest(episode_id: str, capture: CaptureSpec, sampler: SamplerConfig) -> EpisodeManifest:
        return EpisodeManifest(
            episode_id=episode_id,
            engine=EngineInfo(
                kind="llamacpp",
                build=f"{PINNED_TAG} ({PINNED_COMMIT})",
                flags={
                    "diagnostic_attention": state.backend.diagnostic_attention,
                    "n_ctx": eng._ctx_opts["n_ctx"],
                },
            ),
            model=ModelInfo(
                name=state.model_name, path=state.model_path, arch=eng.arch, n_layers=eng.n_layer
            ),
            sampler=sampler.to_dict(),
            seed=sampler.seed,
            capture=capture,
        )

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model": state.model_name, "engine": f"{PINNED_TAG} ({PINNED_COMMIT})"}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": state.model_name, "object": "model", "owned_by": "vitals"}]}

    @app.post("/v1/chat/completions")
    def chat(body: dict, response: Response, x_vitals_episode: str | None = Header(default=None)) -> dict:
        messages = body.get("messages")
        if not messages:
            raise HTTPException(400, "messages required")
        stream = bool(body.get("stream"))
        tools = body.get("tools") or None
        vit: dict[str, Any] = body.get("vitals") or {}
        # chat template variables: the OpenAI-compatible servers' top-level field (vLLM, SGLang) and
        # vitals.chat_kwargs (which wins on a clash), e.g. {"enable_thinking": true}, {"reasoning_effort": "low"}
        chat_kwargs = {**(body.get("chat_template_kwargs") or {}), **(vit.get("chat_kwargs") or {})} or None
        spans = vit.get("spans") or None  # named anchors: [{"name": ..., "text": ...}]
        if spans is not None and not (
            isinstance(spans, list)
            and all(isinstance(s, dict) and s.get("name") and s.get("text") for s in spans)
        ):
            raise HTTPException(400, 'invalid vitals.spans: expected [{"name", "text"}, ...]')
        if spans is None and state.default_spans:
            spans = state.default_spans
        explicit = x_vitals_episode or (body.get("metadata") or {}).get("episode_id") or vit.get("episode_id")
        if explicit is not None:
            try:
                check_episode_id(str(explicit))
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        try:
            capture = (
                CaptureSpec.model_validate(vit["capture"]) if vit.get("capture") else state.default_capture
            )
        except ValidationError as e:  # pydantic validation detail back to the caller
            raise HTTPException(400, f"invalid vitals.capture: {e}") from e
        sampler = _sampler_from_request(body, state.model_sampling)
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)

        with state.lock:
            episode, events = state.tracker.resolve(messages, explicit)
            if not episode.manifest_written:
                state.store.write_manifest(_manifest(episode.id, capture, sampler))
                episode.manifest_written = True
            state.store.append_events(episode.id, events)
            if state.store_messages:
                # tools and template variables travel with the record so replay renders the served prompt
                state.store.write_messages(episode.id, episode.turn, messages, tools=tools, chat_kwargs=chat_kwargs)
            state.backend.sampler_config = sampler

            attn_recv: dict = {"sum": None, "n": 0}  # received-attention accumulator

            def capture_sink(_i: int, step) -> None:
                for layer, arr in step.resid.items():
                    rows = np.asarray(arr, dtype=np.float16).reshape(-1, eng.n_embd)
                    state.store.save_activations(episode.id, layer, episode.turn, rows)
                if step.attn:
                    # each prompt position's received attention: per layer
                    # normalize rows, mean over heads, then mean over layers,
                    # summed over the turn's steps
                    vecs = []
                    for a in step.attn.values():
                        v = np.asarray(a, dtype=np.float64).reshape(-1, np.asarray(a).shape[-1])
                        v = v / np.maximum(v.sum(axis=1, keepdims=True), 1e-9)
                        vecs.append(v.mean(axis=0))
                    if vecs:
                        n = min(len(x) for x in vecs)
                        vec = np.mean([x[:n] for x in vecs], axis=0)
                        if attn_recv["sum"] is None:
                            attn_recv["sum"] = vec.copy()
                        else:
                            m = min(len(attn_recv["sum"]), len(vec))
                            attn_recv["sum"][:m] += vec[:m]
                        attn_recv["n"] += 1

            try:
                res = run_turn(
                    state.backend,
                    messages,
                    episode_id=episode.id,
                    turn=episode.turn,
                    capture=capture,
                    sink=_StoreSink(state.store),
                    events=events,
                    max_tokens=max_tokens,
                    accumulator=episode.accumulator,
                    capture_sink=capture_sink,
                    tools=tools,
                    chat_kwargs=chat_kwargs,
                    spans=spans,
                )
            except CaptureUnsupported as e:
                raise HTTPException(400, str(e)) from e
            except RuntimeError as e:  # e.g. tools requested without a jinja template
                raise HTTPException(400, str(e)) from e
            lc = state.backend.last_ctx
            if lc is not None:
                state.store.write_regions(episode.id, episode.turn, lc.regions)
            if attn_recv["sum"] is not None and attn_recv["n"]:
                recv = attn_recv["sum"] / attn_recv["n"]
                state.store.append_attn_received(episode.id, episode.turn, recv)
                if lc is not None and getattr(lc, "prompt_token_ids", None) is not None:
                    # the per-token table behind peak lift and hottest lines,
                    # the same artifact replay writes
                    toks = [int(t) for t in lc.prompt_token_ids[: len(recv)]]
                    state.store.append_attn_tokens(
                        episode.id, episode.turn, toks,
                        [state.backend.detokenize([t]) for t in toks], lc.regions, recv)
            if lc is not None and getattr(lc, "prompt_stats", None) is not None:
                # prompt-token rows: position i's stats under its region id
                # (sys_0/user_0/obs_k/...); step = absolute prompt position
                message_regions = [(r.id, r.token_start, r.token_end) for r in lc.regions
                                   if r.token_start is not None and r.token_end is not None
                                   and not str(r.id).startswith("span:")]

                def _rid(pos: int) -> str:
                    for rid, ts, te in message_regions:
                        if ts <= pos < te:
                            return rid
                    return "prompt"

                for i, (p_ent, p_m2, p_sup) in enumerate(lc.prompt_stats, start=1):
                    state.store.append(TokenMetrics(
                        episode_id=episode.id, turn=episode.turn, step=i,
                        token_id=int(lc.prompt_token_ids[i]),
                        ent=float(p_ent), m2=float(p_m2), js=0.0, fork=False,
                        region_id=_rid(i), surprisal=float(p_sup)))
            episode.turn += 1
            state.store.flush(episode.id)

            stopped = bool(res.tokens) and eng.is_eog(res.tokens[-1])
            content_tokens = res.tokens[:-1] if stopped else res.tokens
            completion = len(res.tokens)
            prompt_tokens = eng.n_past - (completion - 1 if stopped else completion)
            text = state.backend.detokenize(content_tokens)

        # a cue that ends inside an open think block (Qwen3.x, GLM) means the completion starts in it
        cue = state.backend.detokenize([int(t) for t in lc.prompt_token_ids[-3:]]) \
            if lc is not None and getattr(lc, "prompt_token_ids", None) else ""
        text, reasoning, tool_calls = parse_assistant_output(
            text, tools, think_opened=cue.rstrip().endswith("<think>"), parse_calls=bool(tools))
        message: dict[str, Any] = {"role": "assistant", "content": text if text or not tool_calls else None}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else ("stop" if stopped else "length")
        payload = {
            "id": f"chatcmpl-{episode.id}-{episode.turn - 1}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": state.model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion,
                "total_tokens": prompt_tokens + completion,
            },
        }
        if stream:
            return _stream_response(payload, episode.id)
        response.headers["X-Vitals-Episode"] = episode.id
        return payload

    def _tokens(episode_id: str):
        try:
            return state.store.read_tokens(episode_id)
        except ValueError as e:             # an id that is not one directory name
            raise HTTPException(404, str(e)) from e

    @app.get("/v1/vitals/episodes/{episode_id}")
    def episode_tokens(episode_id: str, offset: int = 0, limit: int = 2000) -> dict:
        table = _tokens(episode_id)
        if table.num_rows == 0:
            raise HTTPException(404, f"no trace for episode {episode_id!r}")
        rows = table.slice(offset, limit).to_pylist()
        for r in rows:
            r["probes"] = dict(r["probes"] or [])
        return {"episode_id": episode_id, "total": table.num_rows, "offset": offset, "tokens": rows}

    @app.get("/v1/vitals/episodes/{episode_id}/report")
    def episode_report(episode_id: str, diagnostics: bool = False) -> dict:
        table = _tokens(episode_id)
        if table.num_rows == 0:
            raise HTTPException(404, f"no trace for episode {episode_id!r}")
        manifest = state.store.read_manifest(episode_id)
        return {
            "episode_id": episode_id,
            "manifest": manifest.model_dump(mode="json"),
            "turns": [t.model_dump(mode="json") for t in turn_summaries(table)],
            # per (turn, declared section): read lift, peak lift and hottest
            # line, consult ratio and contrast; empty without attention capture
            "readmap": episode_readmap(state.store.episode_dir(episode_id), diagnostics=diagnostics),
            "events": [e.model_dump(mode="json") for e in state.store.read_events(episode_id)],
            "regions": state.store.read_regions(episode_id),
        }

    return app


def build_state(
    gguf: str,
    *,
    store_dir: str = "runs/traces",
    n_ctx: int = 8192,
    n_gpu_layers: int = 999,
    diagnostic_attention: bool = False,
    capture_layers: list[int] | None = None,
    capture_attention: bool = False,
    top_logprobs: int = 0,
    prompt_scores: bool = False,
    default_spans: list[dict] | None = None,
    store_messages: bool = False,
    lib_path: str | None = None,
    reuse_kv: bool = False,
    auto_sections: bool = True,
) -> ServerState:
    if capture_attention and not diagnostic_attention:
        raise ValueError(
            "--capture-attention needs --diagnostic-attention: attention tensors "
            "only materialize with flash attention disabled"
        )
    backend = LlamaCppBackend(
        gguf,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        diagnostic_attention=diagnostic_attention,
        lib_path=lib_path,
        reuse_kv=reuse_kv,
    )
    backend.auto_sections = auto_sections
    spec: dict = {}
    if capture_layers:
        spec["activations"] = {"layers": capture_layers}
    if capture_attention:
        # region-pooled = per-token attn_share vectors (last-row capture on the
        # full-attention layers)
        spec["attention"] = {"mode": "region_pooled"}
    if top_logprobs:
        # per-token top-K ids+logprobs and sampled-token surprisal into the parquet
        spec["top_logprobs"] = int(top_logprobs)
    if prompt_scores:
        spec["prompt_scores"] = True
    capture = CaptureSpec.model_validate(spec) if spec else CaptureSpec()
    # model-shipped sampling defaults (absent keys fall through to the built-in fallback)
    model_sampling: dict = {}
    for mk, cast in (("temp", float), ("top_k", int), ("top_p", float), ("min_p", float)):
        raw = backend.engine.meta(f"general.sampling.{mk}")
        if raw is not None:
            try:
                model_sampling[mk] = cast(float(raw))
            except ValueError:
                pass
    if model_sampling:
        log.info("model sampling defaults %s (used when requests omit sampling params)", model_sampling)

    return ServerState(
        backend=backend,
        store=TraceStore(store_dir),
        model_name=Path(gguf).stem,
        model_path=str(gguf),
        default_capture=capture,
        store_messages=store_messages,
        model_sampling=model_sampling,
        default_spans=default_spans or [],
    )
