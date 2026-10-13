"""Commands that get episodes into a trace store: ``import``, ``serve``, ``replay``, and ``engine``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vitals.cli._common import add_episodes, add_model, add_store, episodes


def add_import(sub) -> None:
    p = sub.add_parser("import", help="import OpenAI-style chat logs (JSONL) into a trace store, grouped into episodes, "
                                      "ready for replay and monitoring")
    p.add_argument("--log", required=True, help="JSONL: {request, response} or {messages, tools, response} per line")
    add_store(p, "trace store directory to write")
    p.add_argument("--gguf", required=True, help="reference model GGUF (its tokenizer and chat template render the "
                                                 "calls; no forward pass)")
    p.add_argument("--prefix", default="log", help="episode id prefix when calls carry no session id")
    p.add_argument("--lib", default=None, help="libllama path (default: $VITALS_LLAMA_LIB, else the pinned build)")
    p.set_defaults(func=run_import)


def run_import(args: argparse.Namespace) -> int:
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.replay.importer import import_log

    # the tokenizer and chat template are all an import needs: keep the weights off the GPU
    backend = LlamaCppBackend(args.gguf, n_ctx=4096, n_gpu_layers=0, lib_path=args.lib)
    try:
        res = import_log(args.log, args.store, backend, prefix=args.prefix)
    finally:
        backend.close()
    print(f"{res['imported']} calls imported into {len(res['episodes'])} episodes ({res['forced_tokens']} reply "
          f"tokens); {res['skipped_invalid']} lines without a complete call; {len(res['errors'])} errors")
    for e in res["errors"][:5]:
        print(f"  {e['episode']} turn {e['turn']}: {e['error']}")
    return 0


def add_serve(sub) -> None:
    p = sub.add_parser("serve", help="OpenAI-compatible server that records every call into a trace store, with the "
                                     "/v1/vitals sidecar")
    p.add_argument("--gguf", required=True, help="path to the GGUF model to serve")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8033, help="port (default 8033)")
    p.add_argument("--store", default="runs/traces", help="trace store directory (default runs/traces)")
    p.add_argument("--n-ctx", type=int, default=8192, help="context window in tokens")
    p.add_argument("--n-gpu-layers", type=int, default=999, help="layers offloaded to the GPU (default: all)")
    p.add_argument("--lib", default=None, help="libllama path (default: $VITALS_LLAMA_LIB, else the pinned build)")
    p.add_argument("--diagnostic-attention", action="store_true",
                   help="disable flash attention to enable per-head attention capture (slower)")
    p.add_argument("--capture-layers", default="",
                   help="comma-separated layer indices whose residual activations are saved to activations.zarr, "
                        "e.g. 3,15,31")
    p.add_argument("--store-messages", action="store_true",
                   help="store request messages, tools and chat template options per turn (needed for replay; may "
                        "hold sensitive data)")
    p.add_argument("--capture-attention", action="store_true",
                   help="record per-token region-pooled attention shares (requires --diagnostic-attention)")
    p.add_argument("--top-logprobs", type=int, default=0, metavar="K",
                   help="store the top-K token ids+logprobs and sampled-token surprisal "
                        "per generated token (default 0 = off; ~K*8 bytes/token)")
    p.add_argument("--prompt-scores", action="store_true",
                   help="score PROMPT tokens during prefill (ent/m2/surprisal per "
                        "position, stored under prompt region ids)")
    p.add_argument("--spans-file", default="",
                   help='sections JSON [{"name","text"},...] attached to EVERY request, for section-level attention '
                        "under unmodified agent clients (per-request vitals.spans overrides it)")
    p.add_argument("--no-auto-sections", action="store_true",
                   help="do not add automatic sections (system prompt by heading, tool:<name>, user:<k>, "
                        "self:<k>, tool_result:<tool>#<k>) to the parts of a request no declared span covers")
    p.add_argument("--reuse-kv", action="store_true",
                   help="reuse KV across turns of an episode when the new prompt exactly "
                        "extends the resident tokens (append-only; kills quadratic re-prefill). "
                        "NOTE: prefill activation capture then covers only the delta rows")
    p.set_defaults(func=run_serve)


def run_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from vitals.server.app import build_state, create_app

    layers = [int(x) for x in args.capture_layers.split(",") if x.strip()]
    default_spans = json.loads(Path(args.spans_file).read_text()) if args.spans_file else None
    state = build_state(
        args.gguf,
        store_dir=args.store,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        diagnostic_attention=args.diagnostic_attention,
        capture_layers=layers,
        capture_attention=args.capture_attention,
        top_logprobs=args.top_logprobs,
        prompt_scores=args.prompt_scores,
        default_spans=default_spans,
        store_messages=args.store_messages,
        lib_path=args.lib,
        reuse_kv=args.reuse_kv,
        auto_sections=not args.no_auto_sections,
    )
    print(f"vitals serve: {state.model_name} | store={args.store} | "
          f"capture_layers={layers or 'none'} | http://{args.host}:{args.port}/v1")
    if state.model_sampling:
        print(f"vitals serve: model sampling defaults {state.model_sampling} (used when requests omit sampling params)")
    uvicorn.run(create_app(state), host=args.host, port=args.port, log_level="warning")
    return 0


def add_replay(sub) -> None:
    p = sub.add_parser("replay", help="teacher-forced replay of a trace store: recover attention and next-token "
                                      "signals offline from the logged tokens")
    add_store(p)
    add_model(p, n_ctx=32768, gguf_help="the SERVING model's GGUF (attention must come from the same weights)")
    add_episodes(p)
    p.add_argument("--spans-file", default="", help='sections JSON [{"name","text"},...] (e.g. from vitals annotate), '
                                                    "resolved into span:* regions during replay")
    p.add_argument("--no-auto-sections", action="store_true",
                   help="do not add automatic sections to turns for which neither --spans-file nor the store "
                        "declares any")
    p.add_argument("--sidecar", action="store_true",
                   help="write replay_* files instead of filling the live-capture "
                        "channels (for replay-vs-capture validation)")
    p.add_argument("--force", action="store_true", help="overwrite existing channels")
    p.add_argument("--channels", default="attention,logits",
                   help="comma list: attention (fa-off, slower) and/or logits "
                        "(ent/m2/surprisal/top-K, runs fa-on at full speed)")
    p.add_argument("--top-k", type=int, default=32, help="top-K next-token logprobs stored per forced token")
    p.add_argument("--no-token-attention", action="store_true",
                   help="skip attn_tokens.parquet (the attention each prompt token receives; peaks, hottest lines, "
                        "the ledger and explanations need it)")
    p.add_argument("--section-diagnostics", action="store_true",
                   help="capture exact per-step section peaks and attention-focus JS before sparse "
                        "truncation (section_diagnostics.parquet; needs the attention channel)")
    p.add_argument("--chunk", type=int, default=None,
                   help="forced tokens per engine call (default $VITALS_REPLAY_CHUNK, else 64; 256 is faster when "
                        "device memory allows)")
    p.add_argument("--linear-reading", action="store_true",
                   help="hybrid models: also measure the linear-attention layers per section (retained, "
                        "memory share, read, retrieval) into linear_reading.parquet")
    p.add_argument("--linear-max-slots", type=int, default=64,
                   help="section memories per linear layer; beyond this, old message regions fold together")
    p.add_argument("--linear-group-sep", default="",
                   help="group declared spans by the part of their name before this separator (e.g. '/')")
    p.set_defaults(func=run_replay)


def run_replay(args: argparse.Namespace) -> int:
    from vitals.replay import replay_store

    # only non-default linear options are passed on
    linear_opts = {}
    if args.linear_max_slots != 64:
        linear_opts["max_slots"] = args.linear_max_slots
    if args.linear_group_sep:
        linear_opts["group_sep"] = args.linear_group_sep
    replay_store(
        args.gguf, args.store,
        episodes=episodes(args.episodes),
        spans_file=args.spans_file or None,
        sidecar=args.sidecar, force=args.force,
        n_ctx=args.n_ctx, lib_path=args.lib, n_gpu_layers=args.n_gpu_layers,
        channels=tuple(c for c in args.channels.split(",") if c),
        top_k=args.top_k, token_attn=not args.no_token_attention,
        section_diagnostics=args.section_diagnostics, chunk=args.chunk,
        linear=args.linear_reading, linear_opts=linear_opts or None,
        auto_sections=not args.no_auto_sections)
    return 0


def add_engine(sub) -> None:
    p = sub.add_parser("engine", help="check the llama.cpp engine (vitals engine doctor)")
    esub = p.add_subparsers(dest="engine_cmd", required=True)
    d = esub.add_parser("doctor", help="check libllama presence and required symbols")
    d.add_argument("--lib", default=None, help="path to libllama (default: $VITALS_LLAMA_LIB or vendor/)")
    p.set_defaults(func=run_engine)


def run_engine(args: argparse.Namespace) -> int:
    from vitals.engine import binding

    if args.engine_cmd == "doctor":
        report = binding.doctor(args.lib)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    print(f"unknown engine command {args.engine_cmd!r}", file=sys.stderr)
    return 2
