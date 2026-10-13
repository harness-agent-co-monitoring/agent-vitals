"""One-shot in-loop replay for the read-informed run on HANDBOOK.md (mirror of
replay/inloop_tb.py). Called by the HANDBOOK meta_harness.py after a
candidate's benchmark: takes the candidate's harbor job directory, picks a
fixed, outcome-stratified subsample of its trials (top and bottom halves by
rubric score), ingests them from the OpenHands completion logs (exact logged
requests; the candidate's own `prompt_sections()` declaration labels the
harness sentences, handbook passages come from the document detector) and
replays them through the BF16 reference into <out>/<agent>--handbook/store,
where digest-handbook reads them.

Idempotent: episodes already carrying attn_tokens.parquet are skipped.

Usage:
  vitals metaharness replay-inloop-handbook --jobs <job_dir> [...] \
      --out <logs/<run>/vitals> [--subsample 6] [--agent NAME ...] [--agents-dir DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from vitals.addons.metaharness.config import CFG
from vitals.addons.metaharness.replay import log, manifest_spans
from vitals.addons.metaharness.replay.inloop import acquire_gpu_slot
from vitals.addons.metaharness.replay.inloop_tb import agent_of

N_CTX = 212992          # 2 seqs -> 106k per turn; longer turns are skipped by the replay
HB_AGENT_SRC = Path(os.path.expanduser(os.environ.get("HB_ROOT", "~/benchmarks/handbook"))) / "agent_harness" / "src" / "agent_harness"


def trial_score(trial: Path):
    rf = trial / "result.json"
    if not rf.exists():
        return None
    try:
        vr = json.loads(rf.read_text()).get("verifier_result") or {}
        r = (vr.get("rewards") or {}).get("reward")
        return None if r is None else float(r)
    except (json.JSONDecodeError, OSError):
        return None


def pick_trials(job_dir: Path, subsample: int):
    """Deterministic, outcome-stratified on the rubric score: the strongest
    half and the weakest half of the graded trials that carry completion logs."""
    graded = []
    for trial in sorted(job_dir.iterdir()):
        if not trial.is_dir() or "__" not in trial.name:
            continue
        if not any((trial / "agent" / "completions").glob("*.json")):
            continue
        r = trial_score(trial)
        if r is None:
            continue
        graded.append((trial, r))
    if not graded:
        return []
    by_score = sorted(graded, key=lambda x: (-x[1], x[0].name))
    half = max(1, subsample // 2)
    top, bottom = by_score[:half], list(reversed(by_score))[:half]
    picked, seen = [], set()
    for t, r in top + bottom + by_score:
        if t.name in seen:
            continue
        picked.append((t, r))
        seen.add(t.name)
        if len(picked) >= subsample:
            break
    return picked


def load_declaration(agents_dir: Path | None, agent: str):
    """(`prompt_sections`, `feedback_sections`) from the candidate's module
    (agents/<agent>.py). The module subclasses HandbookRunner (benchmark
    checkout), importable on the host since it imports the SDK lazily."""
    if not agents_dir:
        return None
    names = [agent, agent.replace("-", "_")]
    if agent.endswith("-baseline"):            # kira-baseline -> baseline_kira (the TB convention)
        names.append("baseline_" + agent[: -len("-baseline")].replace("-", "_"))
    path = next((Path(agents_dir) / f"{n}.py" for n in names if (Path(agents_dir) / f"{n}.py").exists()), None)
    if path is None:
        return None
    try:
        for d in (str(HB_AGENT_SRC), str(Path(agents_dir).resolve().parent)):
            if d not in sys.path:
                sys.path.insert(0, d)
        spec = importlib.util.spec_from_file_location(f"_hb_candidate_{agent}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ps = getattr(mod, "prompt_sections", None)
        fb = getattr(mod, "feedback_sections", None)
        return (ps if callable(ps) else None, fb if callable(fb) else None)
    except Exception as exc:
        log(f"  declaration of {agent} unusable ({str(exc)[:80]}); sentence split instead")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="vitals root: stores land at <out>/<agent>--handbook/store")
    ap.add_argument("--subsample", type=int, default=6)
    ap.add_argument("--agent", nargs="*", default=None)
    ap.add_argument("--agents-dir", type=Path, default=None, help="dir holding agents/<name>.py (prompt_sections declarations)")
    ap.add_argument("--tools-json", type=Path, default=CFG.handbook_tools_json,
                    help="cached OpenAI tool schemas for trials without agent/tools_openai.json "
                         "(default: $VITALS_MH_HANDBOOK_TOOLS_JSON)")
    ap.add_argument("--max-prompt-tokens", type=int, default=N_CTX // 2)
    ap.add_argument("--ingest-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from vitals.addons.metaharness.ingest import detok_engine
    from vitals.addons.metaharness.ingest.handbook import ingest_trial, trial_meta
    from vitals.core.store import TraceStore
    from vitals.engine.template import ChatTemplate, template_from_gguf

    tpl = ChatTemplate(template_from_gguf(CFG.reference_gguf))
    detok = detok_engine()
    backend = None
    t0 = time.time()
    for k, job_dir in enumerate(args.jobs):
        if not job_dir.is_dir():
            log(f"SKIP {job_dir}: not a directory")
            continue
        agent = args.agent[k] if args.agent and k < len(args.agent) else agent_of(job_dir)
        store_dir = args.out / f"{agent}--handbook" / "store"
        store = TraceStore(store_dir)
        decl = load_declaration(args.agents_dir, agent)
        declared, feedback_fn = (decl or (None, None))
        picked = pick_trials(job_dir, args.subsample)
        if not picked:
            log(f"SKIP {job_dir.name}: no graded trials with completion logs")
            continue
        log(f"{agent}: {len(picked)} trials (scores {', '.join(f'{r:.2f}' for _, r in picked)}); declaration: {'yes' if declared else 'sentence split'}")
        pending = []
        for trial, _score in picked:
            eid = f"ep-{trial.name}"
            epdir = store_dir / eid
            if (epdir / "attn_tokens.parquet").exists():
                continue
            if not epdir.exists():
                meta_extra = trial_meta(trial)
                meta_extra["memory"] = agent
                meta = ingest_trial(trial, store, eid, tpl, detok, meta_extra, args.max_prompt_tokens,
                                    tools_json=args.tools_json, system_sections=declared, feedback_fn=feedback_fn)
                if meta is None or "error" in meta:
                    log(f"  SKIP {trial.name}: {(meta or {}).get('error', 'no calls')}")
                    continue
            pending.append(eid)
        if args.ingest_only or not pending:
            continue
        if backend is None:
            from vitals.backends.llamacpp import LlamaCppBackend
            log(f"GPU slot {acquire_gpu_slot()}; loading the reference (fa-off, diagnostic, 2 seqs x {N_CTX // 2 // 1024}k)...")
            backend = LlamaCppBackend(str(CFG.reference_gguf), n_ctx=N_CTX, diagnostic_attention=True, n_seq_max=2)
        from vitals.replay import plan_episode_order, replay_episode
        for eid, nxt in plan_episode_order(backend, store_dir, pending):
            reqs = manifest_spans(store_dir / eid)
            try:
                r = replay_episode(backend, store_dir, eid, spans=reqs or None, channels=("attention", "logits"),
                                   token_attn=True, kv_reuse=True, next_prompt=nxt)
                tmr = r.get("timing") or {}
                log(f"  {eid}: {r.get('turns')} turns, {r.get('rows_updated')} rows"
                    + (f", {len(r['skipped'])} turns skipped" if r.get("skipped") else "")
                    + f"; {tmr.get('seconds', 0):.0f}s, prompt {tmr.get('prompt_tokens', 0):,} tok ({tmr.get('reused_tokens', 0):,} reused)")
                with (args.out / "replay_timing.jsonl").open("a") as f:
                    f.write(json.dumps({"agent": agent, "episode": eid, "turns_replayed": r.get("turns"),
                                        **{k: v for k, v in tmr.items() if k != "turns"}}) + "\n")
            except Exception as exc:
                log(f"  {eid}: REPLAY FAILED ({str(exc)[:100]}) -- continuing")
    log(f"done ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
