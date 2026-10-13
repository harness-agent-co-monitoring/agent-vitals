"""One-shot in-loop replay for the read-informed run on terminal-bench.

Called by the TB meta_harness.py after a candidate's benchmark: takes that
candidate's harbor job directory, picks a fixed, outcome-stratified subsample
of its trials, ingests them (declared scaffold sections, stripped obs spans,
harness-feedback spans, msg_index anchoring) and replays them through the BF16
reference with attention + logits + attn_tokens + attn_sparse, into
<out>/<agent>--tb2/store, where the TB digest reads them.

Idempotent: episodes already carrying attn_tokens.parquet are skipped, so a
crash-and-retry of the same iteration is cheap. --ingest-only prepares the
stores without loading the reference (no GPU), for tests.

Usage:
  vitals metaharness replay-inloop-tb --jobs <job_dir> [<job_dir> ...] \
      --out <logs/<run>/vitals> [--subsample 6] [--agent NAME ...]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from vitals.addons.metaharness.config import CFG
from vitals.addons.metaharness.replay import log, manifest_spans
from vitals.addons.metaharness.replay.inloop import acquire_gpu_slot


def trial_reward(trial_dir: Path):
    try:
        d = json.loads((trial_dir / "result.json").read_text())
        vr = d.get("verifier_result") or {}
        r = (vr.get("rewards") or {}).get("reward", vr.get("reward")) \
            if isinstance(vr, dict) else None
        return r if r is not None else d.get("reward")
    except (OSError, json.JSONDecodeError):
        return None


def agent_of(job_dir: Path) -> str:
    """evolve-<name>-tN -> <name>; <baseline>-tN -> <baseline>."""
    name = job_dir.name
    name = re.sub(r"^evolve-", "", name)
    return re.sub(r"-t\d+$", "", name)


def pick_trials(job_dir: Path, subsample: int):
    """Deterministic, outcome-stratified: up to half passing, half failing,
    sorted order; falls back to whatever exists."""
    passed, failed = [], []
    for trial in sorted(job_dir.iterdir()):
        traj = trial / "agent" / "trajectory.json"
        if not traj.exists():
            continue
        r = trial_reward(trial)
        if r is None:
            continue
        (passed if r > 0 else failed).append((trial, float(r)))
    half = max(1, subsample // 2)
    picked = passed[:half] + failed[:half]
    for pool in (passed[half:], failed[half:]):
        for item in pool:
            if len(picked) >= subsample:
                break
            picked.append(item)
    return picked[:subsample]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", nargs="+", type=Path, required=True, help="harbor job dirs (one per candidate)")
    ap.add_argument("--out", type=Path, required=True, help="vitals root: stores land at <out>/<agent>--tb2/store")
    ap.add_argument("--subsample", type=int, default=6)
    ap.add_argument("--agent", nargs="*", default=None, help="agent names, one per --jobs entry (default: derived)")
    ap.add_argument("--turn-cap", type=int, default=999)
    ap.add_argument("--turn-head", type=int, default=0, help="replay only the first N turns of each trial (0 = all)")
    ap.add_argument("--turn-tail", type=int, default=0, help="... and the last M turns (0 = none); ingestion still covers every turn")
    ap.add_argument("--no-kv-reuse", action="store_true", help="re-prefill every turn from scratch (no prefix reuse)")
    ap.add_argument("--ingest-only", action="store_true", help="ingest, do not replay (no GPU)")
    ap.add_argument("--agents-dir", type=Path, default=None,
                    help="agents dir holding the candidates' declarations (default: <job>/../../../agents, the job's own tree)")
    args = ap.parse_args()
    import vitals.addons.metaharness.ingest.tb as _ingest_tb
    for jd in args.jobs:
        d = args.agents_dir or (Path(jd).resolve().parents[2] / "agents")
        if d.is_dir() and d not in _ingest_tb.EXTRA_AGENT_DIRS:
            _ingest_tb.EXTRA_AGENT_DIRS.append(d)
    args.out.mkdir(parents=True, exist_ok=True)

    from vitals.addons.metaharness.ingest import detok_engine
    from vitals.addons.metaharness.ingest.tb import ingest_trial

    detok = detok_engine()
    backend = None
    t0 = time.time()
    for k, job_dir in enumerate(args.jobs):
        if not job_dir.is_dir():
            log(f"SKIP {job_dir}: not a directory")
            continue
        agent = (args.agent[k] if args.agent and k < len(args.agent) else agent_of(job_dir))
        store = args.out / f"{agent}--tb2" / "store"
        picked = pick_trials(job_dir, args.subsample)
        if not picked:
            log(f"SKIP {job_dir.name}: no graded trials with trajectories")
            continue
        log(f"{agent}: {len(picked)} trials ({sum(r > 0 for _, r in picked)} pass)")
        # pass 1: ingest, collect what still needs a replay
        pending = []
        for trial, reward in picked:
            eid = f"ep-{trial.name}"
            epdir = store / eid
            if (epdir / "attn_tokens.parquet").exists():
                continue
            if not epdir.exists():
                ok = ingest_trial(trial / "agent" / "trajectory.json", store, eid,
                                  {"memory": agent, "example_idx": trial.name.split("__")[0],
                                   "reward": reward},
                                  eng=detok, turn_cap=args.turn_cap)
                if not ok:
                    log(f"  SKIP {trial.name}: unusable trajectory")
                    continue
            pending.append(eid)
        if args.ingest_only or not pending:
            continue
        if backend is None:
            from vitals.backends.llamacpp import LlamaCppBackend
            reuse = not args.no_kv_reuse
            log(f"loading the reference (fa-off, diagnostic, {'2 seqs x 48k, prefix reuse' if reuse else '48k ctx'})...")
            log(f"GPU slot {acquire_gpu_slot()}")
            # with prefix reuse the forced output runs on a scratch seq, so two
            # sequences share n_ctx: 96k total keeps the 48k per-turn capacity
            backend = LlamaCppBackend(str(CFG.reference_gguf), n_ctx=98304 if reuse else 49152,
                                      diagnostic_attention=True, n_seq_max=2 if reuse else 1)
        from vitals.replay import plan_episode_order, replay_episode
        # pass 2: replay, consecutive trials ordered to share prompt prefixes
        # (the scaffold head); each gets the next trial's first prompt so the
        # resident prefix is exactly what the next one can reuse
        plan = plan_episode_order(backend, store, pending) if not args.no_kv_reuse else [(e, None) for e in pending]
        for eid, nxt_prompt in plan:
            epdir = store / eid
            reqs = manifest_spans(epdir)
            sel = None
            if args.turn_head or args.turn_tail:
                ts = sorted(int(json.loads(line)["turn"])
                            for line in (epdir / "messages.jsonl").read_text(errors="replace").splitlines()
                            if line.strip())
                sel = set(ts[:args.turn_head]) | (set(ts[-args.turn_tail:]) if args.turn_tail else set())
            try:
                r = replay_episode(backend, store, eid, spans=reqs or None,
                                   channels=("attention", "logits"), token_attn=True,
                                   kv_reuse=not args.no_kv_reuse, turns=sel, next_prompt=nxt_prompt)
                tmr = r.get("timing") or {}
                log(f"  {eid}: {r.get('turns')} turns, {r.get('rows_updated')} rows"
                    + (f", {len(r['skipped'])} turns skipped" if r.get("skipped") else "")
                    + (f"; {tmr.get('seconds', 0):.0f}s, prompt {tmr.get('prompt_tokens', 0):,} tok of which "
                       f"{tmr.get('reused_tokens', 0):,} reused, {tmr.get('gen_tokens', 0):,} forced" if tmr else ""))
                with (args.out / "replay_timing.jsonl").open("a") as f:
                    f.write(json.dumps({"agent": agent, "episode": eid, "turns_replayed": r.get("turns"),
                                        "sampled": (args.turn_head, args.turn_tail) if sel is not None else None,
                                        **{k: v for k, v in tmr.items() if k != "turns"},
                                        "per_turn": tmr.get("turns")}) + "\n")
            except Exception as exc:
                log(f"  {eid}: REPLAY FAILED ({str(exc)[:100]}) -- continuing")
    log(f"done ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
