"""One-shot in-loop vitals replay for the classification optimizer loop.

Called by the text-classification meta_harness.py (``--vitals``) after each
iteration's benchmark: ingests the just-produced val.json files for the
iteration's systems, replays a fixed-index subsample of episodes through the
BF16 reference (attention + logits, fa-off), and emits the raw store --
including the per-prompt-token attn_tokens.parquet -- under the run's vitals
root, where the NEXT propose step can analyze it with vitals_data.py.

It is synchronous and one-shot (loads the backend, replays the named systems,
exits and frees the GPU), writes the raw per-token interface (token_attn=True),
and uses the store layout <out>/<system>--<dataset>/store across iterations.

Idempotent: episodes already carrying attn_tokens.parquet are skipped, so
re-runs after a crash are cheap.

Usage:
  vitals metaharness replay-inloop --logs-dir <.../logs/<run>> \
      --systems cand_a cand_b --out <.../logs/<run>/vitals> --subsample 12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from vitals.addons.metaharness.config import CFG
from vitals.addons.metaharness.replay import log

_EP_IDX = re.compile(r"-(\d{4})-c\d+$")


def fixed_index_subset(n: int, max_idx: int) -> set[int]:
    """The run-wide set of val example indices to replay: n evenly-spaced
    indices over [0, max_idx]. Deterministic and identical for every store
    (same seed/num_val => example i is the same input across all designs), so
    the corpus becomes a panel -- every harness measured on the SAME examples,
    enabling within-example paired comparison across designs."""
    if n <= 0 or n > max_idx + 1:
        return set(range(max_idx + 1))
    step = (max_idx + 1) / n
    return {int(i * step) for i in range(n)}


def subsample_episodes(made: list[str], n: int) -> list[str]:
    """Keep only episodes whose val-example index is in the run-wide fixed
    subset (see fixed_index_subset). All calls (c0, c1, ...) of a kept example
    are retained so multi-call designs stay intact. Returns `made` unchanged
    when n<=0 or the store has <= n examples."""
    idxs = []
    for eid in made:
        m = _EP_IDX.search(eid)
        idxs.append(int(m.group(1)) if m else None)
    present = sorted({i for i in idxs if i is not None})
    if n <= 0 or not present or n >= len(present):
        return made
    keep = fixed_index_subset(n, present[-1])
    return [eid for eid, i in zip(made, idxs, strict=True) if i is None or i in keep]


# replay processes allowed to hold a resident reference model on the GPU at once
GPU_SLOTS = int(os.environ.get("VITALS_GPU_SLOTS", "2"))
_slot_handles: list = []   # held for the life of the process


def acquire_gpu_slot() -> int:
    """Block until one of GPU_SLOTS host-wide replay slots is free. Replays from
    the in-loop hook and from proposer probes share the card; a third resident
    model would not fit. The lock is a file lock, released when the process
    exits."""
    import fcntl

    lock_dir = Path(os.environ.get("VITALS_LOCK_DIR", "/tmp/vitals-gpu"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    waited = 0
    while True:
        for i in range(GPU_SLOTS):
            fh = open(lock_dir / f"slot-{i}.lock", "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                continue
            _slot_handles.append(fh)
            return i
        if waited == 0:
            log(f"all {GPU_SLOTS} GPU replay slots busy; waiting...")
        time.sleep(10)
        waited += 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", type=Path, required=True,
                    help="the run's logs dir (contains <dataset>/<system>/<model>/val.json)")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="vitals root; stores land at <out>/<system>--<dataset>/store")
    ap.add_argument("--subsample", type=int, default=12)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="restrict to these datasets (default: every dataset with a val.json)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from vitals.addons.metaharness.contracts.tc import episode_span_requests
    from vitals.addons.metaharness.ingest.tc import ingest
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.replay import plan_episode_order, replay_episode

    jobs = []  # (system, dataset, val_json)
    for system in args.systems:
        for vj in sorted(args.logs_dir.glob(f"*/{system}/*/val.json")):
            if args.datasets and vj.parts[-4] not in args.datasets:
                continue
            jobs.append((system, vj.parts[-4], vj))
    if not jobs:
        log(f"no val.json found for systems={args.systems} under {args.logs_dir}")
        return

    backend = None
    for system, dataset, vj in jobs:
        store = args.out / f"{system}--{dataset}" / "store"
        try:
            doc = json.loads(vj.read_text())
        except json.JSONDecodeError:
            log(f"SKIP {system}/{dataset}: val.json unreadable")
            continue
        if not doc.get("predictions"):
            log(f"SKIP {system}/{dataset}: no per-prediction records "
                "(TC_REPLAY_CAPTURE off?)")
            continue
        existing = sorted(p.name for p in store.glob("ep-*"))
        if existing:
            # already ingested (crash-and-retry of this iteration): never
            # re-ingest, it would append duplicate token rows
            made = existing
            log(f"REUSE {system} / {dataset} ({len(existing)} episodes ingested)")
        else:
            log(f"INGEST {system} / {dataset} (acc={doc.get('accuracy')}, "
                f"n={doc.get('total')})")
            try:
                made = ingest(vj, store, tag=system[:18])
            except SystemExit as exc:
                log(f"  ingest skipped: {exc}")
                continue
        to_replay = subsample_episodes(made, args.subsample)
        pending = [e for e in to_replay
                   if not (store / e / "attn_tokens.parquet").exists()]
        if not pending:
            log(f"  all {len(to_replay)} subsampled episodes already replayed")
            continue
        if backend is None:
            slot = acquire_gpu_slot()
            log(f"loading the reference backend (fa-off, diagnostic) in GPU slot {slot}...")
            # 2 seqs x 32k: seq 0 keeps the prefix shared between consecutive
            # episodes (a design's scaffold head), the forced output runs on a scratch seq
            backend = LlamaCppBackend(str(CFG.reference_gguf), n_ctx=65536, n_seq_max=2,
                                      diagnostic_attention=True)
        t0, n_rows = time.time(), 0
        reused = 0
        for eid, nxt in plan_episode_order(backend, store, list(pending)):
            reqs = episode_span_requests(store / eid)
            r = replay_episode(backend, store, eid, spans=reqs or None,
                               channels=("attention", "logits"),
                               token_attn=True, kv_reuse=True, next_prompt=nxt)
            n_rows += r.get("rows_updated", 0)
            reused += (r.get("timing") or {}).get("reused_tokens", 0)
        log(f"  REPLAYED {len(pending)}/{len(to_replay)} episodes, "
            f"{n_rows} rows ({time.time() - t0:.0f}s, {reused:,} prompt tokens reused across episodes)")
        try:   # which pooling path ran (GPU head-mean, or the host fallback and why), and whether head weights were on
            from vitals.engine.binding import LlamaEngine as _LE
            log(f"  attention pooling: {'GPU head-mean' if isinstance(_LE._gpu_pool_state, dict) else 'host fallback'}"
                f"{'' if _LE._gpu_pool_state else ' (' + str(getattr(_LE, '_gpu_pool_reason', None)) + ')'}; "
                f"head weights: {os.environ.get('VITALS_HEAD_WEIGHTS') or 'uniform'}")
        except Exception:
            pass

    log("done")


if __name__ == "__main__":
    main()
