"""Monitoring commands: ``monitor``, ``ledger``, ``probe``, ``pipeline`` and ``explain``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from vitals.cli._common import (
    add_cache,
    add_episodes,
    add_model,
    add_store,
    csv,
    episodes,
    load_backend,
    turns,
)


def add_monitor(sub) -> None:
    p = sub.add_parser("monitor", help="checks over a trace store's turns with no labels and no model pass: "
                                       "low-confidence tool arguments, values found nowhere in the context, section "
                                       "drift against a reference store (alarms go to a cache, not into the store)")
    add_store(p)
    p.add_argument("--gguf", default=os.environ.get("VITALS_GGUF") or os.environ.get("VITALS_MH_REFERENCE_GGUF", ""),
                   help="the model's GGUF, for its tokenizer only (no model is loaded; default: $VITALS_GGUF)")
    add_episodes(p)
    p.add_argument("--budget", type=float, default=0.10,
                   help="false-alarm budget: the share of argument values below the warn threshold")
    p.add_argument("--reference", default="", help="reference store for section-drift alarms")
    p.add_argument("--cache-dir", default=None, help="read-map cache for drift (default: beside each episode)")
    add_cache(p, "alarms", what="alarm cache directory")
    p.add_argument("--json", action="store_true", help="print every alarm as JSON lines")
    p.set_defaults(func=run_monitor)


def run_monitor(args: argparse.Namespace) -> int:
    from vitals.monitor.checks import monitor_store
    from vitals.monitor.vocab import Vocab

    if not args.gguf:
        raise ValueError("--gguf (or $VITALS_GGUF) is needed for the tokenizer")
    res = monitor_store(args.store, Vocab.from_gguf(args.gguf), episodes=episodes(args.episodes), budget=args.budget,
                        reference=args.reference or None, out_dir=args.out_dir, cache_dir=args.cache_dir)
    alarms = res["alarms"]
    if args.json:
        for a in alarms:
            print(json.dumps(a, default=str))
        return 0
    th = res["thresholds"]
    warn = "-" if th["warn"] is None else f"{th['warn']:.2f}"
    print(f"{res['episodes']} episodes, {sum(len(r) for r in res['records'].values())} turns, "
          f"{th['n']} argument values; warn below log-probability {warn}")
    for (check, sev), n in sorted(Counter((a["check"], a["severity"]) for a in alarms).items()):
        print(f"  {check:<22} {sev:<6} {n}")
    order = {"alert": 0, "warn": 1, "info": 2}
    for a in sorted(alarms, key=lambda a: (order[a["severity"]], a["episode"], a["turn"] or 0))[:15]:
        print(f"[{a['severity']}] {a['episode']} turn {a['turn']}: {a['message']}")
    return 0


def add_ledger(sub) -> None:
    p = sub.add_parser("ledger", help="section ledger of a store: at each turn's decision, per-head reading of every "
                                      "section, the residual at a few layers, each section's mean residual and writes "
                                      "(teacher-forced passes; written to a cache, not into the store)")
    add_store(p)
    add_model(p, n_ctx=65536)
    add_episodes(p)
    p.add_argument("--turns", default="all", help="comma-separated turns, or all (every turn with an action onset)")
    p.add_argument("--layers", default="", help="comma-separated residual layers to capture (default: at 36%%, 48%%, "
                                                "61%% and 73%% of the model's depth; 23,31,39,47 on a 64-layer model)")
    add_cache(p, "ledger", what="ledger cache directory")
    p.add_argument("--force", action="store_true", help="recapture decisions already in the cache")
    p.add_argument("--no-writes", action="store_true", help="skip the per-section writes (the second pass that "
                                                            "lets a probe be split by section from the ledger)")
    p.add_argument("--linear", action="store_true", help="also split the linear-attention layers' writes (Gated "
                                                         "DeltaNet and similar)")
    p.set_defaults(func=run_ledger)


def run_ledger(args: argparse.Namespace) -> int:
    from vitals.core.store import episode_dirs
    from vitals.monitor.ledger import ledger_episode

    root = Path(args.store)
    eps = episodes(args.episodes) or [p.name for p in episode_dirs(root) if (p / "attn_tokens.parquet").exists()]
    if not eps:
        raise ValueError(f"no replayed episode with attn_tokens.parquet in {root}: run vitals replay first")
    layers = [int(x) for x in csv(args.layers)] or None
    backend = load_backend(args)
    n = 0
    try:
        for e in eps:
            rows = ledger_episode(backend, root, e, turns=turns(args.turns), residual_layers=layers,
                                  out_dir=args.out_dir, force=args.force,
                                  gguf_path=None if args.no_writes else args.gguf, linear=args.linear)
            n += len({r["turn"] for r in rows})
            print(f"{e}: {len({r['turn'] for r in rows})} decisions", flush=True)
    finally:
        backend.close()
    print(f"{n} decisions captured")
    return 0


def add_probe(sub) -> None:
    p = sub.add_parser("probe", help="behavior probes on the ledger: train one from labeled decisions, score a "
                                     "store's decisions with the section measures (probe lift, z against the normal)")
    psub = p.add_subparsers(dest="probe_cmd", required=True)
    t = psub.add_parser("train", help="fit a probe on ledger decisions; stores the threshold and each section's "
                                      "normal probe lift and read lift")
    add_store(t, "trace store the ledger was captured from")
    add_cache(t, "ledger", what="ledger cache directory")
    t.add_argument("--labels", required=True, help="CSV with columns episode, turn, label (0/1) and optionally group "
                                                  "(folds; default: episode)")
    t.add_argument("--name", required=True, help="behavior name")
    t.add_argument("--layer", type=int, required=True, help="residual layer, one of those the ledger captured "
                                                           "(vitals ledger --layers)")
    t.add_argument("--budget", type=float, default=0.10, help="false-alarm budget for the threshold")
    t.add_argument("--C", type=float, default=1e-3, help="logistic regression regularization")
    t.add_argument("--normal-on", choices=("negatives", "all"), default="negatives",
                   help="decisions that form each section's normals (probe lift and read lift)")
    t.add_argument("--out", required=True, help="probe JSON to write")
    s = psub.add_parser("score", help="score every ledger decision of a store and list the flagged ones")
    add_store(s, "trace store the ledger was captured from")
    add_cache(s, "ledger", what="ledger cache directory")
    s.add_argument("--probe", required=True, help="probe JSON (vitals probe train)")
    add_episodes(s)
    s.add_argument("--top", type=int, default=3, help="sections listed per decision")
    p.set_defaults(func=run_probe)


def run_probe(args: argparse.Namespace) -> int:
    import pandas as pd

    from vitals.core.io import write_jsonl
    from vitals.monitor.ledger import ledger_decisions, ledger_dir
    from vitals.monitor.probes import LinearProbe, ledger_inputs, ledger_reads, score_ledger, train_probe

    root = ledger_dir(args.out_dir, args.store)
    if not root.exists():
        raise ValueError(f"no ledger for {args.store} under {root.parent}: run vitals ledger first")
    if args.probe_cmd == "train":
        lab = pd.read_csv(args.labels)
        decs = {(d["episode"], d["turn"]): d for d in ledger_decisions(root, sorted(set(lab.episode.astype(str))))}
        lab = lab[[(str(e), int(t)) in decs for e, t in zip(lab.episode, lab.turn, strict=True)]]
        if lab.empty:
            raise ValueError("no labeled decision is in the ledger")
        chosen = [decs[(str(e), int(t))] for e, t in zip(lab.episode, lab.turn, strict=True)]
        H, splits = ledger_inputs(chosen, args.layer)
        groups = lab.group.astype(str).to_numpy() if "group" in lab else lab.episode.astype(str).to_numpy()
        probe, _ = train_probe(args.name, args.layer, H, lab.label.to_numpy(), groups,
                               splits=splits if all(sp is not None for sp in splits) else None,
                               C=args.C, budget=args.budget, normal_on=args.normal_on, reads=ledger_reads(chosen),
                               store=str(args.store))
        probe.save(args.out)
        m = probe.meta
        print(f"{args.name}: {m['n']} decisions, {m['positives']} positive; out-of-fold AUROC {m['oof_auroc']:.2f}; "
              f"threshold {probe.threshold:+.3f} at a {args.budget:.0%} false-alarm budget; section normal "
              f"{'over ' + str(len(m['normal']['median'])) + ' section keys' if 'normal' in m else 'missing (ledger without writes)'}"
              f"; read normal over {len(m['read_normal']['decision']['median']) if 'read_normal' in m else 0} section keys"
              f" -> {args.out}")
        return 0
    probe = LinearProbe.load(args.probe)
    recs = score_ledger(probe, ledger_decisions(root, episodes(args.episodes)), top=args.top)
    out = root / f"probe_{probe.name}.jsonl"
    write_jsonl(out, recs)
    flagged = [r for r in recs if r["flagged"]]
    print(f"{len(recs)} decisions scored, {len(flagged)} flagged -> {out}")
    for r in flagged[:20]:
        tops = "; ".join(f"{t['section']} (lift {t['lift']:+.1f}" + (f", z {t['z']:+.1f})" if t.get("z") is not None else ")")
                         for t in r.get("top", []) if t.get("lift") is not None)
        print(f"  {r['episode']} turn {r['turn']}: score {r['score']:+.2f}; {tops}")
    return 0


def add_pipeline(sub) -> None:
    p = sub.add_parser("pipeline", help="monitor a trace store end to end: replay, ledger, checks, probes, an "
                                        "explanation per flag (read map + probe split), flags for the dashboard "
                                        "and an optional webhook; --watch repeats it for live stores")
    add_store(p, "trace store directory (vitals import or vitals serve)")
    add_model(p, n_ctx=32768)
    p.add_argument("--probes", default="", help="comma-separated probe JSON files (vitals probe train)")
    p.add_argument("--layers", default="", help="ledger residual layers (default: those of the probes, else the "
                                                "ledger default); a probe's layer must be among them")
    p.add_argument("--reference", default="", help="reference store for section-drift alarms")
    p.add_argument("--confirm", action="store_true", help="confirm probe flags by ablation and add probe peaks (GPU)")
    p.add_argument("--webhook", default="", help="URL to POST each new flag to (JSON)")
    p.add_argument("--webhook-min", choices=("warn", "alert"), default="warn", help="lowest severity sent")
    p.add_argument("--watch", type=float, default=0.0, help="seconds between passes (0: run once)")
    add_episodes(p)
    p.add_argument("--budget", type=float, default=0.10, help="false-alarm budget of the argument checks")
    add_cache(p, "alarms", what="alarm and explanation cache")
    add_cache(p, "ledger", "--ledger-dir", "ledger cache")
    p.add_argument("--no-replay", action="store_true", help="skip replay (stores captured live by vitals serve)")
    p.set_defaults(func=run_pipeline)


def run_pipeline(args: argparse.Namespace) -> int:
    from vitals.monitor.pipeline import run_pipeline as run

    backend = load_backend(args, n_seq_max=2, diagnostic_attention=True)
    kw = dict(backend=backend, probes=csv(args.probes), reference=args.reference or None,
              confirm=args.confirm, webhook=args.webhook or None, webhook_min=args.webhook_min,
              out_dir=args.out_dir, ledger_out=args.ledger_dir, budget=args.budget,
              episodes=episodes(args.episodes), replay=not args.no_replay,
              layers=[int(x) for x in csv(args.layers)] or None)
    try:
        while True:
            r = run(args.store, args.gguf, **kw)
            print(f"{r['episodes']} episodes, {r['decisions']} decisions, {r['flags']} flags ({r['new']} new), "
                  f"{r['explained']} explained" + (f", webhook {r['webhook_sent']} sent / {r['webhook_failed']} failed"
                                                   if args.webhook else "") + f" in {r['seconds']}s -> {r['cache']}",
                  flush=True)
            if args.watch <= 0:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0
    finally:
        backend.close()


def add_explain(sub) -> None:
    p = sub.add_parser("explain", help="explain decisions of stored turns: which sections pushed the action onset, "
                                       "by reading, calibrated reading and exact ablation (results go to a cache, "
                                       "not into the store)")
    add_store(p)
    p.add_argument("--episode", required=True, help="episode id")
    p.add_argument("--turns", default="all", help="comma-separated turns, or all")
    add_model(p, n_ctx=65536)
    p.add_argument("--weights", default="", help="head weights JSON (vitals.analysis.calibration.save_weights); "
                                                 "default: rank by the read map")
    p.add_argument("--behavior", default="", help="behavior map JSON (vitals.analysis.calibration.fit_behavior_map)")
    p.add_argument("--ablate", "--top-k", dest="top_k", type=int, default=4, help="sections ablated per decision")
    add_cache(p, "explanations", what="explanation cache directory")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="print the full explanations as JSON")
    out.add_argument("--agent", action="store_true", help="print the compact form for agents (one JSON object per turn)")
    p.set_defaults(func=run_explain)


def run_explain(args: argparse.Namespace) -> int:
    import pandas as pd

    from vitals.explain import agent_view, explain_store

    ep = Path(args.store) / args.episode
    chosen = turns(args.turns)
    if chosen is None:
        at = ep / "attn_tokens.parquet"
        if not at.exists():
            raise ValueError(f"{ep} has no attn_tokens.parquet: replay the store first (vitals replay)")
        chosen = sorted(int(t) for t in pd.read_parquet(at, columns=["turn"]).turn.unique())
    weights = None
    if args.weights:
        from vitals.analysis.calibration import load_weights

        w, features, _ = load_weights(args.weights)
        weights = (w, features)
    behavior = json.loads(Path(args.behavior).read_text()) if args.behavior else None
    backend = load_backend(args, n_seq_max=2)
    try:
        rows = explain_store(backend, args.store, args.episode, chosen, out_dir=args.out_dir, weights=weights,
                             behavior=behavior, top_k=args.top_k)
    finally:
        backend.close()
    if args.json:
        print(json.dumps(rows, indent=1))
    elif args.agent:
        for r in rows:
            print(json.dumps(agent_view(r)))
    else:
        for r in rows:
            d = r["decision"]
            print(f"turn {r['turn']}: {d['a_text']!r} over {d['b_text']!r}, margin {d['margin']:+.2f}")
            for c in r["claims"]:
                print(f"  - {c}")
    if not rows:
        print("no turn with an action onset", file=sys.stderr)
    return 0
