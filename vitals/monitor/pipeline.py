"""The monitoring pipeline: from a trace store (imported logs, ``vitals import``, or a live ``vitals serve`` store)
to flags with explanations, using one backend for every step.

Per run, for every new turn:

1. replay (read maps; turns already replayed or captured live are skipped);
2. ledger (per-head reading at each action onset and over the reasoning before it, reasoning peaks, residuals and
   per-section writes);
3. checks (``vitals.monitor.checks``) and, given a reference store, section drift;
4. probe scores for every given probe (``vitals.monitor.probes``);
5. an explanation for every flagged decision with both the read map and the probe split
   (``vitals.monitor.flags``), and with ``confirm`` the ablation check and probe peaks. Read z in the read view is
   against every decision in the store's ledger (no labels); beside the probe rows it is against the decisions
   without the behavior stored with the probe;
6. outputs: ``flags.jsonl`` and ``explanations/<episode>.jsonl`` in the alarm cache (the dashboard's Alarms panel
   and its API read them), and a webhook POST per new flag at or above a severity.

Everything is written to caches outside the store. Replay and the ledger skip turns they already hold; checks,
probe scores and flags are recomputed over the whole store on every run (they are cheap); explanations are reused
while their flags, probes and normals are unchanged. File formats: ``docs/reference/outputs.md``.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

from vitals.core.io import atomic_write_json, read_jsonl, write_jsonl
from vitals.monitor.flags import SEVERITY, explain_decision, probe_view

EXPLANATION_VERSION = 3     # cached explanations of another version are recomputed (3: read z fields are read_z_*)


def post_webhook(url: str, payload: dict, *, timeout: float = 5.0, retries: int = 1) -> bool:
    """POST one JSON payload; True on a 2xx answer. Never raises."""
    body = json.dumps(payload, default=str).encode()
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(url, body, {"Content-Type": "application/json", "User-Agent": "agent-vitals"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if 200 <= r.status < 300:
                    return True
        except Exception:  # noqa: BLE001 - a failed delivery is logged by the caller, never fatal
            time.sleep(0.5)
    return False


def flag_id(f: dict) -> str:
    """A flag's identity across runs: episode, turn, check, argument and value (not its severity)."""
    return hashlib.sha1(json.dumps([f["episode"], f.get("turn"), f["check"], f.get("arg"), str(f.get("value"))]).encode()).hexdigest()[:16]


def webhook_payload(store: Path, flag: dict, ex: dict | None) -> dict:
    """The JSON body posted for one flag: the flag's fields, and from its explanation the claims, the top 3 read
    map rows and each probe's top 3 rows."""
    out = {"source": "agent-vitals", "store": str(store), **{k: flag.get(k) for k in ("episode", "turn", "severity",
                                                                                       "check", "message")}}
    if ex:
        out["claims"] = ex["claims"]
        out["read_map"] = [{k: r.get(k) for k in ("section", "read", "read_lift", "reasoning_lift", "read_z_decision",
                                                  "read_z_reasoning", "reason_peak", "reason_hot_line")}
                           for r in ex["read_map"]["rows"][:3]]
        out["probes"] = [{"probe": p["probe"], "score": p["score"], "threshold": p["threshold"],
                          "top": [{k: r.get(k) for k in ("section", "lift", "z", "read", "read_layers",
                                                         "read_z_decision", "read_z_reasoning")} for r in p["rows"][:3]]}
                         for p in ex["probes"]]
    return out


def notify(state: dict, flags: list[dict], url: str, min_severity: str = "warn", *, store: Path | None = None,
           explanations: dict | None = None, post=post_webhook, log=print) -> dict:
    """POST every flag at or above ``min_severity`` not delivered before. ``state["notified"]`` holds what was
    delivered, by flag id and severity, so a failed post is tried again on the next run and a flag that escalates
    (warn to alert) is sent again. Returns counts sent and failed."""
    done = set(state.get("notified", []))
    legacy = {k for k in done if ":" not in k}          # state files before severity was recorded
    sent = failed = 0
    for f in flags:
        key = f"{f['id']}:{f['severity']}"
        if key in done or f["id"] in legacy or SEVERITY.get(f["severity"], 0) < SEVERITY.get(min_severity, 1):
            continue
        ex = (explanations or {}).get(f["episode"], {}).get(f.get("turn"))
        if post(url, webhook_payload(store or Path("."), f, ex)):
            sent += 1
            done.add(key)
        else:
            failed += 1
            log(f"[webhook] delivery failed for {f['episode']} turn {f.get('turn')} ({f['check']}); retried next run")
    state["notified"] = sorted(done)
    return {"sent": sent, "failed": failed}


def explanation_key(flag_ids: list[str], probes: list, read_normals: dict | None) -> str:
    """What a cached explanation depends on besides the ledger: its flags, the probes (weights, thresholds,
    normals) and the store's read normals."""
    from vitals.monitor.measures import normals_to_dict

    h = hashlib.sha1(json.dumps(sorted(flag_ids)).encode())
    for p in probes:
        h.update(json.dumps([p.name, p.layer, p.threshold, p.b, p.meta], sort_keys=True, default=str).encode())
        h.update(p.w.tobytes())
    h.update(json.dumps(normals_to_dict(read_normals) if read_normals else None, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def run_pipeline(store: str | Path, gguf: str, *, backend=None, probes=(), reference: str | Path | None = None,
                 confirm: bool = False, webhook: str | None = None, webhook_min: str = "warn",
                 out_dir: str | Path | None = None, ledger_out: str | Path | None = None,
                 episodes: list[str] | None = None, n_ctx: int = 32768, budget: float = 0.10, top: int = 6,
                 replay: bool = True, layers: list[int] | None = None, log=print) -> dict:
    """One pass over the store (see the module docstring). ``probes``: probe JSON paths or ``LinearProbe``s.
    ``backend``: reuse a loaded backend (fa off, ``n_seq_max >= 2``), else one is loaded and closed. ``layers``:
    ledger residual layers (default: the ledger default plus every probe's layer). ``out_dir`` and ``ledger_out``
    default to the alarm and ledger caches (``vitals.core.paths``)."""
    from vitals.core.store import episode_dirs
    from vitals.monitor.checks import alarm_cache, monitor_store
    from vitals.monitor.ledger import default_layers, ledger_dir, ledger_read_normals
    from vitals.monitor.probes import LinearProbe, ledger_decisions, probe_normal
    from vitals.monitor.vocab import Vocab

    store = Path(store)
    own = backend is None
    if own:
        from vitals.backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend(gguf, n_ctx=2 * n_ctx, n_seq_max=2, diagnostic_attention=True)
    t0 = time.time()
    try:
        loaded = [p if isinstance(p, LinearProbe) else LinearProbe.load(p) for p in probes]
        layers = sorted(set(layers or default_layers(backend.node_map)) | {p.layer for p in loaded})
        names = episodes or [p.name for p in episode_dirs(store) if any(p.glob("tokens-*.parquet"))]
        capture(backend, store, names, gguf, ledger_out=ledger_out, layers=layers, replay=replay, log=log)
        res = monitor_store(store, Vocab.from_gguf(gguf), episodes=names, budget=budget, reference=reference,
                            out_dir=out_dir)
        normals = [probe_normal(p) for p in loaded]
        root = ledger_dir(ledger_out, store)
        decs = ledger_decisions(root, names, log=log)
        for p in loaded:
            if decs and not any(p.layer in list(d["residual_layers"]) and "writes" in d for d in decs):
                log(f"[probes] '{p.name}' reads layer {p.layer}, which the ledger has no writes for: no flags from it")
        flags = collect_flags(res["alarms"], decs, loaded, normals)
        cache = alarm_cache(out_dir, store)
        by_ep, explained = explain_flags(backend, gguf, store, cache, flags, decs, loaded, normals,
                                         read_normals=ledger_read_normals(root), confirm=confirm, top=top)
        write_jsonl(cache / "flags.jsonl", flags, default=str)

        state_path = cache / "_pipeline_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"notified": []}
        known = {k.split(":")[0] for k in state.get("notified", [])} | set(state.get("seen", []))
        new = [f for f in flags if f["id"] not in known]
        sent = {"sent": 0, "failed": 0}
        if webhook:
            sent = notify(state, flags, webhook, webhook_min, store=store, explanations=by_ep, log=log)
        state["seen"] = sorted(known | {f["id"] for f in flags})
        state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(state_path, state)
        return {"episodes": len(names), "decisions": len(decs), "flags": len(flags), "new": len(new),
                "explained": explained, "webhook_sent": sent["sent"], "webhook_failed": sent["failed"],
                "cache": str(cache), "seconds": round(time.time() - t0, 1)}
    finally:
        if own:
            backend.close()


def capture(backend, store: Path, names: list[str], gguf: str, *, ledger_out=None, layers=None, replay: bool = True,
            log=print) -> None:
    """Steps 1 and 2: replay what is not replayed yet (with per-token attention), then the ledger of every replayed
    episode."""
    from vitals.monitor.ledger import ledger_episode

    if replay:
        from vitals.replay.episode import plan_episode_order, replay_episode

        for name, nxt in plan_episode_order(backend, store, names):
            r = replay_episode(backend, store, name, token_attn=True, kv_reuse=True, next_prompt=nxt)
            if r.get("turns"):
                log(f"[replay] {name}: {r['turns']} turns")
    for name in names:
        if (store / name / "attn_tokens.parquet").exists():
            rows = ledger_episode(backend, store, name, gguf_path=gguf, out_dir=ledger_out, residual_layers=layers,
                                  log=log)
            if rows:
                log(f"[ledger] {name}: {len({r['turn'] for r in rows})} decisions")


def collect_flags(alarms: list[dict], decisions: list[dict], probes: list, normals: list) -> list[dict]:
    """Steps 3 and 4: check alarms at warn or above, and a flag per decision where a probe scores at or above its
    threshold; each flag gets its ``id`` (``flag_id``)."""
    flags = [{**a, "kind": "check"} for a in alarms if SEVERITY.get(a.get("severity"), 0) >= 1]
    for p, n in zip(probes, normals, strict=True):
        for d in decisions:
            pv = probe_view(d, p, n, top=3)
            if pv and pv["severity"]:
                flags.append({"episode": d["episode"], "turn": d["turn"], "check": f"probe:{p.name}", "kind": "probe",
                              "severity": pv["severity"], "score": pv["score"],
                              "message": f"Probe '{p.name}' scores {pv['score']:+.2f} (threshold {pv['threshold']:+.2f})."})
    for f in flags:
        f["id"] = flag_id(f)
    return flags


def explain_flags(backend, gguf: str, store: Path, cache: Path, flags: list[dict], decisions: list[dict],
                  probes: list, normals: list, *, read_normals=None, confirm: bool = False,
                  top: int = 6) -> tuple[dict[str, dict[int, dict]], int]:
    """Step 5: an explanation per flagged decision, reused from ``explanations/<episode>.jsonl`` while its key
    (``explanation_key``) and version are unchanged. Marks each flag ``explained``. Returns the explanations of
    the episodes touched, by episode and turn, and how many were computed."""
    from vitals.analysis.readmap import cached_readmap

    exp_dir = cache / "explanations"
    exp_dir.mkdir(parents=True, exist_ok=True)
    by_turn = {(d["episode"], d["turn"]): d for d in decisions}
    readmaps: dict[str, list[dict]] = {}
    by_ep: dict[str, dict[int, dict]] = {}
    explained = 0
    for ep, turn in sorted({(f["episode"], f["turn"]) for f in flags if f.get("turn") is not None}):
        d = by_turn.get((ep, turn))
        if d is None:
            continue                                       # no action onset captured at that turn
        if ep not in by_ep:
            by_ep[ep] = {r["turn"]: r for r in read_jsonl(exp_dir / f"{ep}.jsonl")}
        here = [f for f in flags if f["episode"] == ep and f.get("turn") == turn]
        flagged = [p for p in probes if any(f["check"] == f"probe:{p.name}" for f in here)]
        key = explanation_key([f["id"] for f in here], probes, read_normals)
        old = by_ep[ep].get(turn)
        if old and old.get("key") == key and old.get("version") == EXPLANATION_VERSION \
                and (old.get("confirmed") or not confirm or not flagged):
            continue
        if ep not in readmaps:
            readmaps[ep] = cached_readmap(store / ep) or []
        turn_rows = [r for r in readmaps[ep] if int(r["turn"]) == turn]
        confirmed = _confirm(backend, gguf, store / ep, turn, flagged) if confirm and flagged else {}
        ex = explain_decision(d, probes, normals=normals, turn_rows=turn_rows,
                              alarms=[f for f in here if f["kind"] == "check"], confirmed=confirmed, top=top,
                              read_normals=read_normals)
        ex["flag_ids"], ex["key"], ex["version"] = sorted(f["id"] for f in here), key, EXPLANATION_VERSION
        by_ep[ep][turn] = ex
        explained += 1
    for ep, rows in by_ep.items():
        write_jsonl(exp_dir / f"{ep}.jsonl", (rows[t] for t in sorted(rows)), default=str)
    have = {(e, t) for e, rows in by_ep.items() for t in rows}
    for ep in {f["episode"] for f in flags} - set(by_ep):
        have |= {(ep, r["turn"]) for r in read_jsonl(exp_dir / f"{ep}.jsonl")}
    for f in flags:
        f["explained"] = (f["episode"], f.get("turn")) in have
    return by_ep, explained


def _confirm(backend, gguf: str, ep_dir: Path, turn: int, probes: list) -> dict:
    """``explain_flag`` for each probe at the turn's action onset (GPU: ablation of the top sections by z and
    probe peaks)."""
    from vitals.core.store import turn_inputs, turn_prompt_text
    from vitals.explain import auto_decision
    from vitals.monitor.probes import explain_flag

    inp = turn_inputs(ep_dir, turn)
    d = auto_decision(backend, inp)
    if d is None:
        return {}
    j, pos, _, _ = d
    L = len(inp["prompt"])
    tokens, labels = inp["prompt"] + inp["output"][:j + 1], inp["labels"][:L + j + 1]
    out = {}
    for p in probes:
        try:
            out[p.name] = explain_flag(backend, gguf, p, tokens, labels, pos, prompt_text=turn_prompt_text(ep_dir, turn))
        except Exception as e:  # noqa: BLE001 - a failed confirmation leaves the ledger-based explanation in place
            out[p.name] = {"error": f"{type(e).__name__}: {e}"[:300], "claims": []}
    return out
