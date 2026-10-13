"""What the dashboard shows, computed from stores and runs: episode lists, read maps, section text, store
matrices, monitor alarms, and Meta-Harness evolution."""

from __future__ import annotations

import json
import logging
import math
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vitals.analysis.drift import _natural, drift_report, episode_summary
from vitals.analysis.readmap import cached_readmap
from vitals.core.paths import cache_dir, store_cache
from vitals.core.sections import (  # noqa: F401  (group_of, parent_of: re-exported)
    CONVERSATION,
    INSTANCE,
    group_of,
    is_conversation,
    parent_of,
    section_info,
)
from vitals.core.store import episode_dirs, read_regions
from vitals.dashboard.sources import Run, Store, discover

log = logging.getLogger(__name__)

LIVE_SECONDS = 120
SCORE_KEYS = ("avg_pass_rate", "avg_val", "avg_score", "score", "val_accuracy")
_EP_FILES = ("manifest.json", "regions.jsonl", "attn_received.jsonl", "attn_tokens.parquet", "messages.jsonl")
_CELL_KEYS = ("turn", "section", "tokens", "lift", "peak", "hot_line", "consult_ratio", "consult_contrast")


def _clean(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, float | np.floating):
        return float(v) if math.isfinite(v) else None
    return v


def _num(v) -> float | None:
    """A number from manifest meta, where values are often stored as strings."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def cache_dir_or(given: str | Path | None, kind: str) -> Path:
    return Path(given).expanduser() if given else cache_dir(kind)


class EpisodeBusy(Exception):
    """An episode's files are being rewritten and could not be read consistently."""


def _episode_mtime(ep: Path) -> float:
    times = [(ep / f).stat().st_mtime for f in _EP_FILES if (ep / f).exists()]
    tokens = sorted(ep.glob("tokens-*.parquet"))
    if tokens:
        times.append(tokens[-1].stat().st_mtime)
    return max(times, default=0.0)


def _digest_files(run: Run) -> list[tuple[str, str | None, Path]]:
    """(candidate, dataset, file) for each digest: TB and HANDBOOK keep one per job, classification one per
    dataset. Smoke tests and holdout evaluations are not candidates."""
    found = []
    if run.jobs:
        for f in sorted(run.jobs.glob("*/vitals_digest.parquet")):
            job = f.parent.name
            if not job.startswith(("smoke-", "final-")):
                found.append((re.sub(r"-t\d+$", "", re.sub(r"^evolve-", "", job)), None, f))
    for f in sorted(run.path.glob("*/*/*/vitals_digest.parquet")):
        found.append((f.parents[1].name, f.parents[2].name, f))
    return found


def _run_mtime(run: Run) -> float:
    files = [run.path / "evolution_summary.jsonl"] + [f for _, _, f in _digest_files(run)]
    return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)


def _merge_instances(rows: list[dict]) -> list[dict]:
    """Instances of one named section (``demo:allergy#16``, ``demo:allergy#20``) as one row: lift is
    token-weighted (exact for the union), footprint and mass share add up, peak is the largest, consult
    values are mass-weighted."""
    families: dict[str, list[dict]] = {}
    for r in rows:
        families.setdefault(INSTANCE.sub("", r["section"]), []).append(r)
    out = []
    for name, rs in families.items():
        if len(rs) == 1 and rs[0]["section"] == name:
            out.append(rs[0])
            continue
        mass = [r["mass_share"] for r in rs if r.get("mass_share") is not None]
        out.append({"section": name, "footprint_tok": sum(r.get("footprint_tok") or 0.0 for r in rs),
                    "mass_share": sum(mass) if mass else None, "lift": _wmean(rs, "lift", "footprint_tok"),
                    "peak": max((r["peak"] for r in rs if r.get("peak") is not None), default=None),
                    "consult_ratio": _wmean(rs, "consult_ratio", "mass_share"),
                    "contrast": _wmean(rs, "contrast", "mass_share"), "instances": len(rs)})
    return out


def _wmean(rows: list[dict], key: str, weight: str) -> float | None:
    pairs = [(r[key], r.get(weight) or 0.0) for r in rows if r.get(key) is not None]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


def _meta(ep: Path) -> tuple[dict, str | None]:
    try:
        man = json.loads((ep / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}, None
    return man.get("meta") or {}, man.get("created_at")


class Dashboard:
    """Discovered sources plus cached, change-aware views over them."""

    def __init__(self, roots: list[str | Path], cache_dir: str | Path | None = None, explain_dir: str | Path | None = None,
                 alarm_dir: str | Path | None = None, *, warm: bool = True):
        """``roots``: stores, runs or directories to search. ``cache_dir``: read-map cache (default: beside each
        episode, ``cached_readmap``). ``explain_dir`` / ``alarm_dir``: where ``vitals explain`` and ``vitals
        pipeline`` wrote (default: the caches of ``vitals.core.paths``). ``warm``: compute missing read maps in a
        background thread."""
        self.roots = [Path(r) for r in roots]
        self.cache_dir = cache_dir
        self.explain_dir = cache_dir_or(explain_dir, "explanations")
        self.alarm_dir = cache_dir_or(alarm_dir, "alarms")
        self.stores: dict[str, Store] = {}
        self.runs: dict[str, Run] = {}
        self.version = 0
        self._mtimes: dict[tuple[str, str], float] = {}
        self._changed: list[tuple[int, str, str]] = []
        self._infos: dict[tuple[str, str], tuple[float, dict]] = {}
        self._summ: dict[tuple[str, str], tuple[float, dict]] = {}
        self._evo: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._queued: set[tuple[str, str]] = set()
        self.rescan()
        if warm:
            threading.Thread(target=self._warm_loop, daemon=True).start()

    # -- discovery and change tracking ---------------------------------------------------------------

    def rescan(self, sources: bool = True) -> None:
        stores, runs = discover(self.roots) if sources else (self.stores, self.runs)
        with self._lock:
            if set(stores) != set(self.stores) or set(runs) != set(self.runs):
                self.version += 1
                self._changed.append((self.version, "*", "*"))
            self.stores, self.runs = stores, runs
            for sid, store in stores.items():
                for ep in episode_dirs(store.path):
                    key, mt = (sid, ep.name), _episode_mtime(ep)
                    if self._mtimes.get(key) != mt:
                        self._mtimes[key] = mt
                        self.version += 1
                        self._changed.append((self.version, sid, ep.name))
            for rid, run in runs.items():
                key, mt = (f"run:{rid}", ""), _run_mtime(run)
                if self._mtimes.get(key) != mt:
                    self._mtimes[key] = mt
                    self.version += 1
                    self._changed.append((self.version, f"run:{rid}", ""))
            self._changed = self._changed[-2000:]

    def changes(self, since: int) -> dict:
        with self._lock:
            changed = [{"store": s, "episode": e} for v, s, e in self._changed if v > since]
            return {"version": self.version, "changed": changed[-500:]}

    def watch(self, interval: float = 2.0, discover_every: float = 30.0) -> None:
        """Poll the stores for new or growing episodes, and the roots for new stores and runs."""
        def loop():
            last = time.time()
            while True:
                time.sleep(interval)
                full = time.time() - last >= discover_every
                if full:
                    last = time.time()
                try:
                    self.rescan(sources=full)
                except OSError:
                    pass
        threading.Thread(target=loop, daemon=True).start()

    # -- background read-map computation -------------------------------------------------------------

    def _warm_loop(self) -> None:
        while True:
            sid, ep = self._queue.get()
            try:
                self._summary(sid, ep)
            except Exception:  # noqa: BLE001 - a broken episode must not stop the worker
                log.warning("read map of %s failed", ep, exc_info=True)
            finally:
                self._queued.discard((sid, ep.name))

    def _enqueue(self, sid: str, ep: Path) -> None:
        if (sid, ep.name) not in self._queued:
            self._queued.add((sid, ep.name))
            self._queue.put((sid, ep))

    # -- sources and episodes ------------------------------------------------------------------------

    def _ep(self, sid: str, eid: str) -> Path:
        root = self.stores[sid].path
        if not eid or eid.startswith(".") or "/" in eid or "\\" in eid:
            raise KeyError(eid)
        ep = root / eid
        if not ep.is_dir():
            raise KeyError(eid)
        return ep

    def _mtime(self, sid: str, ep: Path) -> float:
        return self._mtimes.get((sid, ep.name)) or _episode_mtime(ep)

    def episode_info(self, sid: str, ep: Path) -> dict:
        mt = self._mtime(sid, ep)
        hit = self._infos.get((sid, ep.name))
        if not hit or hit[0] != mt:
            meta, created = _meta(ep)
            info = {"episode": ep.name, "task": meta.get("example_idx") or meta.get("task_id") or meta.get("task"),
                    "reward": _num(meta.get("reward")), "turns": len(read_regions(ep)), "created": created, "updated": mt,
                    "replayed": (ep / "attn_received.jsonl").exists(), "fault": meta.get("fault")}
            hit = self._infos[(sid, ep.name)] = (mt, info)
        return {**hit[1], "live": time.time() - mt < LIVE_SECONDS}

    def episodes(self, sid: str) -> list[dict]:
        return [self.episode_info(sid, ep) for ep in episode_dirs(self.stores[sid].path)]

    def sources(self) -> dict:
        stores = []
        for sid, st in self.stores.items():
            infos = self.episodes(sid)
            rewards = [i["reward"] for i in infos if i["reward"] is not None]
            stores.append({"id": sid, "label": st.label, "path": str(st.path), "run": st.run,
                           "episodes": len(infos), "replayed": sum(i["replayed"] for i in infos),
                           "pass_rate": float(np.mean([r > 0 for r in rewards])) if rewards else None,
                           "updated": max((i["updated"] for i in infos), default=None),
                           "live": any(i["live"] for i in infos)})
        runs = [{"id": rid, "label": r.label, "path": str(r.path), "stores": r.stores} for rid, r in self.runs.items()]
        return {"stores": sorted(stores, key=lambda s: _natural(s["label"])),
                "runs": sorted(runs, key=lambda r: _natural(r["label"])), "version": self.version}

    # -- one episode ---------------------------------------------------------------------------------

    def _rows(self, ep: Path, compute: bool = True) -> list[dict] | None:
        """The episode's read map, retried once when a writer is mid-update; None when it is not cached and
        ``compute`` is off."""
        if not (ep / "attn_received.jsonl").exists():
            return []
        try:
            return cached_readmap(ep, self.cache_dir, compute=compute)
        except (OSError, ValueError, KeyError, IndexError):
            time.sleep(0.5)
            try:
                return cached_readmap(ep, self.cache_dir, compute=compute)
            except (OSError, ValueError, KeyError, IndexError) as e:
                raise EpisodeBusy(ep.name) from e

    def explanations(self, sid: str, eid: str) -> list[dict]:
        """Cached decision explanations of an episode (``vitals explain``), one per explained turn."""
        path = store_cache("explanations", self.stores[sid].path, self.explain_dir) / f"{self._ep(sid, eid).name}.jsonl"
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    def alarms(self, sid: str) -> dict:
        """The pipeline's flags for a store (``vitals pipeline``): check alarms and probe flags, newest run's state.
        ``pipeline`` is False when the pipeline has not run on this store."""
        cache = store_cache("alarms", self.stores[sid].path, self.alarm_dir)
        f = cache / "flags.jsonl"
        flags = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
        st = cache / "_pipeline_state.json"
        state = json.loads(st.read_text()) if st.exists() else {}
        return {"pipeline": f.exists(), "last_run": state.get("last_run"), "flags": flags, "cache": str(cache)}

    def flag_explanations(self, sid: str, eid: str) -> list[dict]:
        """The pipeline's explanations of an episode's flagged decisions: read map and probe split per turn."""
        path = store_cache("alarms", self.stores[sid].path, self.alarm_dir) / "explanations" / f"{self._ep(sid, eid).name}.jsonl"
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    def readmap(self, sid: str, eid: str) -> dict:
        ep = self._ep(sid, eid)
        rows = self._rows(ep)
        meta, _ = _meta(ep)

        start: dict[str, int] = {}
        for r in read_regions(ep).get(0, []):
            rid = str(r.get("id", ""))
            if rid.startswith("span:") and r.get("token_start") is not None:
                start.setdefault(rid[5:], int(r["token_start"]))
        first: dict[str, int] = {}
        for r in rows:
            first[r["section"]] = min(first.get(r["section"], r["turn"]), r["turn"])
        names = sorted(first, key=lambda s: (first[s], start.get(s, math.inf), _natural(s)))

        actions = meta.get("turn_actions") if isinstance(meta.get("turn_actions"), list) else []
        entropy = self._turn_entropy(ep)
        turns: dict[int, dict] = {}
        for r in rows:
            t = r["turn"]
            if t not in turns:
                a = actions[t] if t < len(actions) else None
                turns[t] = {"turn": t, "L": _clean(r.get("L")), "n_gen": _clean(r.get("n_gen")),
                            "entropy": entropy.get(t),
                            "action": (a.get("action") or a.get("tool")) if isinstance(a, dict) else a}
        return {"episode": self.episode_info(sid, ep), "turns": [turns[t] for t in sorted(turns)],
                "sections": [section_info(n) for n in names], "fields": list(_CELL_KEYS),
                "cells": [[_clean(r.get(k)) for k in _CELL_KEYS] for r in rows]}

    def _turn_entropy(self, ep: Path) -> dict[int, float]:
        parts = sorted(ep.glob("tokens-*.parquet"))
        if not parts:
            return {}
        try:
            tk = pd.concat([pd.read_parquet(p, columns=["turn", "region_id", "ent"]) for p in parts])
        except (OSError, KeyError, ValueError):
            return {}
        g = tk[tk.region_id == "gen"].groupby("turn").ent.mean()
        return {int(t): _clean(float(v)) for t, v in g.items()}

    def section_text(self, sid: str, eid: str, turn: int, section: str) -> dict:
        """The section's tokens at one turn, each with its read lift (its attention over the context mean)."""
        ep = self._ep(sid, eid)
        region = next((r for r in read_regions(ep).get(turn, []) if r.get("id") == f"span:{section}"), None)
        if region is None or not (ep / "attn_tokens.parquet").exists():
            return {"tokens": [], "reason": "No token-level reading is stored for this section."}
        at = pd.read_parquet(ep / "attn_tokens.parquet", columns=["turn", "idx", "token_str", "attn"],
                             filters=[("turn", "==", turn)])
        if at.empty:
            return {"tokens": [], "reason": "No token-level reading is stored for this turn."}
        mean = float(at.attn.sum()) / len(at) or 1.0
        s, e = int(region["token_start"]), int(region["token_end"])
        part = at[(at.idx >= s) & (at.idx < e)].sort_values("idx")
        return {"tokens": [[str(t), _clean(float(a) / mean)] for t, a in zip(part.token_str, part.attn, strict=True)],
                "start": s, "end": e}

    # -- a whole store -------------------------------------------------------------------------------

    def _summary(self, sid: str, ep: Path, compute: bool = True) -> dict | None:
        """Mean log2 lift and geometric-mean consult ratio per section over one episode's turns; None when
        the read map is not cached and ``compute`` is off."""
        mt = self._mtime(sid, ep)
        hit = self._summ.get((sid, ep.name))
        if hit and hit[0] == mt:
            return hit[1]
        try:
            rows = self._rows(ep, compute)
        except EpisodeBusy:
            return hit[1] if hit else {}
        if rows is None:
            return None
        out = episode_summary(rows)
        self._summ[(sid, ep.name)] = (mt, out)
        return out

    def summaries(self, sid: str, budget: float = 3.0) -> tuple[dict[str, dict], dict]:
        """Section summaries for the store's replayed episodes: those whose read maps are cached (read within
        the budget), with the rest queued for the background worker."""
        t0 = time.time()
        out, total = {}, 0
        for ep in episode_dirs(self.stores[sid].path):
            if not (ep / "attn_received.jsonl").exists():
                continue
            total += 1
            hit = self._summ.get((sid, ep.name))
            summ = hit[1] if hit and hit[0] == self._mtime(sid, ep) else None
            if summ is None and time.time() - t0 < budget:
                summ = self._summary(sid, ep, compute=False)
            if summ is None:
                self._enqueue(sid, ep)
            else:
                out[ep.name] = summ
        return out, {"done": len(out), "total": total}

    def matrix(self, sid: str) -> dict:
        summ, progress = self.summaries(sid)
        infos = {i["episode"]: i for i in self.episodes(sid)}
        eps = sorted(summ, key=lambda e: (infos[e].get("created") or "", _natural(e)))
        count: dict[str, int] = {}
        for e in eps:
            for n in summ[e]:
                count[n] = count.get(n, 0) + 1
        sections = sorted(count, key=lambda n: (is_conversation(n), -count[n], _natural(n)))
        cells = [[e, n, _clean(2 ** summ[e][n]["ll"]), _clean(summ[e][n]["consult"]), summ[e][n]["tokens"],
                  summ[e][n]["turns"]] for e in eps for n in sections if n in summ[e]]
        return {"episodes": [infos[e] for e in eps],
                "sections": [{**section_info(n), "episodes": count[n]} for n in sections],
                "fields": ["episode", "section", "lift", "consult_ratio", "tokens", "turns"], "cells": cells,
                "progress": progress}

    # -- monitor -------------------------------------------------------------------------------------

    def monitor(self, target: str, reference: str, threshold: float = 3.0, harness_only: bool = True,
                min_ref: int = 3) -> dict:
        """Each target episode's section reading against the reference store's normal.

        z is the episode's mean log2 lift for a section minus the reference median, over the reference
        spread (1.4826 x MAD, floored at 0.25 so near-constant sections do not alarm on noise); sections in
        fewer than ``min_ref`` reference episodes get no z. Declared sections present in at least 90% of
        reference episodes but absent from a target episode raise a "missing" alarm. Declared sections the
        reference never had raise a "new" alarm, unless their group's members normally vary across episodes
        (median presence under half, like per-task documents)."""
        tsum, tprog = self.summaries(target)
        rsum, rprog = self.summaries(reference) if reference != target else (tsum, tprog)
        infos = {i["episode"]: i for i in self.episodes(target)}
        order = sorted(tsum, key=lambda e: (infos[e].get("created") or "", _natural(e)))
        res = drift_report(tsum, rsum, threshold=threshold, harness_only=harness_only, min_ref=min_ref,
                           infos=infos, order=order)
        return {"reference": {"store": reference, "episodes": len(rsum), "progress": rprog},
                "target": {"store": target, "progress": tprog}, "threshold": threshold, "harness_only": harness_only,
                **res}

    # -- Meta-Harness evolution ----------------------------------------------------------------------

    def evolution(self, rid: str) -> dict:
        """A Meta-Harness run: its candidates from evolution_summary.jsonl (baselines, which have digests but
        no summary row, first), and each candidate's section reading from its digest."""
        run = self.runs[rid]
        mt = _run_mtime(run)
        hit = self._evo.get(rid)
        if hit and hit[0] == mt:
            return hit[1]
        rows = []
        for line in (run.path / "evolution_summary.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            per_task = r.get("per_task") if isinstance(r.get("per_task"), dict) else None
            rows.append({"iteration": r.get("iteration"), "name": r.get("agent") or r.get("system") or r.get("name"),
                         "score": _clean(next((r[k] for k in SCORE_KEYS if isinstance(r.get(k), int | float)), None)),
                         "parent": r.get("base_system"), "description": r.get("hypothesis") or r.get("changes"),
                         "changes": r.get("changes") if r.get("hypothesis") else None, "evidence": r.get("evidence"),
                         "outcome": r.get("outcome"), "delta": _clean(r.get("delta")), "axis": r.get("axis"),
                         "components": r.get("components") if isinstance(r.get("components"), list) else None,
                         "ctx_chars": r.get("ctx_chars"), "on_pareto": r.get("on_pareto"),
                         "per_task": {str(k): _num(v) for k, v in per_task.items()} if per_task else None})
        digests = self._digests(run)
        summarized = {r["name"] for r in rows}
        baselines = [{"iteration": None, "name": n, "score": None, "baseline": True} for n in digests if n not in summarized]
        candidates = baselines + rows
        count: dict[str, int] = {}
        for c in candidates:
            for d in digests.get(c["name"], []):
                count[d["section"]] = count.get(d["section"], 0) + 1
        sections = sorted(count, key=lambda n: (is_conversation(n), -count[n], _natural(n)))
        fields = ["candidate", "section", "lift", "peak", "consult_ratio", "consult_contrast", "tokens", "mass_share",
                  "instances"]
        cells = [[c["name"], d["section"], d.get("lift"), d.get("peak"), d.get("consult_ratio"), d.get("contrast"),
                  d.get("footprint_tok"), d.get("mass_share"), d.get("instances", 1)]
                 for c in candidates for d in digests.get(c["name"], [])]
        stores: dict[str, list[list[str | None]]] = {}
        for sid in run.stores:
            name, _, dataset = self.stores[sid].path.parent.name.partition("--")
            stores.setdefault(name, []).append([sid, dataset or None])
        scores = [c["score"] for c in rows if c["score"] is not None]
        datasets = sorted({ds for _, ds, _ in _digest_files(run) if ds})
        out = {"run": {"id": rid, "label": run.label, "score_scale": 1 if scores and max(scores) <= 1 else 100,
                       "datasets": datasets if len(datasets) > 1 else []},
               "candidates": candidates, "stores": stores,
               "matrix": {"sections": [{**section_info(n), "candidates": count[n]} for n in sections], "fields": fields,
                          "cells": cells}}
        self._evo[rid] = (mt, out)
        return out

    def _digests(self, run: Run) -> dict[str, list[dict]]:
        found = _digest_files(run)
        multi = len({ds for _, ds, _ in found if ds}) > 1
        out: dict[str, list[dict]] = {}
        for name, ds, f in found:
            try:
                df = pd.read_parquet(f)
            except (OSError, ValueError):
                continue
            for r in df.to_dict("records"):
                r = {k: _clean(v) for k, v in r.items()}
                if multi and ds:
                    r["section"] = f"{ds}:{r['section']}"
                out.setdefault(name, []).append(r)
        return {name: _merge_instances(rs) for name, rs in out.items()}
