"""Checks and alarms over turn records, with no labels.

- ``uncertain_argument``: an argument value written with low confidence, i.e. its minimum value-token
  log-probability is below a percentile of all argument values in the store's history. The default false-alarm
  budget flags the lowest 10% (warn) and the lowest 2% (alert). Where the value came from (a user turn, tool results
  only, the harness text) is evidence in the alarm; severity does not depend on it.
- ``value_not_in_context``: a string argument whose value appears nowhere in the context (made up or computed).
- ``drift``: section reading against a reference store (``vitals.analysis.drift``, the statistics of the dashboard's
  Monitor tab), when a reference is given: info at the z threshold, warn at twice it, warn for a declared section
  missing or new.

Each alarm names the episode, turn, check, severity, the section responsible where there is one, the tool and
argument, the evidence, and a one-sentence message. ``monitor_store`` runs the checks over a store and caches the
alarms outside it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from vitals.core.io import atomic_write_json, write_jsonl
from vitals.core.paths import store_cache
from vitals.core.sections import is_own as _is_own
from vitals.core.sections import is_tool_result as _is_tool_result
from vitals.core.sections import is_user as _is_user

SEVERITY = {"info": 0, "warn": 1, "alert": 2}     # rank of each severity
_IDENT = re.compile(r"^[A-Za-z0-9_.:\-]{4,}$")


def identifier_like(value) -> bool:
    """An id-like string (letters and digits, no spaces: reservation ids, card ids, flight numbers). Dates, amounts
    and free text are left out: the model legitimately computes or paraphrases those."""
    return isinstance(value, str) and bool(_IDENT.match(value)) and any(ch.isdigit() for ch in value) \
        and any(ch.isalpha() for ch in value)


def source_of(found_in: list[str] | None) -> str | None:
    """Where a value came from, counting only sections the model did not write itself: ``user`` (a user turn holds
    it), ``tool_result`` (only tool results do), ``harness`` (only the system prompt, tool schemas or other harness
    text), ``none`` (not in the context, or only in the model's own earlier text), or None when unknown."""
    if found_in is None:
        return None
    ext = [n for n in found_in if not _is_own(n)]
    if not ext:
        return "none"
    if any(_is_user(n) for n in ext):
        return "user"
    if all(_is_tool_result(n) for n in ext):
        return "tool_result"
    if not any(_is_tool_result(n) for n in ext):
        return "harness"
    return "mixed"


def thresholds(records_by_episode: dict[str, list[dict]], budget: float = 0.10, alert_budget: float = 0.02) -> dict:
    """Percentile thresholds of the minimum value-token log-probability over every argument in the history."""
    lps = [a["min_lp"] for recs in records_by_episode.values() for r in recs for c in r["calls"] for a in c["args"]
           if a.get("min_lp") is not None and np.isfinite(a["min_lp"])]
    if len(lps) < 20:
        return {"warn": None, "alert": None, "n": len(lps), "budget": budget}
    return {"warn": float(np.quantile(lps, budget)), "alert": float(np.quantile(lps, alert_budget)), "n": len(lps),
            "budget": budget}


def _short(value, n: int = 60) -> str:
    s = value if isinstance(value, str) else json.dumps(value)
    return s if len(s) <= n else s[:n - 1] + "…"


def check_turn(episode: str, rec: dict, th: dict, tools: list[dict] | None = None) -> list[dict]:
    """The ``uncertain_argument`` and ``value_not_in_context`` alarms of one turn record (``turn_records``), given
    the store's thresholds (``thresholds``) and the turn's tool schemas (string-typed arguments only are checked
    for made-up values)."""
    types = {}
    for t in tools or []:
        fn = t.get("function", {})
        types[fn.get("name")] = {k: (v or {}).get("type") for k, v in ((fn.get("parameters") or {}).get("properties") or {}).items()}
    out = []
    for c in rec["calls"]:
        for a in c["args"]:
            base = {"episode": episode, "turn": rec["turn"], "tool": c["name"], "arg": a["name"], "value": a["value"]}
            src = source_of(a.get("found_in"))
            lp = a.get("min_lp")
            if lp is not None and not np.isfinite(lp):
                lp = None
            if lp is not None and th.get("warn") is not None and lp <= th["warn"]:
                sev = "alert" if lp <= th["alert"] else "warn"
                read = a.get("read_while_writing") or []
                ext = [n for n in (a.get("found_in") or []) if not _is_own(n)]
                section = ext[-1] if ext else None
                where = {"user": "a user turn", "tool_result": "tool results only", "harness": "the harness text only",
                         "mixed": "tool results and the harness text", "none": "nowhere outside the model's own text"
                         }.get(src, "an unknown place")
                out.append({**base, "check": "uncertain_argument", "severity": sev, "section": section,
                            "evidence": {"min_lp": round(lp, 3), "threshold": round(th["warn"], 3), "source": src,
                                         "found_in": a.get("found_in"), "read_while_writing": read},
                            "message": f"{c['name']}.{a['name']} = {_short(a['value'])} was written with low confidence "
                                       f"(log-probability {lp:.2f}); the value appears in {where}"
                                       + (f"; the model was reading {read[0][0]} while writing it." if read else ".")})
            if src == "none" and identifier_like(a["value"]) and types.get(c["name"], {}).get(a["name"], "string") == "string":
                out.append({**base, "check": "value_not_in_context", "severity": "warn", "section": None,
                            "evidence": {"min_lp": None if lp is None else round(lp, 3)},
                            "message": f"{c['name']}.{a['name']} = {_short(a['value'])} appears nowhere in the context: "
                                       "the model made it up or computed it."})
    return out


def drift_alarms(target: str | Path, reference: str | Path, *, threshold: float = 3.0, cache_dir=None) -> list[dict]:
    """Section reading of the target store's episodes against a reference store (``vitals.analysis.drift``, as in
    the dashboard's Monitor tab). ``cache_dir``: where read maps are cached (``cached_readmap``)."""
    from vitals.analysis.drift import creation_order, drift_report, episode_summary
    from vitals.analysis.readmap import cached_readmap
    from vitals.core.store import episode_dirs, is_episode_dir

    summaries = []
    for store in (target, reference):
        if not Path(store).is_dir() or not any(is_episode_dir(p) for p in Path(store).iterdir()):
            raise ValueError("target and reference must be trace stores")
        summaries.append({ep.name: episode_summary(cached_readmap(ep, cache_dir) or [])
                          for ep in episode_dirs(store) if (ep / "attn_received.jsonl").exists()})
    tsum, rsum = summaries if Path(target).resolve() != Path(reference).resolve() else (summaries[0], summaries[0])
    res = drift_report(tsum, rsum, threshold=threshold, order=creation_order(target, tsum))
    out = []
    for a in res["alarms"]:
        z = a.get("z")
        # drift catches blatant section faults but also fires on ordinary episode-to-episode variation
        sev = ("warn" if abs(z) >= 2 * threshold else "info") if z is not None else "warn"
        detail = f"z {z:+.1f}" if z is not None else a["kind"]
        out.append({"episode": a["episode"], "turn": None, "check": "drift", "severity": sev, "section": a["section"],
                    "tool": None, "arg": None, "value": None, "evidence": {"kind": a["kind"], "z": z, "lift": a.get("lift"),
                                                                           "normal": a.get("normal")},
                    "message": f"Section {a['section']} is {a['kind']} against the reference ({detail})."})
    return out


def alarm_cache(out_dir: str | Path | None, store: str | Path) -> Path:
    """The alarm cache directory of a store: ``<out_dir>/<store key>`` (``out_dir`` defaults to the alarms cache,
    ``vitals.core.paths``)."""
    return store_cache("alarms", store, out_dir)


def monitor_store(store: str | Path, vocab, *, episodes: list[str] | None = None, budget: float = 0.10,
                  reference: str | Path | None = None, out_dir: str | Path | None = None,
                  cache_dir=None) -> dict:
    """Turn records and checks for every (or the named) episode of a store; alarms are written to
    ``<out_dir>/<store key>/<episode>.jsonl`` and returned with the thresholds used."""
    from vitals.core.store import episode_dirs
    from vitals.monitor.turns import turn_records

    root = Path(store)
    names = episodes or [p.name for p in episode_dirs(root) if any(p.glob("tokens-*.parquet"))]
    records = {e: turn_records(root / e, vocab) for e in names}
    th = thresholds(records, budget)
    tools_of = {}
    for e in names:
        mp = root / e / "messages.jsonl"
        if mp.exists():
            tools_of[e] = {int(json.loads(x)["turn"]): json.loads(x).get("tools") for x in mp.read_text().splitlines() if x.strip()}
    alarms = []
    for e, recs in records.items():
        for r in recs:
            alarms += check_turn(e, r, th, (tools_of.get(e) or {}).get(r["turn"]))
    if reference is not None:
        alarms += drift_alarms(root, reference, cache_dir=cache_dir)
    cache = alarm_cache(out_dir, root)
    cache.mkdir(parents=True, exist_ok=True)
    by_ep: dict[str, list[dict]] = {e: [] for e in names}
    for a in alarms:
        by_ep.setdefault(a["episode"], []).append(a)
    for e, rows in by_ep.items():
        write_jsonl(cache / f"{e}.jsonl", rows, default=str)
    atomic_write_json(cache / "_thresholds.json", th)
    return {"thresholds": th, "alarms": alarms, "episodes": len(names), "records": records}
