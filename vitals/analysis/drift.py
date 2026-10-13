"""Section drift: each episode's section reading against a reference store's normal.

``episode_summary`` reduces an episode's read map (``vitals.analysis.readmap``) to one value per section: the mean
log2 read lift over its turns, with the geometric-mean consult ratio. ``drift_report`` compares target episodes with
the reference episodes' normal. The dashboard's Monitor tab and the ``drift`` check (``vitals.monitor.checks``) both
use it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from vitals.core.sections import group_of, is_conversation, section_info

SPREAD_FLOOR = 0.25     # log2 lift; near-constant sections would otherwise alarm on noise
MIN_REF = 3


def episode_summary(rows: list[dict]) -> dict[str, dict]:
    """Per section of one episode's read-map rows: ``ll`` (mean log2 read lift over its turns), ``consult``
    (geometric-mean consult ratio), median ``tokens`` and the number of ``turns``."""
    import pandas as pd

    out: dict[str, dict] = {}
    if not rows:
        return out
    df = pd.DataFrame(rows)
    df["ll"] = np.log2(df.lift.astype(float).clip(lower=1e-4))
    for name, x in df.groupby("section"):
        cr = x.consult_ratio.astype(float).dropna()
        cr = cr[cr > 0]
        out[name] = {"ll": float(x.ll.mean()), "consult": float(2 ** np.log2(cr).mean()) if len(cr) else None,
                     "tokens": int(x.tokens.median()), "turns": int(len(x))}
    return out


def drift_report(target: dict[str, dict], reference: dict[str, dict], *, threshold: float = 3.0,
                 harness_only: bool = True, min_ref: int = MIN_REF, infos: dict[str, dict] | None = None,
                 order: list[str] | None = None) -> dict:
    """Each target episode's section reading against the reference episodes' normal. ``target`` and
    ``reference`` map episode names to ``episode_summary`` output; ``infos`` adds per-episode fields (task,
    reward, updated) to the rows and alarms; ``order`` is the target episode order.

    z is the episode's mean log2 lift for a section minus the reference median, over the reference spread
    (1.4826 x MAD, floored at ``SPREAD_FLOOR``); sections in fewer than ``min_ref`` reference episodes get no z.
    Sections present in at least 90% of reference episodes but absent from a target episode raise a ``missing``
    alarm. Sections the reference never had raise a ``new`` alarm, unless their group's members normally vary
    across episodes (median presence under half, like per-task documents). Conversation sections are left out
    with ``harness_only``. Returns ``sections`` (the normal), ``episodes`` (z per section) and ``alarms``."""
    infos = infos or {}

    def keep(n: str) -> bool:
        return not (harness_only and is_conversation(n))

    stats: dict[str, dict] = {}
    for name in {n for secs in reference.values() for n in secs if keep(n)}:
        vals = {e: secs[name]["ll"] for e, secs in reference.items() if name in secs}
        arr = np.array(list(vals.values()))
        med = float(np.median(arr))
        spread = max(1.4826 * float(np.median(np.abs(arr - med))), SPREAD_FLOOR)
        stats[name] = {"median": med, "spread": spread, "n": len(arr), "present": len(arr) / len(reference),
                       "ref": [[e, round(2 ** v, 4)] for e, v in vals.items()]}

    presence: dict[str, list[float]] = {}
    for n, st in stats.items():
        presence.setdefault(group_of(n), []).append(st["present"])
    varying = {g for g, ps in presence.items() if float(np.median(ps)) < 0.5}

    episodes, alarms, new = [], [], {}
    for e in order if order is not None else sorted(target):
        info = infos.get(e, {"episode": e})
        secs = {n: s for n, s in target[e].items() if keep(n)}
        z = {n: (s["ll"] - stats[n]["median"]) / stats[n]["spread"] for n, s in secs.items()
             if n in stats and stats[n]["n"] >= min_ref}
        base = {"episode": e, "task": info.get("task"), "reward": info.get("reward"), "updated": info.get("updated")}
        for n, v in z.items():
            if abs(v) >= threshold:
                alarms.append({**base, "section": n, "kind": "under-read" if v < 0 else "over-read",
                               "z": round(v, 2), "lift": 2 ** secs[n]["ll"], "normal": 2 ** stats[n]["median"]})
        for n in sorted(secs.keys() - stats.keys()):
            if not is_conversation(n):
                new[n] = new.get(n, 0) + 1
                if group_of(n) not in varying:
                    alarms.append({**base, "section": n, "kind": "new", "z": None, "lift": 2 ** secs[n]["ll"],
                                   "normal": None})
        missing = [n for n, st in stats.items() if st["present"] >= 0.9 and n not in secs and not is_conversation(n)]
        for n in missing:
            alarms.append({**base, "section": n, "kind": "missing", "z": None, "lift": None,
                           "normal": 2 ** stats[n]["median"]})
        top = max(z, key=lambda n: abs(z[n]), default=None)
        episodes.append({**info, "max_z": round(abs(z[top]), 2) if top else None, "top": top,
                         "z": {n: round(v, 3) for n, v in z.items()}, "missing": missing,
                         "lift": {n: round(2 ** s["ll"], 4) for n, s in secs.items()}})
    known = sorted(stats, key=lambda n: (is_conversation(n), -stats[n]["present"], _natural(n)))
    sections = [{**section_info(n), "normal": 2 ** stats[n]["median"], "spread": stats[n]["spread"],
                 "n": stats[n]["n"], "present": stats[n]["present"], "ref": stats[n]["ref"],
                 "sparse": stats[n]["n"] < min_ref} for n in known]
    sections += [{**section_info(n), "new": True, "n": 0, "present": 0.0, "varying": group_of(n) in varying}
                 for n in sorted(new, key=lambda n: (-new[n], _natural(n)))]
    return {"sections": sections, "episodes": episodes, "alarms": alarms}


def creation_order(store: str | Path, names) -> list[str]:
    """Episode names by manifest creation time, then by name with numbers compared as numbers."""
    def created(e: str) -> str:
        try:
            return json.loads((Path(store) / e / "manifest.json").read_text()).get("created_at") or ""
        except (OSError, ValueError):
            return ""

    return sorted(names, key=lambda e: (created(e), _natural(e)))


def _natural(s: str) -> tuple:
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"(\d+)", s) if p)
