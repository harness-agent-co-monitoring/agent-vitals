"""Explain on flag: one record per flagged decision that always carries both views.

- **Read map** (where the agent looked, whatever the behavior), from the ledger at two footings that both exist
  before the action, the decision token and the reasoning before it: per section the share and read lift at each,
  read z at each against this store's typical decision (``vitals.monitor.ledger.ledger_read_normals``, no labels),
  and the read peak with its hottest line over the reasoning. From the store's read map (when the episode was
  replayed) the turn's read lift, peak, hottest line and consult contrast. The rows are the top sections by read
  lift over the reasoning, by read z and by peak, so a section strong on any of them is shown.
- **Probe split** (which sections drive a flagged behavior): per probe, the score against its threshold and every
  section's contribution, probe lift and z against the section's normal (``vitals.monitor.measures``). Beside each
  row: the read share over all attention layers, the read share and read lift over the layers the probe split sees
  (those at or below the probe's layer), and read z against the decisions without the behavior (stored with the
  probe). Also the share of the score no section accounts for. Computed from the ledger, so no GPU pass is needed.
- **Confirmation** (optional, GPU): for each flagged probe, ``explain_flag`` replaces the top sections by z with
  filler and re-scores, and adds probe peaks with their hottest lines.

A flag is a check alarm at warn or alert (``vitals.monitor.checks``) or a probe score at or above its threshold.
"""

from __future__ import annotations

import numpy as np

from vitals.monitor.checks import SEVERITY  # noqa: F401  (re-exported)
from vitals.monitor.measures import (
    REASONING,
    TEMPLATE,
    read_lift,
    read_rows,
    read_z,
    section_keys,
    split_report,
)


def _top_by(rows: list[dict], key: str, k: int) -> list[dict]:
    return sorted((r for r in rows if r.get(key) is not None), key=lambda r: -r[key])[:k]


def read_view(dec: dict, turn_rows: list[dict] | None = None, top: int = 6, normals: dict | None = None) -> dict:
    """The read map of one ledger decision (see the module docstring). ``normals``: read normals by footing
    (``ledger_read_normals``). Rows are the union of the top ``top // 2`` sections by read lift over the reasoning
    (at the decision for ledgers without reasoning reading), by read z (reasoning, else decision) and by peak
    (reasoning, else turn), each with ``picked_by``, ordered by that read lift. ``best`` names the top section by
    each over all sections."""
    names = list(dec["sections"])
    rows = read_z(read_rows(names, dec["tokens"], dec["at"], dec.get("span")), normals, "read_z")
    peaks = (dec.get("meta") or {}).get("reason_peaks") or {}
    turn = {r["section"]: r for r in (turn_rows or [])}
    for r in rows:
        pk = peaks.get(r["section"]) or {}
        r.update(reason_peak=pk.get("peak"), reason_hot_line=pk.get("hot_line"))
        t = turn.get(r["section"])
        if t is not None:
            r.update(turn_lift=t.get("lift"), turn_peak=t.get("peak"), turn_hot_line=t.get("hot_line"),
                     consult_contrast=t.get("consult_contrast"))
    cand = [r for r in rows if r["key"] not in (TEMPLATE, REASONING)]
    has = {k: any(r.get(k) is not None for r in cand) for k in ("reasoning_lift", "read_z_reasoning", "reason_peak")}
    lift_key = "reasoning_lift" if has["reasoning_lift"] else "read_lift"
    z_key = "read_z_reasoning" if has["read_z_reasoning"] else "read_z_decision"
    peak_key = "reason_peak" if has["reason_peak"] else "turn_peak"
    k = max(1, top // 2)
    picked: dict[str, list[str]] = {}
    for label, key in (("lift", lift_key), ("z", z_key), ("peak", peak_key)):
        for r in _top_by(cand, key, k):
            picked.setdefault(r["section"], []).append(label)
    sel = [{**r, "picked_by": picked[r["section"]]} for r in cand if r["section"] in picked]
    sel.sort(key=lambda r: -(r.get(lift_key) or 0.0))
    best = {label: (_top_by(cand, key, 1) or [None])[0] for label, key in
            (("lift", lift_key), ("z", z_key), ("peak", peak_key))}
    ctx = np.array([r["key"] != REASONING for r in rows])
    return {"rows": sel, "lift_by": lift_key, "z_by": z_key, "peak_by": peak_key,
            "normal": "store" if normals else None,
            "best": {lab: (None if r is None else r["section"]) for lab, r in best.items()},
            "reasoning_share": float(sum(r["read"] for r, c in zip(rows, ctx, strict=True) if not c))}


def probe_view(dec: dict, probe, normal=None, top: int = 6, read_normals: dict | None = None) -> dict | None:
    """One probe at one ledger decision: score, threshold, the section split ranked by z (by contribution without a
    normal), each row with the read share over all attention layers (``read``), the read share and read lift over
    the attention layers at or below the probe's layer (``read_layers``, ``read_lift_layers``) and read z against
    the probe's read normal (``read_z_decision``, ``read_z_reasoning``; ``read_normals`` defaults to the one stored
    with the probe). None when the ledger holds no writes at the probe's layer."""
    if "writes" not in dec or probe.layer not in list(dec["residual_layers"]):
        return None
    k = list(dec["residual_layers"]).index(probe.layer)
    h = np.asarray(dec["residual_at"][k], dtype=np.float64)
    wl = dec["writes_linear"][k] if "writes_linear" in dec else None
    names = list(dec["sections"])
    rep = split_report(probe, h, np.asarray(dec["writes"][k], dtype=np.float64), names, dec["tokens"],
                       normal=normal, writes_linear=wl)
    at = np.asarray(dec["at"], dtype=np.float64)
    if read_normals is None:
        from vitals.monitor.probes import probe_read_normals

        read_normals = probe_read_normals(probe)
    reads = {r["section"]: r for r in read_z(read_rows(names, dec["tokens"], at, dec.get("span")), read_normals, "read_z")}
    layered = {}
    if dec.get("heads") is not None:
        sel = np.array([int(hd[0]) <= probe.layer for hd in dec["heads"]])
        if sel.any():
            share = at[sel].mean(0)
            lift = read_lift(share, dec["tokens"], section_keys(names))
            layered = {n: (float(share[i]), None if not np.isfinite(lift[i]) else float(lift[i])) for i, n in enumerate(names)}
    rows = []
    for r in rep["rows"]:
        if r["key"] in (TEMPLATE, REASONING):
            continue
        rd = reads[r["section"]]
        sh, lf = layered.get(r["section"], (None, None))
        rows.append({**r, "read": rd["read"], "read_layers": sh, "read_lift_layers": lf,
                     "read_z_decision": rd["read_z_decision"], "read_z_reasoning": rd["read_z_reasoning"]})
    thr = probe.threshold
    alert = probe.meta.get("alert_threshold")
    sev = "alert" if alert is not None and rep["score"] >= alert else "warn" if thr is not None and rep["score"] >= thr else None
    return {"probe": probe.name, "layer": probe.layer, "score": rep["score"], "threshold": thr, "severity": sev,
            "ranked_by": "z" if normal is not None else "contribution", "remainder_share": rep["remainder_share"],
            "read_normal": read_normals is not None, "rows": rows[:top]}


def _fmt_lift(v):
    return "n/a" if v is None else f"{v:+.1f}"


def _read_claim(read: dict) -> str | None:
    rows = {r["section"]: r for r in read["rows"]}
    best = read.get("best") or {}
    parts = []
    r = rows.get(best.get("lift"))
    if r is not None:
        if read["lift_by"] == "reasoning_lift":
            parts.append(f"before acting the agent reads '{r['section']}' most for its size (read lift "
                         f"{r['reasoning_lift']:.1f} over the reasoning, {_fmt_x(r['read_lift'])} at the decision)")
        else:
            parts.append(f"at the decision the agent reads '{r['section']}' most for its size (read lift "
                         f"{_fmt_x(r['read_lift'])}, {r['read']:.1%} of the attention)")
    r = rows.get(best.get("z"))
    if r is not None:
        foot = "over the reasoning" if read["z_by"] == "read_z_reasoning" else "at the decision"
        parts.append(f"against this store's decisions its reading of '{r['section']}' is the most unusual "
                     f"(z {r[read['z_by']]:+.1f} {foot})")
    r = rows.get(best.get("peak"))
    if r is not None:
        reasoning = read["peak_by"] == "reason_peak"
        line = r.get("reason_hot_line" if reasoning else "turn_hot_line")
        parts.append(f"the strongest peak {'over the reasoning' if reasoning else 'over the turn'} is in "
                     f"'{r['section']}' (peak lift {r[read['peak_by']]:.1f})" + (f": \"{line[:80]}\"" if line else ""))
    return "Read map: " + "; ".join(parts) + "." if parts else None


def _fmt_x(v):
    return "n/a" if v is None else f"{v:.1f}"


def claims(read: dict, probes: list[dict], alarms: list[dict], confirmed: dict | None = None) -> list[str]:
    """One-sentence claims of an explanation: the alarms' messages, the read map's strongest sections (by lift, z
    and peak), and per firing probe its most unusual section with read share and read z beside it."""
    out = [a["message"] for a in alarms]
    rc = _read_claim(read)
    if rc:
        out.append(rc)
    for p in probes:
        if p is None or p["severity"] is None:
            continue
        s = f"Probe '{p['probe']}' fires ({p['score']:+.2f} against threshold {p['threshold']:+.2f})."
        if p["rows"]:
            r = p["rows"][0]
            z = f", z {r['z']:+.1f} against its normal" if r.get("z") is not None else ""
            rz, foot = ((r.get("read_z_reasoning"), "over the reasoning") if r.get("read_z_reasoning") is not None
                        else (r.get("read_z_decision"), "at the decision"))
            rzs = f" and read z {rz:+.1f} {foot} for this behavior" if rz is not None else ""
            s += (f" Section most unusual for it: '{r['section']}' (probe lift {_fmt_lift(r['lift'])}{z}); the read map "
                  f"gives that section {r['read']:.1%} of the attention{rzs}.")
        s += f" {p['remainder_share']:.0%} of the score comes from outside the section split."
        out.append(s)
        for c in (confirmed or {}).get(p["probe"], {}).get("claims", []):
            out.append(c)
    return out


def explain_decision(dec: dict, probes: list, *, normals: list | None = None, turn_rows: list[dict] | None = None,
                     alarms: list[dict] | None = None, confirmed: dict | None = None, top: int = 6,
                     read_normals: dict | None = None) -> dict:
    """The explanation record of one ledger decision (``ledger.read_decision``): both views, the alarms raised at that turn, optional confirmations and the claims. ``read_normals``:
    the store's label-free read normals (``ledger_read_normals``) for read z in the read view."""
    normals = normals or [None] * len(probes)
    read = read_view(dec, turn_rows, top, normals=read_normals)
    pv = [probe_view(dec, p, n, top) for p, n in zip(probes, normals, strict=True)]
    alarms = alarms or []
    meta = dec.get("meta") or {}
    return {"episode": dec["episode"], "turn": int(dec["turn"]), "position": meta.get("position"),
            "decision": {"a_text": meta.get("a_text"), "b_text": meta.get("b_text")},
            "alarms": alarms, "read_map": read, "probes": [p for p in pv if p is not None],
            "confirmed": confirmed or {}, "claims": claims(read, [p for p in pv if p is not None], alarms, confirmed)}
