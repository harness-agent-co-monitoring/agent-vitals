"""Section measures for one decision, from the read map and from a probe split.

Read map (where the agent looked), from the ledger's head-mean reading at two footings: the decision token and the
reasoning before it (every step up to and including the decision, so both exist before the action):

  share         the section's share of the attention.
  read lift     share over the context's share, per token, over the uniform 1/L. The turn's own reasoning so far is
                left out of both.
  read z        log2 read lift against the section's normal over reference decisions, (log2 lift - median) /
                max(1.4826 MAD, READ_FLOOR). The normal comes from every decision of a store (no labels) or from
                decisions without a behavior (stored with a probe).
  read peak     the mean lift of the section's 8 most-read content tokens over the reasoning, and the line holding
                them (``section_peak``, computed by the ledger).

A probe split gives every section s a signed contribution c_s to a probe's score (``split_report``): the
probe direction applied to what the section wrote into the hidden state. It is reported in the same measures as a
read map, so the two can be read side by side:

  contribution  c_s and its share c_s / sum |c|: the analog of attention mass.
  probe lift    (c_s / |s|) / (sum |c| / L): per-token contribution over the uniform 1/L, signed. Like read lift it
                does not favor long sections. L and the sum run over the context; the turn's own reasoning so far
                is reported but left out of both.
  z             probe lift against the section's normal over reference decisions, (lift - median) /
                max(1.4826 MAD, floor). Raw contributions rank the same sections first in every decision (a rule
                that always writes along the probe direction ranks first whether or not it matters here); z says
                what is unusual at this decision.
  probe peak    the mean per-position contribution of the section's 8 strongest content tokens (in the direction
                of c_s) over sum |c_p| / L over the positions given, and the prompt line holding them
                (``section_peak``). Sections under 32 tokens report none.

A normal needs the same key for a section across decisions (``vitals.core.sections.section_keys``): harness
sections keep their name; conversation sections become role (and tool) plus recency, e.g. ``user:latest``,
``tool_result:get_order:earlier``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from vitals.analysis.readmap import PEAK_K, PEAK_MIN_TOK, content_token, hottest_line
from vitals.core.sections import CONVERSATION, REASONING, TEMPLATE, section_keys  # noqa: F401  (re-exported)

# Floor on the spread of log2 read lift for read z at one decision. A spread under one doubling of lift is noise at a
# single decision; a lower floor lets sections whose reading barely varies dominate the z ranking.
READ_FLOOR = 1.0


def split_measures(contrib, sections: list[str], tokens, *, keys: list[str] | None = None) -> list[dict]:
    """One row per section: key, tokens, contribution, share and probe lift. ``contrib`` [R] signed contributions,
    ``tokens`` [R] each section's token count up to the decision."""
    c = np.asarray(contrib, dtype=np.float64)
    n = np.asarray(tokens, dtype=np.float64)
    keys = keys or section_keys(sections)
    ctx = np.array([k != REASONING for k in keys])
    abs_ctx = float(np.abs(c[ctx]).sum()) or 1.0
    L = float(n[ctx].sum()) or 1.0
    abs_all = float(np.abs(c).sum()) or 1.0
    rows = []
    for i, name in enumerate(sections):
        lift = (c[i] / n[i]) / (abs_ctx / L) if ctx[i] and n[i] > 0 else None
        rows.append({"section": name, "key": keys[i], "tokens": int(n[i]), "contribution": float(c[i]),
                     "share": float(c[i] / abs_all), "lift": None if lift is None else float(lift)})
    return rows


@dataclass
class Normal:
    """Each key's typical value of a measure over reference decisions: median and 1.4826 × MAD, the spread floored
    so near-constant sections do not produce large z on noise. Keys seen in fewer than ``min_ref`` decisions get
    no z."""

    median: dict[str, float] = field(default_factory=dict)
    spread: dict[str, float] = field(default_factory=dict)
    n: dict[str, int] = field(default_factory=dict)
    value: str = "lift"
    min_ref: int = 5

    @classmethod
    def fit(cls, rows: Iterable[dict], *, value: str = "lift", min_ref: int = 5, floor: float | None = None,
            floor_frac: float = 0.1) -> Normal:
        """``rows``: section rows from many decisions (``split_measures`` output). ``floor`` defaults to
        ``floor_frac`` times the median spread over the keys with enough decisions."""
        vals: dict[str, list[float]] = {}
        for r in rows:
            v = r.get(value)
            if v is not None and np.isfinite(v):
                vals.setdefault(r["key"], []).append(float(v))
        med = {k: float(np.median(v)) for k, v in vals.items()}
        raw = {k: 1.4826 * float(np.median(np.abs(np.asarray(v) - med[k]))) for k, v in vals.items()}
        enough = [raw[k] for k, v in vals.items() if len(v) >= min_ref]
        if floor is None:
            floor = floor_frac * float(np.median(enough)) if enough else 0.0
        floor = max(float(floor), 1e-9)
        return cls(med, {k: max(s, floor) for k, s in raw.items()}, {k: len(v) for k, v in vals.items()}, value, min_ref)

    def z(self, key: str, v: float | None) -> float | None:
        if v is None or self.n.get(key, 0) < self.min_ref:
            return None
        return (float(v) - self.median[key]) / self.spread[key]

    def to_dict(self) -> dict:
        return {"median": self.median, "spread": self.spread, "n": self.n, "value": self.value, "min_ref": self.min_ref}

    @classmethod
    def from_dict(cls, d: dict) -> Normal:
        return cls(dict(d["median"]), dict(d["spread"]), {k: int(v) for k, v in d["n"].items()}, d.get("value", "lift"),
                   int(d.get("min_ref", 5)))


def read_lift(share, tokens, keys: list[str]) -> np.ndarray:
    """Per-section read lift from the shares at one footing: share over the context's share of the attention, per
    token, over 1/L. The turn's own reasoning so far is left out of both; it and empty sections get nan."""
    a = np.asarray(share, dtype=np.float64)
    n = np.asarray(tokens, dtype=np.float64)
    ctx = np.array([k != REASONING for k in keys])
    mass = float(a[ctx].sum()) or 1.0
    L = float(n[ctx].sum()) or 1.0
    return np.where(ctx & (n > 0), a / mass * L / np.where(n > 0, n, 1.0), np.nan)


def _num(v) -> float | None:
    return float(v) if v is not None and np.isfinite(v) else None


def _log2(v) -> float | None:
    return float(np.log2(v)) if v is not None and np.isfinite(v) and v > 0 else None


def read_rows(sections: list[str], tokens, at, span=None, *, keys: list[str] | None = None) -> list[dict]:
    """One row per section, in prompt order: key, tokens, ``read`` and ``read_lift`` at the decision,
    ``read_reasoning`` and ``reasoning_lift`` over the reasoning before it, and the log2 lifts read z is computed
    on. ``at`` and ``span`` are shares per section, or per head [n_heads, R] (averaged here, the read map)."""
    keys = keys or section_keys(list(sections))
    a = np.asarray(at, dtype=np.float64)
    a = a.mean(0) if a.ndim == 2 else a
    s = None
    if span is not None:
        s = np.asarray(span, dtype=np.float64)
        s = s.mean(0) if s.ndim == 2 else s
    la = read_lift(a, tokens, keys)
    ls = read_lift(s, tokens, keys) if s is not None else np.full(len(keys), np.nan)
    n = np.asarray(tokens)
    return [{"section": name, "key": keys[i], "tokens": int(n[i]), "read": float(a[i]), "read_lift": _num(la[i]),
             "log_read_lift": _log2(la[i]), "read_reasoning": float(s[i]) if s is not None else None,
             "reasoning_lift": _num(ls[i]), "log_reasoning_lift": _log2(ls[i])} for i, name in enumerate(sections)]


def read_normals(rows_per_decision: Iterable[list[dict]], *, floor: float = READ_FLOOR, min_ref: int = 5) -> dict[str, Normal]:
    """Each section key's normal log2 read lift over reference decisions (``read_rows`` of each): ``decision``
    and, when the decisions have reasoning reading, ``reasoning``."""
    rows = [r for rs in rows_per_decision for r in rs]
    out = {"decision": Normal.fit(rows, value="log_read_lift", floor=floor, min_ref=min_ref)}
    if any(r.get("log_reasoning_lift") is not None for r in rows):
        out["reasoning"] = Normal.fit(rows, value="log_reasoning_lift", floor=floor, min_ref=min_ref)
    return out


def read_z(rows: list[dict], normals: dict[str, Normal] | None, prefix: str = "read_z") -> list[dict]:
    """Adds ``<prefix>_decision`` and ``<prefix>_reasoning`` (read z at each footing) to ``read_rows`` rows."""
    for r in rows:
        for foot, col in (("decision", "log_read_lift"), ("reasoning", "log_reasoning_lift")):
            n = (normals or {}).get(foot)
            r[f"{prefix}_{foot}"] = n.z(r["key"], r.get(col)) if n is not None else None
    return rows


def normals_to_dict(normals: dict[str, Normal]) -> dict:
    """Read normals by footing as plain JSON (stored with a probe)."""
    return {k: n.to_dict() for k, n in normals.items()}


def normals_from_dict(d: dict | None) -> dict[str, Normal] | None:
    """The inverse of ``normals_to_dict``; None for None."""
    return {k: Normal.from_dict(v) for k, v in d.items()} if d else None


def split_report(probe, h, writes, sections: list[str], tokens, *, normal: Normal | None = None,
                 writes_linear=None) -> dict:
    """The probe's score at one decision with every section's measures. ``writes`` [R, n_embd] per-section writes
    at the probe's layer (``SectionWrites``), ``writes_linear`` their DeltaNet part when split. Rows are ordered
    by z when a normal is given (sections without one go last), else by contribution."""
    w = probe.w_eff
    c = np.asarray(writes, dtype=np.float64) @ w
    total = float(np.asarray(h, dtype=np.float64) @ w)
    rem = total - float(c.sum())
    rows = split_measures(c, sections, tokens)
    if writes_linear is not None:
        cl = np.asarray(writes_linear, dtype=np.float64) @ w
        for r, v in zip(rows, cl, strict=True):
            r["contribution_linear"] = float(v)
    if normal is not None:
        for r in rows:
            r["z"] = normal.z(r["key"], r["lift"])
        rows.sort(key=lambda r: (r["z"] is None, -(r["z"] or 0.0)))
    else:
        rows.sort(key=lambda r: -r["contribution"])
    return {"probe": probe.name, "score": float(probe.score(h)[0]), "remainder": rem,
            "remainder_share": abs(rem) / ((abs(rem) + abs(float(c.sum()))) or 1.0), "section_total": float(c.sum()),
            "rows": rows}


def section_peak(c_pos, labels: list[str], token_text: list[str], *, prompt_text: str | None = None,
                 sections: Iterable[str] | None = None) -> dict[str, dict]:
    """Per section: peak, its tokens and hottest line, for any per-position values over the context. Read peak:
    head-mean attention per prompt position (``HeadReading(positions=True)``), the 8 most-read content tokens.
    Probe peak: per-position contributions (``SectionWrites(project=...)``), the 8 strongest in the direction of
    the section's sum. ``labels`` and ``token_text`` are the section and text of each position. The normalization
    is sum |v| / L over all positions given, as lift's is over sections."""
    c = np.asarray(c_pos, dtype=np.float64)
    lab = np.asarray(labels[:len(c)])
    L = len(c)
    norm = (float(np.abs(c).sum()) / L) or 1.0
    names = list(dict.fromkeys(lab)) if sections is None else list(sections)
    out = {}
    for name in names:
        idx = np.flatnonzero(lab == name)
        if len(idx) < PEAK_MIN_TOK:
            continue
        cand = np.array([i for i in idx if content_token(token_text[i])])
        if len(cand) < PEAK_K:
            continue
        sign = 1.0 if c[idx].sum() >= 0 else -1.0
        top = np.sort(cand[np.argsort(-sign * c[cand])[:PEAK_K]])
        local = {int(i): k for k, i in enumerate(idx)}
        text = hottest_line([token_text[i] for i in idx], [local[int(i)] for i in top],
                            [sign * float(c[i]) for i in top], prompt_text)
        out[name] = {"peak": float(c[top].mean() / norm), "positions": top.tolist(),
                     "tokens": [token_text[i] for i in top], "hot_line": text[:120] if text else None}
    return out


probe_peak = section_peak      # the name for per-position probe contributions (same routine)
