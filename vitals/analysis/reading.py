"""Per-turn harness reading and its relation to uncertainty, per stored episode.

Everything here is computed from a stored episode alone: the token parquet
(need = mean action entropy per turn, fork density), attn_received.jsonl
(read weight per prompt position, per turn), and regions.jsonl (which
positions come from the harness; span:* sub-spans for per-section stats).

Definitions:
  read weight  W_i(j)   = mean received-attention mass per token of region j
                          during the decode of turn i
  read lift    Lam_i(j) = W_i(j) * L_i / C_i  (weight relative to uniform
                          reading of the turn's L_i context positions, where
                          C_i = the share of the decode's attention that lands
                          on the context at all; the rest is on the model's
                          own generated tokens, so a section's lift does not
                          fall just because the model reasoned longer)
  need         u_i      = mean per-token entropy of turn i's generated tokens
  coupling     chi      = corr_i(u_i, W_i(H))  over the episode's turns
  adrift turns          = { i : u_i > med(u)  and  W_i(H) < med(W(H)) }

``chi`` and the adrift set are reported in both the raw-share form (W) and
the lift form (Lam); lift removes the mechanical 1/L_i decay as the context
grows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vitals.core.store import read_received_attention, read_regions, read_token_rows, span_index

HARNESS_IDS = ("sys_0", "user_0", "schema_0")

# The first few context positions are attention sinks that hold a few percent
# of all received mass. They sit inside sys_0, so harness read stats exclude
# them; otherwise W_H and lift_H track the sink, not the reading.
SINK_SKIP = 4


def _span_stats(vec: list[float], s: int, e: int, ctx: float | None = None) -> tuple[float, float]:
    """(read weight, read lift) of region [s, e) under this turn's vector.
    ``ctx`` is the context's share of the decode's attention (sum of the
    vector); lift is the weight against uniform reading of the context."""
    e = min(e, len(vec))
    if e <= s:
        return 0.0, 0.0
    if ctx is None:
        ctx = float(sum(vec)) or 1.0
    w = float(sum(vec[s:e])) / (e - s)
    return w, w * len(vec) / ctx


def episode_turns(ep_dir: str | Path, tokens=None) -> list[dict]:
    """Per-turn table: need, fork density, harness read weight/lift, context
    length, and per-span-section weight/lift. ``tokens`` may pass a preloaded
    pandas frame of the episode's parquet rows to avoid a re-read."""
    ep = Path(ep_dir)
    if tokens is None:
        tokens = read_token_rows(ep)
        if tokens is None:
            return []
    gen = tokens[tokens.region_id == "gen"]
    regions = read_regions(ep)
    attn = read_received_attention(ep)
    rows: list[dict] = []
    for t, vec in sorted(attn.items()):
        g = gen[gen.turn == t]
        if len(g) == 0:
            continue
        ctx = float(sum(vec)) or 1.0  # context share of the decode's attention
        row: dict = {"turn": int(t), "n_gen": int(len(g)),
                     "u": float(g.ent.mean()), "fork": float(g.fork.mean()),
                     "L": len(vec), "ctx_share": ctx, "W_H": 0.0, "lift_H": 0.0,
                     "lift_spans": None, "sections": {}}
        h_mass = h_tokens = 0
        for r in regions.get(t, []):
            rid = str(r.get("id", ""))
            s, e = r.get("token_start"), r.get("token_end")
            if s is None or e is None:
                continue
            if rid in HARNESS_IDS:
                s2, e2 = max(int(s), SINK_SKIP), min(int(e), len(vec))
                if e2 > s2:
                    h_mass += float(sum(vec[s2:e2]))
                    h_tokens += e2 - s2
            elif rid.startswith("span:"):
                w, lam = _span_stats(vec, int(s), int(e), ctx)
                row["sections"][rid.removeprefix("span:")] = {
                    "W": round(w, 8), "lift": round(lam, 4)}
        if h_tokens:
            row["W_H"] = h_mass / h_tokens
            row["lift_H"] = row["W_H"] * len(vec) / ctx
        if row["sections"]:
            # mean lift over the named sections; sink-free because spans sit
            # past the prompt front
            row["lift_spans"] = round(float(np.mean(
                [d["lift"] for d in row["sections"].values()])), 4)
        rows.append(row)
    return rows


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def episode_coupling(ep_dir: str | Path, tokens=None) -> dict:
    """chi (raw + lift), adrift turns/fraction, and the per-turn table."""
    turns = episode_turns(ep_dir, tokens=tokens)
    out: dict = {"n_turns": len(turns), "turns": turns, "chi": None,
                 "chi_lift": None, "chi_spans": None, "adrift_turns": [],
                 "adrift_fraction": None, "adrift_turns_spans": [],
                 "regime": None}
    if len(turns) < 4:
        return out
    u = np.array([r["u"] for r in turns])
    w = np.array([r["W_H"] for r in turns])
    lam = np.array([r["lift_H"] for r in turns])
    out["chi"] = _corr(u, w)
    out["chi_lift"] = _corr(u, lam)
    med_u, med_w = np.median(u), np.median(w)
    out["adrift_turns"] = [r["turn"] for r, ui, wi in zip(turns, u, w, strict=True)
                           if ui > med_u and wi < med_w]
    out["adrift_fraction"] = round(len(out["adrift_turns"]) / len(turns), 4)
    sp = [(r["turn"], ui, r["lift_spans"]) for r, ui in zip(turns, u, strict=True)
          if r["lift_spans"] is not None]
    if len(sp) >= 4:
        us = np.array([x[1] for x in sp])
        ls = np.array([x[2] for x in sp])
        out["chi_spans"] = _corr(us, ls)
        mu, ml = np.median(us), np.median(ls)
        out["adrift_turns_spans"] = [t for t, ui, li in sp if ui > mu and li < ml]
    ref = next((out[k] for k in ("chi_spans", "chi_lift", "chi")
                if out[k] is not None), None)
    if ref is not None:
        out["regime"] = "consulting" if ref > 0 else "abandonment"
    return out


def needed_but_unread(ep_dir: str | Path, tokens=None) -> list[dict]:
    """Rank sections by 'needed but unread', ascending: the sections read
    LEAST at the turns where reading matters. Primary score is each section's
    mean lift over the episode's adrift turns; when no turn is adrift, the
    need-weighted mean lift is the fallback. Top entries are where a text edit
    is least likely to be read."""
    turns = episode_turns(ep_dir, tokens=tokens)
    if not turns:
        return []
    u = np.array([r["u"] for r in turns])
    wts = u - u.min()
    wts = wts / wts.sum() if wts.sum() > 0 else np.full(len(u), 1 / len(u))
    sp = [(i, r["lift_spans"]) for i, r in enumerate(turns)
          if r["lift_spans"] is not None]
    adrift_idx: list[int] = []
    if len(sp) >= 4:
        us = np.array([u[i] for i, _ in sp])
        ls = np.array([lam for _, lam in sp])
        mu, ml = np.median(us), np.median(ls)
        adrift_idx = [i for (i, lam), ui in zip(sp, us, strict=True) if ui > mu and lam < ml]
    names = sorted({n for r in turns for n in r["sections"]})
    out = []
    for name in names:
        lifts = np.array([r["sections"].get(name, {}).get("lift", 0.0) for r in turns])
        adrift_lift = (round(float(lifts[adrift_idx].mean()), 4)
                       if adrift_idx else None)
        nwl = round(float((wts * lifts).sum()), 4)
        chi_j = _corr(u, lifts)  # is THIS section consulted more under need?
        out.append({"section": name, "adrift_lift": adrift_lift,
                    "need_weighted_lift": nwl,
                    "mean_lift": round(float(lifts.mean()), 4),
                    "chi": None if chi_j is None else round(chi_j, 3)})
    out.sort(key=lambda d: d["adrift_lift"] if d["adrift_lift"] is not None
             else d["need_weighted_lift"])
    return out


def _token_section_arrays(ep: Path, tokens):
    """Yield (turn, section, lifts, shares, ents, forks) for every (turn,
    span-section) with generation-side attn_share data. ``shares`` are the
    section's share of each generated token's whole attention (the consult
    measures use these); ``lifts`` renormalize each step over the context
    (1 - the trailing generated bucket) before dividing by the section's
    length share. Averaging these lifts with context-mass weights matches
    episode_turns; an equal-step mean is a different estimator."""
    gen = tokens[tokens.region_id == "gen"]
    gen = gen[gen.attn_share.map(lambda s: s is not None and len(s) > 0)]
    if len(gen) == 0:
        return
    regions = read_regions(ep)
    attn = read_received_attention(ep)
    span_idx = {int(t): span_index(regs) for t, regs in regions.items()}
    n_reg = {int(t): len(regs) for t, regs in regions.items()}
    for t, g in gen.groupby("turn"):
        m = span_idx.get(int(t))
        L = len(attn.get(int(t), []))
        if not m or not L:
            continue
        ents = g.ent.to_numpy()
        forks = g.fork.to_numpy().astype(bool)
        # attn_share = one share per region + a trailing generated-tokens bucket;
        # the context share of a step is 1 - that bucket
        nr = n_reg.get(int(t), 0)
        ctx = np.array([max(1.0 - float(s[-1]), 1e-9) if len(s) == nr + 1 else 1.0 for s in g.attn_share])
        for name, (i, size) in m.items():
            shares = np.array([float(s[i]) if i < len(s) else 0.0
                               for s in g.attn_share])
            yield int(t), name, shares / ctx / (size / L), shares, ents, forks


def consult_ratio(ents, shares, min_tokens: int = 8):
    """The consult ratio: a section's attention share averaged with each
    generated token weighted by its entropy, over its plain average share.

        CR(s) = ( sum_t H_t a_t(s) / sum_t H_t ) / ( mean_t a_t(s) )

    1 = zero covariance with entropy, which does not exclude a nonlinear
    relationship; > 1 = positive covariance; < 1 = negative covariance.
    Algebraically 1 + cov(H, a)/(mean H * mean a): the same covariance as the
    entropy-share correlation, scaled by means instead of standard deviations,
    so its sign always agrees with the correlation's and its magnitude reads
    as a multiplicative effect. None when the turn is too short, entropy is
    identically zero, or the section drew no attention (log2 of the ratio is
    the symmetric form for averaging)."""
    ents = np.asarray(ents, dtype=float)
    shares = np.asarray(shares, dtype=float)
    if len(ents) < min_tokens or ents.sum() <= 0 or shares.mean() <= 0:
        return None
    return float((ents * shares).sum() / ents.sum() / shares.mean())


def consult_contrast(ents, shares, q: float = 0.1, min_side: int = 16):
    """The consult contrast (the consult ratio's tail form): the section's
    mean attention share over the q most-uncertain generated tokens divided by
    its mean share over the q most-confident ones. None when either side has
    fewer than min_side tokens or the confident side drew no attention. The
    ratio says whether the model reaches for a section when unsure; the
    contrast says how much, at the extremes the covariance compresses."""
    ents = np.asarray(ents, dtype=float)
    shares = np.asarray(shares, dtype=float)
    n = len(ents)
    if n < 2 * min_side:
        return None
    hi, lo = np.quantile(ents, 1 - q), np.quantile(ents, q)
    top, bot = shares[ents >= hi], shares[ents <= lo]
    if len(top) < min_side or len(bot) < min_side or bot.mean() <= 0:
        return None
    return float(top.mean() / bot.mean())


def token_uptake_turns(ep_dir: str | Path, tokens=None) -> list[dict]:
    """PER-TURN token-level combination of internal state with reading
    (generation-side attn_share joined with each token's entropy and fork
    flag). One row per (turn, section):

      uptake  -- mean lift at that turn's fork tokens / mean lift at its
                 other tokens (needs >= 3 fork tokens, else None)
      chi_tok -- within-turn correlation of token entropy with the section's
                 share (needs >= 8 tokens and variance, else None). Context
                 length is ~constant inside a turn, so no decay correction.
      consult_ratio -- entropy-weighted mean share over plain mean share
                 (see consult_ratio()), same sign as chi_tok
      consult_contrast -- see consult_contrast()
    """
    ep = Path(ep_dir)
    if tokens is None:
        tokens = read_token_rows(ep)
        if tokens is None:
            return []
    out = []
    for t, name, lifts, shares, ents, forks in _token_section_arrays(ep, tokens):
        fl = float(lifts[forks].mean()) if forks.sum() >= 3 else None
        nl = float(lifts[~forks].mean()) if (~forks).any() else None
        chi_tok = None
        if len(ents) >= 8 and np.std(shares) > 0 and np.std(ents) > 0:
            chi_tok = round(float(np.corrcoef(ents, shares)[0, 1]), 3)
        cr = consult_ratio(ents, shares)
        cc = consult_contrast(ents, shares)
        out.append({"turn": t, "section": name,
                    "fork_lift": None if fl is None else round(fl, 4),
                    "nonfork_lift": None if nl is None else round(nl, 4),
                    "uptake": (None if fl is None or not nl else round(fl / nl, 3)),
                    "chi_tok": chi_tok,
                    "consult_ratio": None if cr is None else round(cr, 4),
                    "consult_contrast": None if cc is None else round(cc, 4),
                    "n": int(len(ents)), "n_fork": int(forks.sum())})
    return out


def token_uptake(ep_dir: str | Path, tokens=None) -> list[dict]:
    """Episode-level aggregate of token_uptake_turns: per section, fork/
    nonfork lift pooled over all turns (uptake = ratio), chi_tok as the mean
    of the within-turn correlations, consult_ratio as the geometric mean of
    the within-turn ratios."""
    ep = Path(ep_dir)
    if tokens is None:
        tokens = read_token_rows(ep)
        if tokens is None:
            return []
    acc: dict[str, dict[str, list]] = {}
    for _, name, lifts, shares, ents, forks in _token_section_arrays(ep, tokens):
        d = acc.setdefault(name, {"fork": [], "nonfork": [], "chi": [], "lcr": []})
        d["fork"].extend(lifts[forks].tolist())
        d["nonfork"].extend(lifts[~forks].tolist())
        if len(ents) >= 8 and np.std(shares) > 0 and np.std(ents) > 0:
            d["chi"].append(float(np.corrcoef(ents, shares)[0, 1]))
        cr = consult_ratio(ents, shares)
        if cr is not None and cr > 0:
            d["lcr"].append(float(np.log2(cr)))
    out = []
    for name, d in acc.items():
        fl = float(np.mean(d["fork"])) if d["fork"] else None
        nl = float(np.mean(d["nonfork"])) if d["nonfork"] else None
        out.append({
            "section": name,
            "fork_lift": None if fl is None else round(fl, 4),
            "nonfork_lift": None if nl is None else round(nl, 4),
            "uptake": (None if fl is None or not nl else round(fl / nl, 3)),
            "chi_tok": (round(float(np.mean(d["chi"])), 3) if d["chi"] else None),
            "n_turns_chi": len(d["chi"]),
            # geometric mean over turns: the symmetric average of a ratio
            "consult_ratio": (round(float(2 ** np.mean(d["lcr"])), 4) if d["lcr"] else None),
        })
    out.sort(key=lambda r: -(r["uptake"] or 0))
    return out

