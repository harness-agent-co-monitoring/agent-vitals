"""Calibration of section reading against section ablation, and of ablation effects against behavior.

**Head weights.** Per-head section reading (``vitals.probes.head_reading``) is a share of attention per head and
section. A calibrated reading is a nonnegative weighted sum over heads, ŷ_s = Σ_h w_h r_{h,s}, with weights fit so
that ŷ tracks each section's share of the decision's total ablation effect, |Δm_s| / Σ|Δm|. No outcome labels
enter: the targets are the model's own ablation effects. The read map is the special case of equal weights, so
``cross_validate`` reports both on held-out decisions. Reading carries no sign; the sign comes from ablating the
sections calibrated reading ranks first.

**Behavior map.** A monotone (isotonic) map from a section's ablation effect Δm to the change in a sampled
behavior rate when the section is removed, fit on decisions where both were measured. It applies to
section-removal claims only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import nnls


def _rows(units: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for u in units:
        e = np.abs(np.asarray(u["effect"], dtype=np.float64))
        if e.sum() <= 0:
            continue
        X.append(np.asarray(u["reading"], dtype=np.float64).T)                         # [R_u, F]
        y.append(e / e.sum())
    return np.vstack(X), np.concatenate(y)


def fit_head_weights(units: list[dict], *, ridge: float = 1.0) -> np.ndarray:
    """``units``: dicts with ``reading`` [F, R_u] (features per ablated section, e.g. per-head reading) and
    ``effect`` [R_u] (Δm of each section). Nonnegative least squares, shrunk toward equal weights."""
    X, y = _rows(units)
    F = X.shape[1]
    w0 = np.full(F, np.linalg.lstsq(X.mean(1, keepdims=True), y, rcond=None)[0][0] / F)
    lam = np.sqrt(ridge * len(y) / F)
    w, _ = nnls(np.vstack([X, lam * np.eye(F)]), np.concatenate([y, lam * w0]), maxiter=50 * F)
    return w


def score(units: list[dict], w: np.ndarray | None) -> dict:
    """Within-decision agreement of a reading (calibrated with ``w``, or the head mean when None) with |Δm|:
    mean Spearman, share of decisions whose largest effect is ranked first, and the units scored."""
    rho, top = [], []
    for u in units:
        e = np.abs(np.asarray(u["effect"], dtype=np.float64))
        r = np.asarray(u["reading"], dtype=np.float64)
        est = r.mean(0) if w is None else w @ r
        if len(e) < 3 or np.ptp(e) == 0 or np.ptp(est) == 0:
            continue
        rho.append(stats.spearmanr(est, e).statistic)
        top.append(int(np.argmax(est) == np.argmax(e)))
    return {"spearman": float(np.mean(rho)) if rho else float("nan"), "top1": float(np.mean(top)) if top else float("nan"),
            "units": len(rho)}


def cross_validate(units: list[dict], groups, *, k: int = 5, ridges=(0.1, 1.0, 10.0), seed: int = 0) -> dict:
    """Held-out comparison of calibrated reading with the head mean (the read map), folds by ``groups``. The
    ridge is chosen inside each training fold."""
    groups = np.asarray(groups)
    ug = np.random.default_rng(seed).permutation(np.unique(groups))
    fold = {g: i % k for i, g in enumerate(ug)}
    calibrated, equal, chosen = [], [], []
    for f in range(min(k, len(ug))):
        tr = [u for u, g in zip(units, groups, strict=True) if fold[g] != f]
        te = [u for u, g in zip(units, groups, strict=True) if fold[g] == f]
        tg = np.array([g for g in groups if fold[g] != f])
        best, best_r = -np.inf, ridges[0]
        tug = np.unique(tg)
        for r in ridges:
            inner = []
            for i in range(min(3, len(tug))):
                hold = set(tug[i::3])
                a = [u for u, g in zip(tr, tg, strict=True) if g not in hold]
                b = [u for u, g in zip(tr, tg, strict=True) if g in hold]
                if a and b:
                    inner.append(score(b, fit_head_weights(a, ridge=r))["spearman"])
            if inner and np.nanmean(inner) > best:
                best, best_r = float(np.nanmean(inner)), r
        w = fit_head_weights(tr, ridge=best_r)
        calibrated.append(score(te, w))
        equal.append(score(te, None))
        chosen.append(best_r)

    def pool(rows):
        n = sum(r["units"] for r in rows)
        return {"spearman": sum(r["spearman"] * r["units"] for r in rows if r["units"]) / max(1, n),
                "top1": sum(r["top1"] * r["units"] for r in rows if r["units"]) / max(1, n), "units": n}

    return {"calibrated": pool(calibrated), "read_map": pool(equal), "ridges": chosen}


def save_weights(path, w: np.ndarray, features: list, meta: dict) -> None:
    """Write head weights as JSON for ``vitals explain --weights``: ``features`` lists ("span" or "at", layer,
    head) in the order of ``w``; ``meta`` (how they were fit) is stored beside them."""
    def plain(x):
        return x if isinstance(x, str) else int(x) if float(x).is_integer() else float(x)

    feats = [[plain(x) for x in f] if isinstance(f, (tuple, list)) else plain(f) for f in features]
    Path(path).write_text(json.dumps({"features": feats, "weights": [float(x) for x in w],
                                      **{k: (plain(v) if isinstance(v, (int, float, np.number)) else v) for k, v in meta.items()}}, indent=1))


def load_weights(path) -> tuple[np.ndarray, list, dict]:
    """(weights, features, meta) from a ``save_weights`` file."""
    d = json.loads(Path(path).read_text())
    return np.asarray(d.pop("weights")), d.pop("features"), d


def fit_behavior_map(delta_m, delta_p) -> dict:
    """Isotonic (pool-adjacent-violators) fit of Δp on Δm, increasing; returns knots for ``apply_behavior_map``."""
    x, y = np.asarray(delta_m, dtype=np.float64), np.asarray(delta_p, dtype=np.float64)
    o = np.argsort(x)
    x, y = x[o], y[o]
    blocks = [[y[i], 1.0, x[i], x[i]] for i in range(len(y))]                          # mean, weight, x_lo, x_hi
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m0, w0, lo, _ = blocks[i]
            m1, w1, _, hi = blocks[i + 1]
            blocks[i:i + 2] = [[(m0 * w0 + m1 * w1) / (w0 + w1), w0 + w1, lo, hi]]
            i = max(0, i - 1)
        else:
            i += 1
    return {"x": [float((b[2] + b[3]) / 2) for b in blocks], "y": [float(b[0]) for b in blocks], "n": int(len(x))}


def apply_behavior_map(bmap: dict, delta_m: float) -> float:
    """The expected change in behavior for a logit-margin change ``delta_m``, interpolated on a
    ``fit_behavior_map`` result."""
    return float(np.interp(delta_m, bmap["x"], bmap["y"]))
