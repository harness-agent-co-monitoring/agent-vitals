"""Residual-stream probes that detect, localized by section and confirmed by ablation.

The section part is reported in the read map's measures (``vitals.monitor.measures``): contribution, probe lift,
z against the section's normal, probe peak and hottest line. ``train_probe`` fits a probe and stores its
threshold and section normal with it; ``explain_flag`` reports one decision in those terms.

A ``LinearProbe`` scores the hidden state h_L at a decision: s = w . (h_L - mu) / sd + b. When it fires:

1. **Localize** (``split_report``): every attention layer up to L adds the writes of what its heads read, and those
   split exactly by section (``vitals.probes.section_writes``), so s - (constant) = sum_s c_s + remainder with
   c_s = w_eff . A_{s,L}. Each section's contribution is a read map weighted by the probe direction: how much what
   the model read from that section pushes toward the flag. The remainder (embedding, MLPs, recurrent layers) is
   reported, not hidden.
2. **Confirm** (``ablation_scores``): replace a section with filler, re-decode from it with prefix reuse, and re-score
   the probe at the same decision. A section drives the flag when the score drops without it.
3. **Explain** (``explain_flag``): the score, the section measures ranked by z (by contribution when the probe
   has no normal), the plain read map beside them, the confirmed drops, one-sentence claims.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from vitals.core.io import atomic_write_json
from vitals.monitor.ledger import ledger_decisions  # noqa: F401  (re-exported)
from vitals.monitor.measures import (
    READ_FLOOR,
    REASONING,
    TEMPLATE,
    Normal,
    normals_from_dict,
    normals_to_dict,
    read_normals,
    read_rows,
    read_z,
    section_peak,
    split_report,
)
from vitals.probes.ablation import filler_tokens, spans_from_labels

PROBE_FORMAT = 1
EXCLUDE = ("output", "region:", "cue")      # sections never ablated: the turn's own output, template text, cues


@dataclass
class LinearProbe:
    """A linear probe on the residual at ``layer``: score(h) = w . (h - mu) / sd + b, flagged at or above
    ``threshold``. ``meta`` carries what ``train_probe`` stores with it (alert threshold, section normal, read
    normal, out-of-fold AUROC, counts). Saved as JSON (``save`` / ``load``)."""

    name: str
    layer: int
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    b: float
    threshold: float | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, name: str, layer: int, H: np.ndarray, y: np.ndarray, C: float = 1e-3, **meta) -> LinearProbe:
        """Standardized L2 logistic regression on residuals ``H`` [N, n_embd] with labels ``y`` (needs torch)."""
        from vitals.monitor.linear import fit_lr

        mu, sd, w, b = fit_lr(H, y, C)
        return cls(name, int(layer), mu, sd, w, float(b), None, dict(meta))

    def score(self, h) -> np.ndarray:
        """Scores of one residual [n_embd] or several [N, n_embd]."""
        h = np.atleast_2d(np.asarray(h, dtype=np.float64))
        return (h - self.mu) / self.sd @ self.w + self.b

    @property
    def w_eff(self) -> np.ndarray:
        """The direction in residual space: score = w_eff . h + constant."""
        return self.w / self.sd

    def save(self, path: str | Path) -> None:
        d = asdict(self)
        for k in ("mu", "sd", "w"):
            d[k] = [float(x) for x in d[k]]
        atomic_write_json(path, {"format": PROBE_FORMAT, **d})

    @classmethod
    def load(cls, path: str | Path) -> LinearProbe:
        d = json.loads(Path(path).read_text())
        if int(d.pop("format", 1)) > PROBE_FORMAT:
            raise ValueError(f"{path} was written by a newer agent-vitals (probe format {d.get('format')})")
        known = {f.name for f in fields(cls)}
        for k in ("mu", "sd", "w"):
            d[k] = np.asarray(d[k], dtype=np.float64)
        return cls(**{k: v for k, v in d.items() if k in known})


def decompose(probe: LinearProbe, h: np.ndarray, writes: np.ndarray, sections: list[str]) -> dict:
    """Split the probe's score at one decision by section (the raw form of ``split_report``). ``h``: the residual
    at the probe's layer; ``writes``: [R, n_embd] attention writes per section summed over the attention layers up
    to that layer."""
    w = probe.w_eff
    c = np.asarray(writes, dtype=np.float64) @ w
    total = float(np.asarray(h, dtype=np.float64) @ w)
    rem = total - float(c.sum())
    order = np.argsort(-c)
    return {"score": float(probe.score(h)[0]), "sections": [(sections[i], float(c[i])) for i in order],
            "remainder": rem, "section_total": float(c.sum())}


def auroc(y, s) -> float:
    """Area under the ROC curve of scores ``s`` for binary labels ``y`` (ties count half); nan with one class."""
    y, s = np.asarray(y, dtype=bool), np.asarray(s, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    vals, inv = np.unique(s, return_inverse=True)            # average ranks over ties
    ranks = (np.bincount(inv, ranks) / np.bincount(inv))[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def train_probe(name: str, layer: int, H, y, groups, *, splits: list[tuple] | None = None, C: float = 1e-3,
                folds: int = 5, budget: float = 0.10, normal_on: str = "negatives", reads: list[list[dict]] | None = None,
                read_floor: float = READ_FLOOR, **meta) -> tuple[LinearProbe, np.ndarray]:
    """Fit a probe on residuals ``H`` [N, n_embd] at ``layer`` with labels ``y``; out-of-fold scores with folds
    by ``groups`` (task or episode). The returned probe is fit on everything and carries:

    - ``threshold``: the out-of-fold score negatives exceed at rate ``budget`` (the false-alarm budget), and
      ``meta["alert_threshold"]`` at a fifth of it;
    - ``meta["normal"]``: each section key's normal probe lift, from ``splits`` (per decision: writes [R, n_embd]
      at ``layer``, section names, section token counts), every decision split by the probe that did not see it.
      By default only decisions without the behavior form the normal, so z answers "what is different here when
      the behavior happens"; ``normal_on="all"`` uses every decision;
    - ``meta["read_normal"]``: from ``reads`` (per decision, ``read_rows``), each section key's normal log2 read lift
      at the decision and over the reasoning over the same decisions, so read z beside the probe rows answers the
      same question as probe z;
    - ``meta["oof_auroc"]``, counts and the budget.

    Returns (probe, out-of-fold scores)."""
    from vitals.monitor.linear import group_kfold

    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y))
    rows = []
    for tr, te in group_kfold(np.asarray(groups), folds):
        p = LinearProbe.fit(name, layer, H[tr], y[tr], C=C)
        oof[te] = p.score(H[te])
        if splits is not None:
            for i in te:
                if normal_on == "negatives" and y[i] != 0 or splits[i] is None:
                    continue
                writes, sections, tokens = splits[i]
                rows += split_report(p, H[i], writes, sections, tokens)["rows"]
    probe = LinearProbe.fit(name, layer, H, y, C=C, **meta)
    neg = oof[y == 0]
    probe.threshold = float(np.quantile(neg, 1 - budget)) if len(neg) else None
    probe.meta["alert_threshold"] = float(np.quantile(neg, 1 - budget / 5)) if len(neg) else None
    probe.meta.update({"oof_auroc": auroc(y, oof), "n": int(len(y)), "positives": int(y.sum()), "budget": budget,
                       "folds": folds, "C": C})
    if rows:
        probe.meta["normal"] = Normal.fit(rows).to_dict()
        probe.meta["normal_on"] = normal_on
    if reads is not None:
        ref = [r for i, r in enumerate(reads) if r is not None and (normal_on != "negatives" or y[i] == 0)]
        if ref:
            probe.meta["read_normal"] = normals_to_dict(read_normals(ref, floor=read_floor))
    return probe, oof


def probe_normal(probe: LinearProbe) -> Normal | None:
    """The section normal stored with a probe (probe lift over decisions without its behavior), or None."""
    d = probe.meta.get("normal")
    return Normal.from_dict(d) if d else None


def probe_read_normals(probe: LinearProbe) -> dict[str, Normal] | None:
    """The read normals stored with a probe (decisions without its behavior), or None."""
    return normals_from_dict(probe.meta.get("read_normal"))


def ledger_reads(decisions: list[dict]) -> list[list[dict] | None]:
    """``read_rows`` of each ledger decision (``train_probe(reads=...)``), None for one without reading."""
    return [read_rows(list(d["sections"]), d["tokens"], d["at"], d.get("span")) if "at" in d else None for d in decisions]


def ledger_inputs(decisions: list[dict], layer: int) -> tuple[np.ndarray, list[tuple]]:
    """Residuals and per-section writes at ``layer`` from ledger decisions (``ledger.load_decision``), as
    ``train_probe`` takes them. Decisions captured without writes get None splits."""
    H, splits = [], []
    for d in decisions:
        k = list(d["residual_layers"]).index(layer)
        H.append(np.asarray(d["residual_at"][k], dtype=np.float64))
        splits.append((np.asarray(d["writes"][k], dtype=np.float64), d["sections"], d["tokens"]) if "writes" in d else None)
    return np.stack(H), splits


def score_ledger(probe: LinearProbe, decisions: list[dict], *, top: int = 3) -> list[dict]:
    """The probe's score at every ledger decision, one record each, with the remainder share and the top sections
    (by z when the probe has a normal, else by contribution) where the decision has per-section writes."""
    normal, rnorm = probe_normal(probe), probe_read_normals(probe)
    H, splits = ledger_inputs(decisions, probe.layer)
    out = []
    for d, h, sp in zip(decisions, H, splits, strict=True):
        rec = {"episode": d["episode"], "turn": d["turn"], "score": float(probe.score(h)[0])}
        rec["flagged"] = probe.threshold is not None and rec["score"] >= probe.threshold
        if sp is not None:
            rep = split_report(probe, h, *sp, normal=normal)
            reads = ledger_reads([d])[0]
            rz = {r["section"]: r for r in read_z(reads, rnorm, "read_z")} if reads else {}
            rows = [{**r, **{k: rz.get(r["section"], {}).get(k) for k in ("read_z_decision", "read_z_reasoning")}}
                    for r in rep["rows"] if r["key"] not in (TEMPLATE, REASONING)]
            rec.update(remainder_share=rep["remainder_share"],
                       top=[{k: r.get(k) for k in ("section", "key", "contribution", "lift", "z", "read_z_decision",
                                                    "read_z_reasoning")} for r in rows[:top]])
        out.append(rec)
    return out


def ablation_scores(backend, probe: LinearProbe, tokens: list[int], position: int,
                    ablations: dict[str, list[tuple[int, int]]], *, seed: int = 0,
                    replacements: dict[str, list[list[int]]] | None = None) -> dict:
    """The probe's score at ``position`` with the original input and with each named set of spans replaced by
    filler (or by ``replacements``, which must keep each span's length). The base sequence is decoded once in
    n_batch chunks; each ablation forks it at the chunk boundary before its first span and re-decodes only from
    there, in the same chunk layout, so a no-op replacement reproduces the original score exactly. Needs a backend
    with n_seq_max >= 2."""
    eng = backend.engine
    if int(eng._ctx_opts.get("n_seq_max", 1)) < 2:
        raise ValueError("ablation_scores forks the decoded prefix: load the backend with n_seq_max >= 2")
    nb = int(eng._ctx_opts["n_batch"])
    node = backend.node_map.residual_node(probe.layer)
    rng = random.Random(seed)
    toks = list(tokens)

    def finish(seq: int, seqtoks: list[int], start: int) -> np.ndarray:
        eng.set_capture(set())
        for off in range(start, position, nb):
            eng.decode(seqtoks[off:min(off + nb, position)], pos0=off, seq_id=seq)
        eng.take_device_parts()
        eng.set_capture({node})
        eng.decode([seqtoks[position]], pos0=position, seq_id=seq)
        parts = eng.take_device_parts()[node]
        h = np.concatenate([p.reshape(-1, p.shape[-1]).float().cpu().numpy() for p in parts])[-1]
        eng.set_capture(set())
        return h

    eng.reset_context(capture_names={node}, capture_modes={node: "device"})
    eng.set_capture(set())
    plan = []
    for name, spans in ablations.items():
        spans = sorted((int(s), int(e)) for s, e in spans)
        if not spans or spans[-1][1] > position:
            raise ValueError(f"ablation {name!r}: spans must end at or before the decision position")
        reps = (replacements or {}).get(name) or [filler_tokens(eng, e - s, rng) for s, e in spans]
        if any(len(r) != e - s for r, (s, e) in zip(reps, spans, strict=True)):
            raise ValueError(f"ablation {name!r}: replacements must keep each span's length")
        plan.append((name, spans, reps))
    out, cursor = {}, 0
    for name, spans, reps in sorted(plan, key=lambda x: x[1][0][0]):
        fork = (spans[0][0] // nb) * nb
        while cursor < fork:
            eng.decode(toks[cursor:cursor + nb], pos0=cursor, seq_id=0)
            cursor += nb
        eng.take_device_parts()
        alt = list(toks)
        for (s, e), r in zip(spans, reps, strict=True):
            alt[s:e] = list(r)
        eng.branch_copy(1, 0)
        out[name] = float(probe.score(finish(1, alt, fork))[0])
        eng.branch_drop(1)
    eng.branch_copy(1, 0)
    base = float(probe.score(finish(1, toks, cursor))[0])
    eng.branch_drop(1)
    return {"base": base, "without": out}


def explain_flag(backend, gguf_path: str, probe: LinearProbe, tokens: list[int], labels: list[str], position: int, *,
                 confirm_top: int = 2, exclude: tuple[str, ...] = EXCLUDE, linear: bool = False,
                 prompt_text: str | None = None) -> dict:
    """Score, localize and confirm one decision (see the module docstring). Sections are ranked by z against the
    probe's normal when it has one (``train_probe``), else by contribution; the top ``confirm_top`` sections that
    push toward the flag are confirmed by ablation and get a probe peak and hottest line."""
    from vitals.probes.section_writes import section_writes

    cap = section_writes(backend, gguf_path, layers=[probe.layer], linear=linear).capture(
        tokens, labels, [position], project=probe.w_eff)
    q = cap["positions"][position]
    names = cap["sections"]
    lab = labels[:position + 1]
    n_tok = [sum(1 for x in lab if x == n) for n in names]
    rep = split_report(probe, q["h"][0], q["writes"][0], names, n_tok, normal=probe_normal(probe),
                       writes_linear=q["writes_linear"][0] if "writes_linear" in q else None)
    read = dict(zip(names, (float(v) for v in q["read"][0]), strict=True))
    for r in rep["rows"]:
        r["read"] = read[r["section"]]
    spans = spans_from_labels(labels[:position])
    ranked = [r for r in rep["rows"] if r["key"] not in (TEMPLATE, REASONING) and r["section"] in spans
              and not r["section"].startswith(exclude)]
    cands = [r["section"] for r in ranked if r["contribution"] > 0][:confirm_top]
    conf = ablation_scores(backend, probe, tokens, position, {n: spans[n] for n in cands}) if cands \
        else {"base": rep["score"], "without": {}}
    text = [backend.engine.detokenize([t]) for t in tokens[:position + 1]]
    peaks = section_peak(q["contrib_pos"][0], lab, text, prompt_text=prompt_text, sections=cands)
    by = {r["section"]: r for r in rep["rows"]}
    claims = []
    for n in cands:
        r = by[n]
        z = f", z {r['z']:+.1f} against its normal" if r.get("z") is not None else ""
        line = f"; strongest line: \"{peaks[n]['hot_line']}\"" if n in peaks and peaks[n]["hot_line"] else ""
        claims.append(f"Section '{n}' pushes toward '{probe.name}' at probe lift {r['lift']:+.1f}{z}; the plain read "
                      f"map gives it {r['read']:.1%} of the attention{line}; without it the score goes from "
                      f"{conf['base']:+.2f} to {conf['without'][n]:+.2f}.")
    return {"probe": probe.name, "score": conf["base"], "threshold": probe.threshold,
            "flagged": probe.threshold is not None and conf["base"] >= probe.threshold,
            "ranked_by": "z" if probe_normal(probe) else "contribution", "rows": rep["rows"][:12],
            "contributions": [(r["section"], r["contribution"]) for r in ranked[:10]],
            "remainder": rep["remainder"], "remainder_share": rep["remainder_share"], "section_total": rep["section_total"],
            "peaks": peaks, "confirmed": conf["without"], "claims": claims}
