"""Optional decompositions of section attention, computed within each turn.

These preserve the original four metrics. All context positions contribute to
the context mass, including undeclared positions and attention sinks. Section
regions may overlap enclosing message regions, so summing their shares is not
a valid way to recover total context mass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vitals.analysis.reading import consult_contrast, consult_ratio
from vitals.core.store import read_received_attention, read_regions, read_token_rows

DIAGNOSTIC_FIELDS = (
    "context_attention", "section_attention", "section_share_equal_step",
    "section_share_mass_weighted", "lift_equal_step", "lift_mass_weighted",
    "context_consult_ratio", "relative_consult_ratio",
    "context_consult_contrast", "relative_consult_contrast",
)


def attention_diagnostics(entropy, section, context, *, length: int, section_length: int,
                          min_tokens: int = 8, min_side: int = 16, q: float = 0.1) -> dict:
    """Separate context allocation, section allocation and length enrichment.

    ``section`` and ``context`` contain whole-attention shares at the SAME
    output steps. The relative consult factor divides the original section
    ratio by the context ratio. It describes relative allocation under entropy
    weighting, not a temporal or causal return-and-select mechanism.

    Equal-step lift is undefined if ANY step has zero context mass. Mass-weighted
    lift remains defined when total context mass is positive. Zero-entropy or
    zero-attention denominators yield None; they are never replaced by epsilon.
    """
    h, a, c = (np.asarray(x, dtype=float) for x in (entropy, section, context))
    if h.ndim != 1 or a.shape != h.shape or c.shape != h.shape:
        raise ValueError("entropy, section and context must be aligned one-dimensional arrays")
    if not 0 < section_length <= length:
        raise ValueError("section_length must be positive and no greater than context length")
    if not 0 < q < 0.5:
        raise ValueError("q must be between 0 and 0.5")
    if not all(np.isfinite(x).all() for x in (h, a, c)) or any((x < 0).any() for x in (h, a, c)):
        raise ValueError("entropy and attention shares must be finite and nonnegative")
    if (c > 1 + 1e-6).any() or (a > c + 1e-6).any():
        raise ValueError("section attention cannot exceed context attention, which cannot exceed one")
    out = dict.fromkeys(DIAGNOSTIC_FIELDS)
    out.update(n_steps=len(h), n_context_steps=int((c > 0).sum()))
    if not len(h):
        return out
    out.update(context_attention=float(c.mean()), section_attention=float(a.mean()))
    if c.sum() > 0:
        share = float(a.sum() / c.sum())
        out.update(section_share_mass_weighted=share, lift_mass_weighted=length / section_length * share)
    if (c > 0).all():
        share = float(np.mean(a / c))
        out.update(section_share_equal_step=share, lift_equal_step=length / section_length * share)
    cr = consult_ratio(h, c, min_tokens=min_tokens)
    sr = consult_ratio(h, a, min_tokens=min_tokens)
    cc = consult_contrast(h, c, q=q, min_side=min_side)
    sc = consult_contrast(h, a, q=q, min_side=min_side)
    out.update(
        context_consult_ratio=cr,
        relative_consult_ratio=sr / cr if sr is not None and cr is not None and cr > 0 else None,
        context_consult_contrast=cc,
        relative_consult_contrast=sc / cc if sc is not None and cc is not None and cc > 0 else None,
    )
    return out


def episode_attention_diagnostics(ep_dir: str | Path, tokens=None) -> dict[tuple[int, str], dict]:
    """Read exact section shares; reject missing/legacy rows instead of imputing.

    Complete ``attn_share`` rows end in the generated-token bucket. Its
    complement is total context mass, regardless of declaration coverage.
    Missing steps invalidate the turn, preventing comparisons on different
    subsets of steps. Sparse top-K weights are not an exact fallback.
    """
    ep = Path(ep_dir)
    if tokens is None:
        tokens = read_token_rows(ep)
        if tokens is None:
            return {}
    regions = read_regions(ep)
    # context length from the stored attention vector, which covers undeclared prompt positions too
    lengths = {t: len(v) for t, v in read_received_attention(ep).items()}
    out = {}
    for turn, g in tokens[tokens.region_id == "gen"].groupby("turn"):
        turn = int(turn)
        regs, length = regions.get(turn, []), lengths.get(turn, 0)
        if not regs or not length:
            continue
        valid = "attn_share" in g and all(
            isinstance(row, (list, tuple, np.ndarray)) and len(row) == len(regs) + 1 for row in g.attn_share)
        status = "complete" if valid else "missing_complete_attention"
        if "step" not in g or sorted(g.step.tolist()) != list(range(len(g))):
            valid, status = False, "missing_output_steps"
        shares = np.asarray(g.attn_share.tolist(), dtype=float) if valid else None
        for i, r in enumerate(regs):
            rid, start, end = str(r.get("id", "")), r.get("token_start"), r.get("token_end")
            if not rid.startswith("span:") or start is None or end is None:
                continue
            if not 0 <= start < end <= length:
                continue
            values = dict.fromkeys(DIAGNOSTIC_FIELDS)
            values.update(n_steps=len(g), n_context_steps=None)
            section_status = status
            if valid:
                try:
                    values = attention_diagnostics(g.ent.to_numpy(), shares[:, i],
                                                   1.0 - shares[:, -1], length=length,
                                                   section_length=int(end - start))
                except ValueError:
                    section_status = "invalid_attention"
            out[turn, rid[5:]] = {"status": section_status, **values}
    return out


def format_diagnostics(rows: list[dict]) -> str:
    """Compact companion to the original four-metric report."""
    head = ["episode", "turn", "section", "lift/step", "lift/mass", "context CR",
            "relative CR", "context CC", "relative CC", "peak/step*", "gap*", "focus JS/bits", "status"]
    keys = ["lift_equal_step", "lift_mass_weighted", "context_consult_ratio", "relative_consult_ratio",
            "context_consult_contrast", "relative_consult_contrast",
            "peak_step_mass_weighted", "peak_gap_mass_weighted", "focus_js_bits"]
    body = []
    for row in rows:
        d = row.get("diagnostics", {})
        body.append([row.get("episode", ""), str(row["turn"]), row["section"]]
                    + ["—" if d.get(k) is None else f"{d[k]:.3f}" for k in keys]
                    + [d.get("status", "unavailable")])
    if not body:
        return "(no section diagnostics)"
    widths = [max(len(x) for x in col) for col in zip(head, *body, strict=True)]
    lines = ["  ".join(x.ljust(w) for x, w in zip(row, widths, strict=True)) for row in [head, *body]]
    return "\n".join(lines) + "\n* Per-step peak and gap use context-attention mass weights. Missing capture is shown as —."
