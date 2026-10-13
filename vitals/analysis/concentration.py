"""Exact peaks and attention-focus JS accumulated before sparse truncation.

Memory is O(context length + number of sections), independent of output
length. Both sides of each peak comparison share the same step weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vitals.analysis.readmap import PEAK_K, PEAK_MIN_TOK, content_token
from vitals.core.io import atomic_write_parquet

PEAK_FIELDS = tuple(f"peak_{kind}_{weight}" for weight in ("mass_weighted", "equal_step")
                    for kind in ("pooled", "step", "gap"))
FOCUS_FIELDS = ("focus_js_bits", "focus_pooled_entropy_bits", "focus_step_entropy_bits",
                "focus_weight_entropy_bits", "focus_attention_mass", "focus_n_attended_steps")
CONCENTRATION_FIELDS = PEAK_FIELDS + FOCUS_FIELDS
CONCENTRATION_VERSION = 2


def _xlog2x(values):
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values)
    positive = values > 0
    out[positive] = values[positive] * np.log2(values[positive])
    return out


class SectionConcentration:
    """Accumulate full, head/layer-pooled prompt attention one step at a time."""

    def __init__(self, token_strs: list[str], regions: list):
        self.length = len(token_strs)
        self.sections = []
        mask = np.array([content_token(t) for t in token_strs], dtype=bool)
        for r in regions:
            r = r if isinstance(r, dict) else r.model_dump()
            rid, start, end = str(r.get("id", "")), r.get("token_start"), r.get("token_end")
            if not rid.startswith("span:") or start is None or end is None:
                continue
            start, end = int(start), int(end)
            if not 0 <= start < end <= self.length:
                raise ValueError(f"invalid section range: {rid}")
            positions = np.flatnonzero(mask[start:end]) + start
            eligible = end - start >= PEAK_MIN_TOK and len(positions) >= PEAK_K
            self.sections.append((rid[5:], start, end, positions if eligible else None))
        self.n_steps = self.n_context_steps = 0
        self.raw_sum = np.zeros(self.length, dtype=float)
        self.normalized_sum = np.zeros(self.length, dtype=float)
        self.peak_raw_sum = np.zeros(len(self.sections), dtype=float)
        self.peak_normalized_sum = np.zeros(len(self.sections), dtype=float)
        self.context_sum = 0.0
        self.focus_entropy_sum = np.zeros(len(self.sections), dtype=float)
        self.focus_mass_log_sum = np.zeros(len(self.sections), dtype=float)
        self.focus_attended_steps = np.zeros(len(self.sections), dtype=np.int64)

    def update(self, prompt_attention) -> None:
        row = np.asarray(prompt_attention, dtype=float)
        if row.shape != (self.length,) or not np.isfinite(row).all() or (row < 0).any():
            raise ValueError("prompt attention must be a finite nonnegative vector of context length")
        context = float(row.sum())
        self.n_steps += 1
        self.raw_sum += row
        self.context_sum += context
        if context > 0:
            self.n_context_steps += 1
            self.normalized_sum += row / context
        # Include all tokens, independently of the mask used only for peaks.
        # Sum each span directly: subtracting prefix sums can erase the tiny
        # mass of a nearly unattended section after a high-attention span.
        log_row = _xlog2x(row)
        for i, (_, start, end, positions) in enumerate(self.sections):
            mass = float(row[start:end].sum())
            if mass > 0:
                mass_log_mass = mass * np.log2(mass)
                self.focus_entropy_sum[i] += mass_log_mass - float(log_row[start:end].sum())
                self.focus_mass_log_sum[i] += mass_log_mass
                self.focus_attended_steps[i] += 1
            if positions is not None:
                peak = float(np.partition(row[positions], -PEAK_K)[-PEAK_K:].mean())
                self.peak_raw_sum[i] += peak
                if context > 0:
                    self.peak_normalized_sum[i] += peak / context

    def finish(self, turn: int) -> list[dict]:
        rows = []
        for i, (name, start, end, positions) in enumerate(self.sections):
            row = dict.fromkeys(CONCENTRATION_FIELDS)
            row.update(version=CONCENTRATION_VERSION, turn=turn, section=name,
                       token_start=start, token_end=end, L=self.length,
                       n_gen=self.n_steps, n_context_steps=self.n_context_steps,
                       peak_k=PEAK_K, peak_status="complete" if positions is not None else "ineligible",
                       focus_status="no_attention" if self.n_steps else "no_steps")
            mass = float(self.raw_sum[start:end].sum())
            row.update(focus_attention_mass=mass, focus_n_attended_steps=int(self.focus_attended_steps[i]))
            if mass > 0:
                pooled_entropy = np.log2(mass) - float(_xlog2x(self.raw_sum[start:end]).sum()) / mass
                step_entropy = self.focus_entropy_sum[i] / mass
                weight_entropy = np.log2(mass) - self.focus_mass_log_sum[i] / mass
                js = pooled_entropy - step_entropy
                # Entropy concavity bounds the result. Clip only roundoff;
                # a larger violation indicates a broken accumulator.
                upper = min(np.log2(end - start), weight_entropy)
                if js < -1e-8 or js > upper + 1e-8:
                    raise ArithmeticError("attention-focus JS violates its entropy bounds")
                row.update(focus_js_bits=max(0.0, min(js, max(0.0, upper))),
                           focus_pooled_entropy_bits=max(0.0, pooled_entropy),
                           focus_step_entropy_bits=max(0.0, step_entropy),
                           focus_weight_entropy_bits=max(0.0, weight_entropy), focus_status="complete")
            if positions is not None and self.context_sum > 0:
                for weight, summed, peaks, denominator in (
                    ("mass_weighted", self.raw_sum, self.peak_raw_sum, self.context_sum),
                    ("equal_step", self.normalized_sum, self.peak_normalized_sum,
                     self.n_steps if self.n_context_steps == self.n_steps else 0),
                ):
                    if not denominator:
                        continue
                    pooled = self.length * float(np.partition(summed[positions], -PEAK_K)[-PEAK_K:].mean()) / denominator
                    step = self.length * float(peaks[i]) / denominator
                    row[f"peak_pooled_{weight}"] = pooled
                    row[f"peak_step_{weight}"] = step
                    # A zero denominator has no defined gap, even though both peaks are zero.
                    row[f"peak_gap_{weight}"] = max(0.0, 1.0 - pooled / step) if step > 0 else None
            rows.append(row)
        return rows


def write_concentration_turns(ep: str | Path, rows_by_turn: dict, *, sidecar: bool = False) -> None:
    """Replace several turns' rows in one read-modify-write; a turn mapped to an
    empty list invalidates its stale capture."""
    import pandas as pd

    if not rows_by_turn:
        return
    path = Path(ep) / ("replay_section_diagnostics.parquet" if sidecar else "section_diagnostics.parquet")
    frames = []
    if path.exists():
        old = pd.read_parquet(path)
        frames.append(old[~old.turn.isin(list(rows_by_turn))])
    for _t, rows in sorted(rows_by_turn.items()):
        if rows:
            frames.append(pd.DataFrame(rows))
    frames = [f for f in frames if len(f)]
    if frames:
        atomic_write_parquet(pd.concat(frames, ignore_index=True), path)
    elif path.exists():
        path.unlink()


def write_concentration(ep: str | Path, turn: int, rows: list[dict], *, sidecar: bool = False) -> None:
    """Replace a turn's rows; an empty replacement invalidates stale capture."""
    write_concentration_turns(ep, {turn: rows}, sidecar=sidecar)


def read_concentration(ep: str | Path) -> dict[tuple[int, str], dict]:
    import pandas as pd

    path = Path(ep) / "section_diagnostics.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if "version" not in df or not df.version.isin([1, CONCENTRATION_VERSION]).all():
        raise ValueError("unsupported section_diagnostics version")
    for field in FOCUS_FIELDS:
        if field not in df:
            df[field] = None
    if "focus_status" not in df:
        df["focus_status"] = "not_captured"
    else:
        df["focus_status"] = df.focus_status.fillna("not_captured")
    # Convert parquet NaNs to JSON nulls, including absent equal-step values.
    rows = df.astype(object).where(pd.notna(df), None).to_dict("records")
    return {(int(r["turn"]), r["section"]): r for r in rows}
