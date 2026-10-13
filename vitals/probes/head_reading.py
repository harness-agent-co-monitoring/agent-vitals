"""Per-head section reading: how much attention each head of each attention layer puts on each section.

The read maps store the head average of this quantity. Calibration needs it per head: a weighted sum over heads,
with weights fit to section ablation effects (``vitals.analysis.calibration``), estimates how much each section
matters to a decision in one pass. Equal weights give back the read map.

``HeadReading.measure`` teacher-forces a token sequence with a section label per position and returns, per head
and section, the attention weight averaged over a span of positions (for example the reasoning that leads to a
decision) and at each decision position; with ``positions=True`` also the head-mean attention each key position
gets over the span (read peaks and hottest lines). Attention is recomputed from the captured queries and keys, so
no model weights are loaded.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from vitals.engine.linear_memory import fp32_exact
from vitals.engine.nodes import write_plan
from vitals.engine.weights import default_device
from vitals.probes.write_share import pick


class HeadReading:
    """Per-head attention from chosen positions to each section, from one teacher-forced pass over a labeled token
    sequence (``measure``), for every softmax-attention layer; optionally the residual at ``residual_layers``."""

    def __init__(self, backend, *, chunk: int = 64, device: str | None = None, residual_layers: list[int] | None = None):
        self.eng = backend.engine
        self.plan = [x for x in write_plan(self.eng, backend.node_map) if x.kind == "attention"]
        if not self.plan:
            raise ValueError("no softmax attention layers in this model")
        self.res = {int(L): backend.node_map.residual_node(int(L)) for L in (residual_layers or [])}
        self.names_prompt = {x.k_node.format(i=x.layer) for x in self.plan} | set(self.res.values())
        self.names_wanted = self.names_prompt | {x.q_node.format(i=x.layer) for x in self.plan}
        self.chunk, self.device = chunk, default_device(device)
        self.heads = [(x.layer, h) for x in self.plan for h in range(x.n_head)]

    def measure(self, tokens: list[int], labels: list[str], *, span: tuple[int, int], decisions: list[int],
                positions: bool = False) -> dict:
        """``span``: positions [start, end) whose attention is averaged; ``decisions``: positions read alone.
        Returns sections, heads (layer, head), ``span`` [n_heads, R] and ``at`` {position: [n_heads, R]}. With
        ``positions``: ``span_pos`` [last + 1], the attention each key position gets averaged over the span and over
        heads (summed by section it gives the head mean of ``span``). With ``residual_layers``: ``residual_at``
        {position: [n_layers, n_embd]} and ``section_residual`` [n_layers, R, n_embd], the mean residual over each
        section's positions (the section activation cache)."""
        s0, s1 = int(span[0]), int(span[1])
        dec = sorted({int(p) for p in decisions})
        last = max([s1 - 1, *dec])
        if s0 < 0 or s1 <= s0 or last >= len(tokens) or len(labels) < last + 1:
            raise ValueError("span and decisions must lie inside the labelled token sequence")
        names = list(dict.fromkeys(labels[:last + 1]))
        slot = {nm: i for i, nm in enumerate(names)}
        self._oh = torch.nn.functional.one_hot(torch.tensor([slot[x] for x in labels[:last + 1]], device=self.device),
                                               len(names)).float()
        self._kv = {x.layer: {"k": [], "v": []} for x in self.plan}
        want = set(range(s0, s1)) | set(dec)
        self._sum = torch.zeros(len(self.heads), len(names), device=self.device)
        self._at: dict[int, torch.Tensor] = {}
        self._pos = torch.zeros(last + 1, device=self.device) if positions else None
        self._span, self._dec = (s0, s1), set(dec)
        self._rsum = {L: None for L in self.res}
        self._rat: dict[int, dict[int, torch.Tensor]] = {}
        self.eng.reset_context(capture_names=self.names_wanted, capture_modes=dict.fromkeys(self.names_wanted, "device"))
        toks = list(tokens)
        nb = int(self.eng._ctx_opts["n_batch"])
        first = min(want)
        with torch.no_grad(), fp32_exact():
            self.eng.set_capture(self.names_prompt)
            for off in range(0, first, nb):
                part = toks[off:min(off + nb, first)]
                self.eng.decode(part, pos0=off, seq_id=0)
                parts = self.eng.take_device_parts()
                self._keys(parts)
                self._residual(parts, off, len(part))
            self.eng.set_capture(self.names_wanted)
            for off in range(first, last + 1, self.chunk):
                part = toks[off:min(off + self.chunk, last + 1)]
                self.eng.decode(part, pos0=off, seq_id=0)
                parts = self.eng.take_device_parts()
                self._keys(parts)
                self._residual(parts, off, len(part))
                self._read(parts, off, len(part))
        n = max(1, s1 - s0)
        out = {"sections": names, "heads": self.heads, "span": (self._sum / n).cpu().numpy(),
               "at": {p: v.cpu().numpy() for p, v in self._at.items()}}
        if self._pos is not None:
            out["span_pos"] = (self._pos / (n * len(self.heads))).cpu().numpy()
        if self.res:
            counts = self._oh.sum(0).clamp_min(1.0)[:, None]
            out["residual_layers"] = list(self.res)
            out["section_residual"] = torch.stack([self._rsum[L] / counts for L in self.res]).cpu().numpy()
            out["residual_at"] = {p: torch.stack([self._rat[p][L] for L in self.res]).cpu().numpy() for p in self._rat}
        return out

    def _residual(self, parts: dict, off: int, m: int) -> None:
        if not self.res:
            return
        oh = self._oh[off:off + m]
        for L, node in self.res.items():
            h = torch.cat([q.reshape(-1, q.shape[-1]) for q in parts[node]]).float()[-m:]      # [m, n_embd]
            acc = oh.T @ h
            self._rsum[L] = acc if self._rsum[L] is None else self._rsum[L] + acc
            for i in range(m):
                if off + i in self._dec:
                    self._rat.setdefault(off + i, {})[L] = h[i].clone()

    def _keys(self, parts: dict) -> None:
        for x in self.plan:
            self._kv[x.layer]["k"].append(pick(parts, x.k_node.format(i=x.layer), x.n_kv, x.d_k).half())

    def _read(self, parts: dict, off: int, m: int) -> None:
        pos = torch.arange(off, off + m, device=self.device)
        in_span = (pos >= self._span[0]) & (pos < self._span[1])
        is_dec = [i for i in range(m) if off + i in self._dec]
        row = 0
        for x in self.plan:
            K = torch.cat(self._kv[x.layer]["k"]).float()                                    # [T, n_kv, d]
            T = K.shape[0]
            q = pick(parts, x.q_node.format(i=x.layer), x.n_head, x.d_k)[-m:]
            Ke = K[:, torch.arange(x.n_head, device=K.device) // (x.n_head // x.n_kv)]
            s = torch.einsum("mhd,thd->hmt", q, Ke) / math.sqrt(x.d_k)
            keys = torch.arange(T, device=K.device)
            mask = keys[None, :] > pos[:, None]
            if x.window:
                mask |= (pos[:, None] - keys[None, :]) >= x.window
            A = torch.softmax(s.masked_fill(mask[None], float("-inf")), -1)                 # [H, m, T]
            mass = torch.einsum("hmt,tr->hmr", A, self._oh[:T])                             # [H, m, R]
            self._sum[row:row + x.n_head] += mass[:, in_span].sum(1)
            if self._pos is not None:
                self._pos[:T] += A[:, in_span].sum((0, 1))
            for i in is_dec:
                p = off + i
                if p not in self._at:
                    self._at[p] = torch.zeros_like(self._sum)
                self._at[p][row:row + x.n_head] = mass[:, i]
            row += x.n_head


def as_read_map(reading: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Collapse [n_heads, R] per-head reading to one value per section: weighted by ``weights`` (calibrated)
    or averaged over heads (the read map)."""
    if weights is None:
        return reading.mean(0)
    return np.asarray(weights, dtype=np.float64) @ reading
