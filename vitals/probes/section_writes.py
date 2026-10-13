"""Per-section writes of the token-mixing layers at chosen positions, for section-factored linear probes.

A linear probe w on the hidden state h_L after layer L scores s = w . h_L. Every token-mixing layer l <= L adds
W_out x_l to h_L, and x_l splits exactly into per-section parts (see ``write_share``), so

    w . h_L = sum_s w . A_{s,L} + w . (h_L - sum_s A_{s,L}),    A_{s,L} = sum_{mixing l <= L} W_out x_{l,s}

The last term is the remainder: the token embedding and the MLPs, plus the recurrent layers when they are not
split.

``SectionWrites.capture`` teacher-forces a token sequence with a section label per position and returns, at
each requested position and each cutoff layer L: h_L, A_{s,L} for every section, and the head-mean attention
weight on each section averaged over the attention layers <= L (the read map at that token). By default only
the attention layers are split. With ``linear=True`` the recurrent (delta-rule) layers are split too, exactly,
by advancing one memory per section (``vitals.engine.linear_memory``); their part of A_{s,L} is also returned
on its own as ``writes_linear``.

With ``project=w`` (a direction in the hidden state, e.g. a probe's effective weights) ``capture`` also returns each
earlier position's contribution to w . h_L through the attention layers: sum over heads of the attention weight
times w . W_out (gate * value). Summed over a section's positions it equals w . (the section's attention writes).
The recurrent layers are not split per position: that would need one memory per token.
"""

from __future__ import annotations

import numpy as np
import torch

from vitals.engine.linear_memory import fp32_exact
from vitals.probes.write_share import WriteShareProbe, flat


class SectionWrites(WriteShareProbe):
    """Per-section writes into the hidden state at cutoff ``layers``, at chosen positions (see the module
    docstring); ``section_writes`` returns one reused per backend."""

    def __init__(self, backend, gguf_path: str, *, layers, linear: bool = False, check: bool = False,
                 device: str | None = None):
        nm = backend.node_map
        self.layers = sorted(int(L) for L in layers)
        if not self.layers or self.layers[0] < 0 or self.layers[-1] >= nm.n_layers:
            raise ValueError(f"cutoff layers must lie in [0, {nm.n_layers})")
        top = self.layers[-1]
        # layers above the last cutoff never reach a captured residual: neither split nor loaded
        super().__init__(backend, gguf_path, check=check, out_chunk=1, device=device,
                         keep=lambda x: x.layer <= top and (linear or x.kind != "linear"))
        self._u = None
        if not self.W:
            raise ValueError(f"no token-mixing layer at or below layer {top}")
        self.n_embd = int(next(iter(self.W.values())).shape[0])
        lin_inputs = self.lin.node_names() if self.lin else set()
        self.names_prompt = set().union(lin_inputs, *(x.nodes("prompt") for x in self.plan))
        self.names_wanted = set().union(lin_inputs, *(x.nodes("wanted", check) for x in self.plan))
        self.res = {L: nm.residual_node(L) for L in self.layers}
        self.names_wanted = self.names_wanted | set(self.res.values())

    def capture(self, tokens: list[int], labels: list[str], positions, *, project=None) -> dict:
        """``labels``: a section name per position, reaching the last requested position. Returns the
        sections, the cutoff layers and per position ``h`` [n_cut, n_embd], ``writes`` [n_cut, R, n_embd] and
        ``read`` [n_cut, R] (float32 arrays), plus ``writes_linear`` [n_cut, R, n_embd] (the recurrent layers'
        part of ``writes``) when the recurrent layers are split, and ``contrib_pos`` [n_cut, p + 1] (each
        position's contribution to ``project`` . h through the attention layers) when ``project`` is given."""
        pos = sorted({int(p) for p in positions})
        if not pos or pos[0] < 0 or pos[-1] >= len(tokens) or len(labels) < pos[-1] + 1:
            raise ValueError("positions must lie inside the labelled token sequence")
        self._u = None
        if project is not None:
            wt = torch.as_tensor(np.asarray(project, dtype=np.float32), device=self.device)
            self._u = {x.layer: self.W[x.layer].float().T @ wt for x in self.plan if x.kind == "attention"}
        names = self._begin(list(labels[:pos[-1] + 1]))
        toks = list(tokens)
        nb = int(self.eng._ctx_opts["n_batch"])
        out, cursor = {}, 0
        with torch.no_grad(), fp32_exact():
            for p in pos:
                self.eng.set_capture(self.names_prompt)
                for off in range(cursor, p, nb):
                    part = toks[off:min(off + nb, p)]
                    self.eng.decode(part, pos0=off, seq_id=0)
                    self._consume(off, len(part), wanted=False)
                self.eng.set_capture(self.names_wanted)
                self._vec, self._cpos = {}, {}
                for w in self._weight.values():
                    w.zero_()
                self.eng.decode([toks[p]], pos0=p, seq_id=0)
                self._consume(p, 1, wanted=True)
                out[p] = self._snapshot()
                cursor = p + 1
        return {"sections": names, "layers": self.layers, "positions": out}

    def _process(self, parts: dict, off: int, m: int, *, wanted: bool) -> None:
        if wanted:
            self._h = {L: flat(parts, n, self.n_embd)[-1] for L, n in self.res.items()}
        super()._process(parts, off, m, wanted=wanted)

    def _accumulate(self, x, A, Ve, g, ref, kind: str, T: int) -> None:
        super()._accumulate(x, A, Ve, g, ref, kind, T)
        if self._u is not None and x.layer in self._u:
            u = self._u[x.layer].view(x.n_head, x.d_v)
            gu = u if g is None else g[-1] * u                                       # [H, d_v]
            self._cpos[x.layer] = torch.einsum("ht,th->t", A[:, -1, :], torch.einsum("thd,hd->th", Ve, gu))

    def _write(self, x, parts_x: torch.Tensor, ref, kind: str) -> None:
        super()._write(x, parts_x, ref, kind)
        self._vec[x.layer] = parts_x[:, -1] @ self.W[x.layer].float().T                  # [R, n_embd]

    def _snapshot(self) -> dict:
        zero = torch.zeros(self._R, self.n_embd, device=self.device)
        h, writes, lin, read, cpos = [], [], [], [], []
        for L in self.layers:
            upto = [lay for lay in sorted(self._vec) if lay <= L]
            att = [lay for lay in upto if lay in self._weight]               # attention layers have read weights
            writes.append(sum((self._vec[lay] for lay in upto), zero))
            lin.append(sum((self._vec[lay] for lay in upto if lay not in self._weight), zero))
            read.append(torch.stack([self._weight[lay] for lay in att]).mean(0) if att else torch.zeros(self._R, device=self.device))
            h.append(self._h[L])
            if self._u is not None:
                cl = [self._cpos[lay] for lay in att if lay in self._cpos]
                cpos.append(sum(cl[1:], cl[0]) if cl else torch.zeros(len(self._lab), device=self.device))
        out = {"h": torch.stack(h).cpu().numpy().astype(np.float32),
               "writes": torch.stack(writes).cpu().numpy().astype(np.float32),
               "read": torch.stack(read).cpu().numpy().astype(np.float32)}
        if self._linear:
            out["writes_linear"] = torch.stack(lin).cpu().numpy().astype(np.float32)
        if self._u is not None:
            n = min(len(c) for c in cpos)
            out["contrib_pos"] = torch.stack([c[:n] for c in cpos]).cpu().numpy().astype(np.float32)
        return out


def section_writes(backend, gguf_path: str, *, layers, linear: bool = False) -> SectionWrites:
    """A ``SectionWrites`` for this backend, reused across calls (building one loads output weights). The cache
    lives on the backend, so closing and dropping the backend frees the weights too."""
    cache = backend.__dict__.setdefault("_section_writes", {})
    key = (str(gguf_path), tuple(sorted(int(L) for L in layers)), bool(linear))
    sw = cache.get(key)
    if sw is None:
        sw = cache[key] = SectionWrites(backend, gguf_path, layers=layers, linear=linear)
    return sw
