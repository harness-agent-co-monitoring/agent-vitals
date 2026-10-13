"""Signed section attribution of decisions: how far each context section pushes the model toward one next
token over another, through the direct path of every token-mixing layer.

At a position that predicts the next token, the logit margin between tokens a and b is linear in the final
hidden state h once the final RMSNorm scale is fixed at its value:

    m = z_a - z_b = d . h,    d = gamma * (W_U[a] - W_U[b]) / rms(h)

Every token-mixing layer adds W_out x_l to h, and x_l splits exactly into per-section parts (see
``write_share``), so section s contributes sum_l (W_out x_{l,s}) . d. What the sections together leave of m
is the remainder: the token embedding, the MLPs, and whatever sections reach the decision only through later
layers' nonlinear paths. The sign says which way a section pushes; the size is in logits.

``SectionAttribution.attribute`` takes a token sequence (prompt plus any forced output) with a section label
per position and named decisions (position, a, b); positions between decisions are decoded in large chunks
and each decision position on its own. It returns one row per decision and section, and per decision the
engine's own margin, the linearized margin d . h and the remainder. ``linear=False`` skips the recurrent
(DeltaNet) layers, whose writes then fall into the remainder; that is much cheaper with many sections.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from vitals.engine.linear_memory import fp32_exact
from vitals.engine.weights import GGUFWeights
from vitals.probes.write_share import WriteShareProbe, flat


class SectionAttribution(WriteShareProbe):
    """Signed per-section contributions to the logit margin of named decisions, through the direct path of every
    token-mixing layer (``attribute``; see the module docstring)."""

    def __init__(self, backend, gguf_path: str, *, linear: bool = True, check: bool = False, chunk: int = 64,
                 norm_eps: float = 1e-6, device: str | None = None):
        super().__init__(backend, gguf_path, check=check, chunk=chunk, out_chunk=1, device=device)
        if not linear:
            self.lin, self._linear = None, []
            self.names_prompt = set().union(*(x.nodes("prompt") for x in self.plan if x.kind != "linear"))
            self.names_wanted = set().union(*(x.nodes("wanted", check) for x in self.plan if x.kind != "linear"))
        nm = backend.node_map
        self.h_node = nm.residual_node(nm.n_layers - 1)
        self.names_wanted = self.names_wanted | {self.h_node}
        self._wts = GGUFWeights(gguf_path, device=device)
        self.gamma = self._wts.get("output_norm.weight", torch.float32)
        self.unembed = "output.weight" if "output.weight" in self._wts else "token_embd.weight"
        self.norm_eps = norm_eps
        self._dirs: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ public
    def attribute(self, tokens: list[int], labels: list[str], decisions: dict[str, tuple[int, int, int]]) -> dict:
        """``decisions``: name -> (position, a, b); attribute the margin z_a - z_b of the distribution at
        ``position`` (the one predicting the token after it). ``labels``: a section name per position of
        ``tokens``, which must reach the last decision position."""
        if not decisions:
            raise ValueError("need at least one decision")
        dec = {k: (int(p), int(a), int(b)) for k, (p, a, b) in decisions.items()}
        last = max(p for p, _, _ in dec.values())
        if not 0 < last < len(tokens) or len(labels) < last + 1 or min(p for p, _, _ in dec.values()) < 1:
            raise ValueError("decision positions must lie inside the labelled token sequence (after position 0)")
        ids = sorted({t for _, a, b in dec.values() for t in (a, b)})
        self._rows_U = dict(zip(ids, self._wts.rows(self.unembed, ids), strict=True))
        self._signed = {k: {} for k in dec}
        self._summary = {}
        names = self._begin(list(labels[:last + 1]))
        by_pos: dict[int, list[str]] = {}
        for k, (p, _, _) in dec.items():
            by_pos.setdefault(p, []).append(k)
        toks = list(tokens)
        nb = int(self.eng._ctx_opts["n_batch"])
        cursor = 0
        with torch.no_grad(), fp32_exact():
            for p in sorted(by_pos):
                self.eng.set_capture(self.names_prompt)
                for off in range(cursor, p, nb):
                    part = toks[off:min(off + nb, p)]
                    self.eng.decode(part, pos0=off, seq_id=0)
                    self._consume(off, len(part), wanted=False)
                self.eng.set_capture(self.names_wanted)
                self._active = {k: dec[k] for k in by_pos[p]}
                self.eng.decode([toks[p]], pos0=p, seq_id=0)
                logits = np.asarray(self.eng.logits(-1), dtype=np.float64)
                for k, (_, a, b) in self._active.items():
                    self._summary[k] = {"position": p, "a": a, "b": b, "margin": float(logits[a] - logits[b])}
                self._consume(p, 1, wanted=True)
                self._dirs, cursor = {}, p + 1
        shares = {r["section"]: r["write_all"] for r in self._rows(names)}
        kinds = [x.kind for x in self.plan]
        att = [i for i, kd in enumerate(kinds) if kd != "linear"]
        lin = [i for i, kd in enumerate(kinds) if kd == "linear"]
        rows = []
        for k in dec:
            per = np.stack([self._signed[k][x.layer].cpu().numpy() if x.layer in self._signed[k]
                            else np.zeros(len(names)) for x in self.plan])                  # [layers, R]
            direct = per.sum(0)
            s = self._summary[k]
            s.update(direct_total=float(direct.sum()), remainder=s["margin_linear"] - float(direct.sum()),
                     direct_linear_layers=float(per[lin].sum()) if lin else 0.0)
            for i, name in enumerate(names):
                rows.append({"decision": k, "section": name, "direct": float(direct[i]),
                             "direct_attention": float(per[att, i].sum()),
                             "direct_linear": float(per[lin, i].sum()) if lin else 0.0,
                             "per_layer": per[:, i].astype(np.float32).tolist(), "write_share": shares.get(name)})
        return {"rows": rows, "summary": self._summary}

    # ------------------------------------------------------------------ hooks
    def _process(self, parts: dict, off: int, m: int, *, wanted: bool) -> None:
        if wanted:
            h = flat(parts, self.h_node, self.gamma.shape[0])[-1]                          # [n_embd]
            rms = torch.sqrt((h * h).mean() + self.norm_eps)
            self._dirs = {}
            for k, (_, a, b) in self._active.items():
                d = self.gamma * (self._rows_U[a] - self._rows_U[b]) / rms
                self._dirs[k] = d
                self._summary[k]["margin_linear"] = float((d * h).sum())
        super()._process(parts, off, m, wanted=wanted)

    def _write(self, x, parts_x: torch.Tensor, ref, kind: str) -> None:
        super()._write(x, parts_x, ref, kind)
        if not self._dirs:
            return
        W = self.W[x.layer].float()
        for k, d in self._dirs.items():
            c = (parts_x[:, -1] * (d @ W)[None]).sum(-1)                                     # [R]
            self._signed[k][x.layer] = c if x.layer not in self._signed[k] else self._signed[k][x.layer] + c


def runner_up(logits, chosen: int, exclude: set[int] | None = None) -> int:
    """The strongest competitor of ``chosen`` in a logits vector."""
    z = np.asarray(logits, dtype=np.float64).copy()
    z[chosen] = -math.inf
    for t in exclude or ():
        z[t] = -math.inf
    return int(np.argmax(z))
