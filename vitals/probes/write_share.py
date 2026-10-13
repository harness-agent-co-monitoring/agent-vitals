"""Section shares of what every token-mixing layer writes into the hidden state, for one turn.

Every token-mixing layer -- softmax attention, sliding-window attention, compressed (MLA) attention or a
linear delta-rule layer -- ends with an output projection W_out applied to a per-head vector x:
w_l(t) = W_out x_l(t). Before W_out, x splits exactly into per-section parts x_{l,s} (vitals.engine.nodes
.WriteLayer says how for each kind), and W_out is linear, so the layer's write splits too. The write share of
section s at layer l and position t is

    <W x_s, W x> / |W x|^2  =  x_s . (W^T W x) / |W x|^2           (sums to 1 over sections)

so no per-section write has to be formed. The share counts what reaches the hidden state: heads the output
gate switches off and values that cancel count for little, unlike the raw attention weights.

``WriteShareProbe.measure`` teacher-forces one turn (prompt + logged output) on a fresh context and returns, per
section, the share averaged over the positions that predict the output tokens, for every layer, and its means
over all mixing layers (``write_all``), the attention layers (``write_attention``) and the linear layers
(``write_linear``), plus the attention layers' raw head-mean attention weight share (``weight``). With
``check`` the summed parts are compared with the engine's own tensor before W_out, per layer kind.

The backend must run with flash attention off when the model has MLA layers (their softmax rows are captured).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch

from vitals.engine.linear_memory import (
    SectionMemory,
    advance,
    concat_inputs,
    fp32_exact,
    layer_inputs,
    segments,
)
from vitals.engine.nodes import WriteLayer, linear_spec, write_plan
from vitals.engine.weights import GGUFWeights, default_device

_ACT = {"none": lambda g: g, "sigmoid": torch.sigmoid, "silu": torch.nn.functional.silu}


def pick(parts: dict, name: str, *tail: int) -> torch.Tensor:
    """Concatenate the captured parts of ``name`` whose trailing shape is ``tail``, tokens first. Same-named
    tensors of other shapes (llama.cpp names a raw projection and its reshaped view alike) are dropped."""
    ps = [p.reshape(-1, *tail) for p in parts.get(name, []) if tuple(p.shape[-len(tail):]) == tail]
    if not ps:
        raise KeyError(f"no capture for {name} with trailing shape {tail}")
    return torch.cat(ps).float()


def flat(parts: dict, name: str, width: int) -> torch.Tensor:
    """Concatenate the captured parts of ``name`` as rows of ``width`` values, tokens first."""
    ps = parts.get(name, [])
    if not ps:
        raise KeyError(f"no capture for {name}")
    return torch.cat([p.reshape(-1, width) for p in ps]).float()


_pick, _flat = pick, flat     # older names


class WriteShareProbe:
    """Per-section write shares of every token-mixing layer of a loaded model (Qwen3.5/3.6, Muse Glimmer,
    GLM-5.3-Flash). ``gguf_path`` is the model file the backend loaded (any shard)."""

    def __init__(self, backend, gguf_path: str, *, check: bool = True, chunk: int = 64, out_chunk: int = 64,
                 device: str | None = None, keep=None):
        """``keep``: optional predicate on ``WriteLayer``; only the layers it accepts are split (and their weights
        loaded)."""
        self.backend, self.eng = backend, backend.engine
        self.plan: list[WriteLayer] = write_plan(self.eng, backend.node_map)
        if keep is not None:
            self.plan = [x for x in self.plan if keep(x)]
        self.lin = linear_spec(self.eng, backend.node_map)
        if self.lin is not None:
            kept = tuple(x.layer for x in self.plan if x.kind == "linear")
            self.lin = replace(self.lin, layers=kept) if kept else None
        device = default_device(device)
        self.check, self.chunk, self.out_chunk, self.device = check, chunk, out_chunk, device
        w = GGUFWeights(gguf_path, device=device)
        self.W = {x.layer: w.get(x.out_weight) for x in self.plan}
        self.vb = {x.layer: w.get(x.vb_weight) for x in self.plan if x.vb_weight}
        self.norm = {x.layer: w.get(x.norm_weight, torch.float32) for x in self.plan if x.norm_weight}
        lin_inputs = self.lin.node_names() if self.lin else set()
        self.names_prompt = set().union(lin_inputs, *(x.nodes("prompt") for x in self.plan))
        self.names_wanted = set().union(lin_inputs, *(x.nodes("wanted", check) for x in self.plan))
        self._linear = [x for x in self.plan if x.kind == "linear"]

    # ------------------------------------------------------------------ public
    def measure(self, prompt: list[int], output: list[int], labels: list[str]) -> list[dict]:
        """``labels``: a section name for every prompt and output position (e.g. linear_reading.section_labels).
        Returns one row per section."""
        L, n = len(prompt), len(output)
        if len(labels) != L + n or L < 1 or n < 1:
            raise ValueError("labels must cover prompt + output, both non-empty")
        names = self._begin(labels)
        toks = list(prompt) + list(output)
        nb = int(self.eng._ctx_opts["n_batch"])
        with torch.no_grad(), fp32_exact():
            self.eng.set_capture(self.names_prompt)
            for off in range(0, L - 1, nb):
                part = toks[off:min(off + nb, L - 1)]
                self.eng.decode(part, pos0=off, seq_id=0)
                self._consume(off, len(part), wanted=False)
            self.eng.set_capture(self.names_wanted)
            for off in range(L - 1, L + n - 1, self.out_chunk):     # the positions that predict output tokens
                part = toks[off:min(off + self.out_chunk, L + n - 1)]
                self.eng.decode(part, pos0=off, seq_id=0)
                self._consume(off, len(part), wanted=True)
        return self._rows(names)

    def _begin(self, labels: list[str]) -> list[str]:
        """Fresh per-turn state for these position labels and a fresh engine context; returns the sections."""
        names = list(dict.fromkeys(labels))
        R = len(names)
        slot = {nm: i for i, nm in enumerate(names)}
        self._lab = torch.tensor([slot[x] for x in labels], device=self.device)
        self._onehot = torch.nn.functional.one_hot(self._lab, R).float()
        self._R = R
        self._kv = {x.layer: {"k": [], "v": []} for x in self.plan if x.kind != "linear"}
        self._mem = None
        if self.lin:
            self._mem = SectionMemory(self.lin, R, device=self.device)
            self._mem.n_slots = R
        self._share = {x.layer: torch.zeros(R, device=self.device) for x in self.plan}
        self._weight = {x.layer: torch.zeros(R, device=self.device) for x in self.plan if x.kind != "linear"}
        self._err = {k: [0.0, 0.0] for k in ("attention", "mla", "linear")}
        self._n_want = 0
        modes = dict.fromkeys(self.names_wanted, "device")
        self.eng.reset_context(capture_names=self.names_wanted, capture_modes=modes)
        return names

    # ------------------------------------------------------------------ per decode call
    def _consume(self, off: int, m: int, *, wanted: bool) -> None:
        self._process(self.eng.take_device_parts(), off, m, wanted=wanted)

    def _process(self, parts: dict, off: int, m: int, *, wanted: bool) -> None:
        for x in self.plan:                                   # keys / values of every position
            if x.kind == "attention":
                self._kv[x.layer]["k"].append(pick(parts, x.k_node.format(i=x.layer), x.n_kv, x.d_k))
                self._kv[x.layer]["v"].append(pick(parts, x.v_node.format(i=x.layer), x.n_kv, x.d_v))
            elif x.kind == "mla":
                lat = parts[x.v_node.format(i=x.layer)]
                self._kv[x.layer]["v"].append(torch.cat([p.reshape(-1, p.shape[-1]) for p in lat]).float())
        if self._mem is not None:
            self._advance_linear(parts, off, m, wanted)
        if wanted:
            for x in self.plan:
                if x.kind == "attention":
                    self._attention(x, parts, off, m)
                elif x.kind == "mla":
                    self._mla(x, parts, m)
            self._n_want += m

    def _write(self, x: WriteLayer, parts_x: torch.Tensor, ref: torch.Tensor | None, kind: str) -> None:
        """Accumulate the write shares of one layer: ``parts_x`` [R, m, H*d_v] per-section vectors before
        the output projection (their sum is the layer's own vector, checked against ``ref``)."""
        full = parts_x.sum(0)
        if ref is not None:
            d = full - ref
            self._err[kind][0] += float((d * d).sum())
            self._err[kind][1] += float((ref * ref).sum())
        W = self.W[x.layer].float()
        wf = full @ W.T                                         # [m, n_embd]
        u = wf @ W                                              # [m, H*d_v]
        den = (wf * wf).sum(-1).clamp_min(1e-30)
        num = (parts_x * u[None]).sum(-1)                       # [R, m]
        self._share[x.layer] += (num / den[None]).sum(1)

    def _attention(self, x: WriteLayer, parts: dict, off: int, m: int) -> None:
        lay = x.layer
        K, V = torch.cat(self._kv[lay]["k"]), torch.cat(self._kv[lay]["v"])      # [T, n_kv, d]
        T = K.shape[0]
        K, V = K.half().float(), V.half().float()               # the engine's cache holds f16
        q = pick(parts, x.q_node.format(i=lay), x.n_head, x.d_k)[-m:]
        g = flat(parts, x.gate_node.format(i=lay), x.n_head * x.d_v)[-m:].view(m, x.n_head, x.d_v) \
            if x.gate_node else None
        if g is not None:
            g = _ACT[x.gate_act](g)
        ref = flat(parts, x.check_node.format(i=lay), x.n_head * x.d_v)[-m:] if self.check and x.check_node else None
        rep = x.n_head // x.n_kv
        Ke, Ve = K[:, torch.arange(x.n_head, device=K.device) // rep], V[:, torch.arange(x.n_head, device=K.device) // rep]
        pos = torch.arange(off, off + m, device=K.device)
        keys = torch.arange(T, device=K.device)
        s = torch.einsum("mhd,thd->hmt", q, Ke) / math.sqrt(x.d_k)
        mask = keys[None, :] > pos[:, None]
        if x.window:
            mask |= (pos[:, None] - keys[None, :]) >= x.window
        A = torch.softmax(s.masked_fill(mask[None], float("-inf")), -1)            # [H, m, T]
        self._accumulate(x, A, Ve, g, ref, "attention", T)

    def _mla(self, x: WriteLayer, parts: dict, m: int) -> None:
        lay = x.layer
        lat = torch.cat(self._kv[lay]["v"])                                         # [T, d_latent]
        T = lat.shape[0]
        rows = [p.reshape(x.n_head, -1, p.shape[-1]) for p in parts[x.rows_node.format(i=lay)]]
        A = torch.cat([torch.nn.functional.pad(r, (0, max(0, T - r.shape[-1]))) for r in rows], 1)[:, -m:, :T].float()
        ref = flat(parts, x.check_node.format(i=lay), x.n_head * x.d_v)[-m:] if self.check and x.check_node else None
        vb = self.vb[lay].float()                                                   # [H, d_v, d_latent]
        R = self._R
        o = torch.stack([torch.einsum("hmt,tc->hmc", A * self._onehot[:T, r][None, None], lat) for r in range(R)])
        parts_x = torch.einsum("rhmc,hdc->rmhd", o, vb).reshape(R, m, x.n_head * x.d_v)
        self._weight[lay] += (torch.einsum("hmt,tr->mr", A, self._onehot[:T]) / x.n_head).sum(0)
        self._write(x, parts_x, ref, "mla")

    def _accumulate(self, x: WriteLayer, A: torch.Tensor, Ve: torch.Tensor, g, ref, kind: str, T: int) -> None:
        R, m = self._R, A.shape[1]
        oh = self._onehot[:T]
        self._weight[x.layer] += (torch.einsum("hmt,tr->mr", A, oh) / x.n_head).sum(0)
        members = [torch.nonzero(oh[:, r]).flatten() for r in range(R)]
        o = torch.stack([torch.einsum("hmt,thd->mhd", A[:, :, idx], Ve[idx]) if len(idx)
                         else torch.zeros(m, x.n_head, x.d_v, device=A.device) for idx in members])  # [R, m, H, d]
        if g is not None:
            o = o * g[None]
        self._write(x, o.reshape(R, m, x.n_head * x.d_v), ref, kind)

    def _advance_linear(self, parts: dict, off: int, m: int, wanted: bool) -> None:
        spec, mem, R = self.lin, self._mem, self._R
        nv, dv = spec.n_v_heads, spec.d_v
        lab = self._lab[off:off + m]
        segs = segments(lab)
        want = torch.ones(m, dtype=torch.bool, device=lab.device) if wanted else None
        for j, x in enumerate(self._linear):
            xi = layer_inputs(spec, parts, x.layer, device=self.device)
            S = mem.S[:, mem.heads(j)]
            hook = None
            if wanted:
                gate = _ACT[x.gate_act](flat(parts, x.gate_node.format(i=x.layer), nv * dv).view(-1, nv, dv)[-m:])
                ref = None
                if self.check and x.check_node:
                    ref = flat(parts, x.check_node.format(i=x.layer), nv * dv).view(-1, nv, dv)[-m:]
                    if x.check_pre_gate:
                        ref = ref * gate
                hook = self._linear_hook(x, gate, ref)
            advance(S, concat_inputs([xi]), lab, R, spec.decay, want=want, chunk=self.chunk, segs=segs,
                    on_output=hook)

    def _linear_hook(self, x: WriteLayer, gate: torch.Tensor, ref: torch.Tensor | None):
        w_norm = self.norm[x.layer]

        def hook(o: torch.Tensor, full: torch.Tensor, positions: list[int]) -> None:
            idx = torch.tensor(positions, device=o.device)
            rms = torch.sqrt((full * full).mean(-1) + x.norm_eps)                    # [H, n]
            gt = gate[idx].transpose(0, 1)                                          # [H, n, d]
            scale = (w_norm[None, None] * gt / rms[..., None])                      # [H, n, d]
            parts_x = (o * scale[None]).permute(0, 2, 1, 3).reshape(o.shape[0], len(positions), -1)
            r = ref[idx].reshape(len(positions), -1) if ref is not None else None
            self._write(x, parts_x, r, "linear")
        return hook

    # ------------------------------------------------------------------ summary
    def _rows(self, names: list[str]) -> list[dict]:
        n = max(1, self._n_want)
        share = np.stack([(self._share[x.layer] / n).cpu().numpy() for x in self.plan])       # [layers, R]
        att = [i for i, x in enumerate(self.plan) if x.kind != "linear"]
        lin = [i for i, x in enumerate(self.plan) if x.kind == "linear"]
        weight = np.stack([(self._weight[self.plan[i].layer] / n).cpu().numpy() for i in att]) if att else None
        rel = {k: (math.sqrt(e / r) if r > 0 else None) for k, (e, r) in self._err.items()}
        rows = []
        for s, name in enumerate(names):
            rows.append({
                "section": name,
                "write_all": float(share[:, s].mean()),
                "write_attention": float(share[att, s].mean()) if att else None,
                "write_linear": float(share[lin, s].mean()) if lin else None,
                "weight": float(weight[:, s].mean()) if weight is not None else None,
                "per_layer": share[:, s].astype(np.float32).tolist(),
                "n_positions": self._n_want,
                **{f"check_rel_err_{k}": v for k, v in rel.items() if v is not None},
            })
        return rows

    @property
    def layer_kinds(self) -> list[tuple[int, str]]:
        return [(x.layer, x.kind) for x in self.plan]
