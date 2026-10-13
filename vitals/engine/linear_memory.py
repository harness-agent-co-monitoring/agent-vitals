"""Exact per-section split of linear-attention (delta-rule) memories.

A linear-attention layer (Gated DeltaNet in Qwen3.5/3.6, Kimi Delta Attention in GLM-5.3-Flash) keeps one
memory S (key x value) per value head instead of a key/value cache. Per token,

    S <- (I - b k k^T) D S + b k v^T,        o = S^T q        (q carries the 1/sqrt(d_k) scale)

where D fades the memory: one factor per head (GDN) or one per key channel (KDA). D, b and k depend only on
the token, never on S, so S is linear in the written values v and in the starting memory. Splitting the
values by section (v zeroed outside section r) and running the same update gives one memory S_r per
section, exactly, with sum_r S_r = S and sum_r o_r = o. This module advances those per-section memories.

The update runs in chunks of tokens as matrix products (the WY form of the delta rule, generalized to a
per-channel fade), with every section batched. Within a chunk with cumulative log fades G (G_t = sum of g
up to t), and E[t, j] = exp(G_t - G_j) for j <= t (applied per channel for KDA):

    A[t, j]  = sum_c k_t k_j E[t, j]            strictly lower
    M[t, j]  = sum_c q_t k_j E[t, j]            lower, diagonal included
    T        = (I + diag(b) A)^-1 diag(b)
    U_r      = T (V_r - K~ S_r),     K~ = exp(G) * K
    O_r      = Q~ S_r + M U_r,       Q~ = exp(G) * Q
    S_r     <- exp(G_end) * S_r + (exp(G_end - G) * K)^T U_r

Every exponent is <= 0, so strong fades underflow to zero rather than overflow. The reference
implementation (one token at a time, ``reference_outputs``) is kept for tests.

Matrix products run in full fp32 (TF32 off): the split is checked against the engine's own output.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from vitals.engine.nodes import LinearMixerSpec
from vitals.engine.weights import default_device


@contextmanager
def fp32_exact() -> Iterator[None]:
    """Full-precision fp32 matrix products for the duration (TF32 off)."""
    prev = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev[0]
        torch.set_float32_matmul_precision(prev[1])


# --------------------------------------------------------------------------- inputs

@dataclass
class LayerInputs:
    """One layer group's inputs for T tokens, heads concatenated over the group's layers."""

    q: torch.Tensor                 # [T, H, d_k]  L2-normalized, times 1/sqrt(d_k)
    k: torch.Tensor                 # [T, H, d_k]  L2-normalized
    v: torch.Tensor                 # [T, H, d_v]
    g: torch.Tensor                 # [T, H] (fade per head) or [T, H, d_k] (per channel); log fade, <= 0
    beta: torch.Tensor              # [T, H]
    ref: torch.Tensor | None = None  # [T, H, d_v]  the engine's own output, for the exactness check


def _l2(x: torch.Tensor, eps: float) -> torch.Tensor:
    # x / sqrt(|x|^2 + eps). llama.cpp's GDN path uses x / max(|x|, eps); with eps = 1e-6 and |x| of order
    # one the two agree far below fp32 resolution.
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def layer_inputs(spec: LinearMixerSpec, parts: dict, layer: int, *, check: bool = False,
                 device: str | torch.device | None = None) -> LayerInputs:
    """Inputs of one linear layer from an engine's "device"-mode captures (name -> list of per-ubatch
    tensors, tokens leading after the shape reversal)."""
    nk, nv, dk, dv = spec.n_k_heads, spec.n_v_heads, spec.d_k, spec.d_v
    device = default_device(device)

    def cat(fmt: str, *tail: int) -> torch.Tensor:
        ps = parts.get(fmt.format(i=layer))
        if not ps:
            raise KeyError(f"no capture for {fmt.format(i=layer)}")
        return torch.cat([p.reshape(-1, *tail) for p in ps]).to(device=device, dtype=torch.float32)

    if spec.qkv_node:
        qkv = cat(spec.qkv_node, 2 * nk * dk + nv * dv)
        q = qkv[:, :nk * dk].view(-1, nk, dk)
        k = qkv[:, nk * dk:2 * nk * dk].view(-1, nk, dk)
        v = qkv[:, 2 * nk * dk:].view(-1, nv, dv)
    else:
        q, k, v = cat(spec.q_node, nk, dk), cat(spec.k_node, nk, dk), cat(spec.v_node, nv, dv)
    if nk != nv:
        idx = torch.arange(nv, device=q.device) % nk          # value head h reads key head h % nk
        q, k = q[:, idx], k[:, idx]
    q = _l2(q, spec.norm_eps) * dk ** -0.5
    k = _l2(k, spec.norm_eps)
    g = cat(spec.g_node, nv) if spec.decay == "head" else cat(spec.g_node, nv, dk)
    beta = cat(spec.beta_node, nv)
    ref = cat(spec.out_node, nv, dv) if check else None
    return LayerInputs(q=q, k=k, v=v, g=g, beta=beta, ref=ref)


def concat_inputs(xs: list[LayerInputs]) -> LayerInputs:
    if len(xs) == 1:
        return xs[0]
    return LayerInputs(
        q=torch.cat([x.q for x in xs], 1), k=torch.cat([x.k for x in xs], 1), v=torch.cat([x.v for x in xs], 1),
        g=torch.cat([x.g for x in xs], 1), beta=torch.cat([x.beta for x in xs], 1),
        ref=None if xs[0].ref is None else torch.cat([x.ref for x in xs], 1))


# --------------------------------------------------------------------------- memories

class SectionMemory:
    """Per-section memories of every linear layer: S[r] for section slot r, heads of all layers along the
    second axis (layer-major). The first ``n_slots`` slots are in use; sum over them is the real memory."""

    def __init__(self, spec: LinearMixerSpec, max_slots: int, device: str | torch.device | None = None):
        self.spec = spec
        self.max_slots = max_slots
        self.n_heads = len(spec.layers) * spec.n_v_heads
        device = default_device(device)
        self.S = torch.zeros(max_slots, self.n_heads, spec.d_k, spec.d_v, device=device)
        self.n_slots = 0

    def heads(self, layer_pos: int, n_layers: int = 1) -> slice:
        """Head range of ``n_layers`` consecutive linear layers starting at position ``layer_pos``."""
        h = self.spec.n_v_heads
        return slice(layer_pos * h, (layer_pos + n_layers) * h)

    def reset(self) -> None:
        self.S[:self.n_slots].zero_()
        self.n_slots = 0

    def copy_from(self, other: SectionMemory) -> None:
        n = max(self.n_slots, other.n_slots)
        self.S[:n].copy_(other.S[:n])
        self.n_slots = other.n_slots

    def merge(self, src: int, dst: int) -> None:
        """Fold slot ``src`` into ``dst`` (exact, by linearity) and clear ``src``."""
        self.S[dst] += self.S[src]
        self.S[src].zero_()

    def energy(self) -> torch.Tensor:
        """[n_slots] squared Frobenius norm of each section's memory, summed over all heads."""
        return torch.stack([torch.linalg.vector_norm(self.S[r]) ** 2 for r in range(self.n_slots)]) \
            if self.n_slots else torch.zeros(0, device=self.S.device)

    def cross(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per layer, each section's projection on the whole memory: ([n_slots, n_layers] <S_r, S> summed
        over the layer's heads, [n_layers] |S|^2). Each layer's projections sum to its |S|^2."""
        n_layers, nv = len(self.spec.layers), self.spec.n_v_heads
        if not self.n_slots:
            return torch.zeros(0, n_layers, device=self.S.device), torch.zeros(n_layers, device=self.S.device)
        S = self.S[:self.n_slots].view(self.n_slots, n_layers, nv, self.spec.d_k, self.spec.d_v)
        tot = S.sum(0)
        dots = torch.stack([torch.einsum("lhij,lhij->l", S[r], tot) for r in range(self.n_slots)])
        return dots, torch.einsum("lhij,lhij->l", tot, tot)


def segments(labels: torch.Tensor) -> list[tuple[int, int]]:
    """Maximal runs of equal labels: [(start, end), ...]."""
    lab = labels.tolist()
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            out.append((s, i))
            s = i
    return out


@dataclass
class AdvanceResult:
    want_pos: list[int] = field(default_factory=list)   # token offsets (within the call) of ``num``/``den`` rows
    num: torch.Tensor | None = None       # [n_want, n_slots]  sum over heads of <o_r, o>
    den: torch.Tensor | None = None       # [n_want]           sum over heads of |o|^2
    seg_energy: list[float] = field(default_factory=list)   # per segment: energy of its slot right after it
    err: float = 0.0                      # sum |o - o_engine|^2      (check only)
    ref: float = 0.0                      # sum |o_engine|^2


def _chunk(S: torch.Tensor, x: LayerInputs, lab: torch.Tensor, R: int, decay: str, need_out: bool):
    """Advance S[:R] ([R, H, dk, dv], in place) over one chunk. Returns the per-section outputs
    [R, H, c, dv] when ``need_out``."""
    c = x.q.shape[0]
    dev = x.q.device
    qh, kh, vh = x.q.transpose(0, 1), x.k.transpose(0, 1), x.v.transpose(0, 1)      # [H, c, d]
    bh = x.beta.transpose(0, 1)                                                      # [H, c]
    tri = torch.ones(c, c, dtype=torch.bool, device=dev).tril()
    strict = tri & ~torch.eye(c, dtype=torch.bool, device=dev)
    if decay == "head":
        G = x.g.transpose(0, 1).cumsum(-1)                                           # [H, c]
        E = torch.exp((G[:, :, None] - G[:, None, :]).masked_fill(~tri, float("-inf")))
        A = ((kh @ kh.transpose(1, 2)) * E).masked_fill(~strict, 0.0)
        M = (qh @ kh.transpose(1, 2)) * E
        eG = torch.exp(G)[..., None]
        kdec = kh * torch.exp(G[:, -1:] - G)[..., None]
        fade_end = torch.exp(G[:, -1])[:, None, None]                                # [H, 1, 1]
    else:
        G = x.g.transpose(0, 1).cumsum(1)                                            # [H, c, dk]
        E = torch.exp((G[:, :, None, :] - G[:, None, :, :]).masked_fill(~tri[None, :, :, None], float("-inf")))
        A = torch.einsum("htd,hjd,htjd->htj", kh, kh, E).masked_fill(~strict, 0.0)
        M = torch.einsum("htd,hjd,htjd->htj", qh, kh, E)
        eG = torch.exp(G)
        kdec = kh * torch.exp(G[:, -1:, :] - G)
        fade_end = torch.exp(G[:, -1, :])[:, :, None]                                # [H, dk, 1]: rows
    del E
    eye = torch.eye(c, device=dev)
    T = torch.linalg.solve_triangular(eye + bh[..., None] * A, torch.diag_embed(bh),
                                      upper=False, unitriangular=True)               # [H, c, c]
    Sr = S[:R]
    U = -((T @ (kh * eG))[None] @ Sr)                                                # [R, H, c, dv]
    onehot = torch.nn.functional.one_hot(lab, R).to(vh.dtype)                       # [c, R]
    present = torch.nonzero(onehot.sum(0)).flatten()
    if len(present):
        U[present] += torch.einsum("hts,sp,hsd->phtd", T, onehot[:, present], vh)
    out = (qh * eG)[None] @ Sr + M[None] @ U if need_out else None
    Sr.mul_(fade_end[None]).add_(kdec.transpose(1, 2)[None] @ U)
    return out


OutputHook = Callable[[torch.Tensor, torch.Tensor, list[int]], None]


def advance(S: torch.Tensor, x: LayerInputs, labels: torch.Tensor, n_slots: int, decay: str, *,
            want: torch.Tensor | None = None, chunk: int = 64, segs: list[tuple[int, int]] | None = None,
            result: AdvanceResult | None = None, on_output: OutputHook | None = None) -> AdvanceResult:
    """Advance one layer group's per-section memories ``S`` ([>= n_slots, H, dk, dv], in place) over the
    tokens of ``x``, whose section slots are ``labels`` ([T], < n_slots). Chunks never cross a section
    boundary, so the energy of each segment's slot right after the segment is exact.

    ``want`` ([T] bool) marks positions whose per-section outputs are wanted. By default their
    ``num``/``den`` rows (sums over all heads of the group) are added into ``result``; pass the same
    result for every layer group of a call. ``on_output(parts, full, positions)`` replaces that default
    with any other reduction: ``parts`` [R, H, n, d_v] per-section outputs at the wanted rows of a chunk,
    ``full`` [H, n, d_v] their sum, ``positions`` the rows' token offsets. When ``x.ref`` is set, the
    squared error against the engine's output is accumulated as well."""
    res = result or AdvanceResult()
    segs = segs if segs is not None else segments(labels)
    T = labels.shape[0]
    want = want if want is not None else torch.zeros(T, dtype=torch.bool, device=labels.device)
    wpos = torch.nonzero(want).flatten().tolist()
    if res.num is None:
        res.want_pos = wpos
        res.num = torch.zeros(len(wpos), n_slots, device=S.device)
        res.den = torch.zeros(len(wpos), device=S.device)
        res.seg_energy = [0.0] * len(segs)
    row = {p: i for i, p in enumerate(res.want_pos)}
    need_check = x.ref is not None
    for si, (a, b) in enumerate(segs):
        for c0 in range(a, b, chunk):
            c1 = min(c0 + chunk, b)
            sl = slice(c0, c1)
            xc = LayerInputs(q=x.q[sl], k=x.k[sl], v=x.v[sl], g=x.g[sl], beta=x.beta[sl],
                             ref=None if x.ref is None else x.ref[sl])
            w = want[sl]
            need = need_check or bool(w.any())
            out = _chunk(S, xc, labels[sl], n_slots, decay, need)
            if out is None:
                continue
            full = out.sum(0)                                                        # [H, c, dv]
            if need_check:
                d = full - xc.ref.transpose(0, 1)
                res.err += float((d * d).sum())
                res.ref += float((xc.ref * xc.ref).sum())
            if w.any():
                idx = torch.nonzero(w).flatten()
                o = out[:, :, idx]                                                   # [R, H, n, dv]
                f = full[:, idx]                                                     # [H, n, dv]
                if on_output is not None:
                    on_output(o, f, [c0 + int(i) for i in idx])
                    continue
                rows = torch.tensor([row[c0 + int(i)] for i in idx], device=S.device)
                res.num[rows] += torch.einsum("rhnd,hnd->nr", o, f)
                res.den[rows] += (f * f).sum((0, 2))
        r = int(labels[b - 1])
        res.seg_energy[si] += float(torch.linalg.vector_norm(S[r]) ** 2)
    return res


# --------------------------------------------------------------------------- reference

def reference_outputs(S0: torch.Tensor, x: LayerInputs, labels: torch.Tensor, n_slots: int, decay: str):
    """One token at a time, exactly as the engine's kernel: returns (per-section outputs [T, R, H, dv],
    final per-section memories). For tests."""
    S = S0[:n_slots].clone()
    outs = []
    for t in range(x.q.shape[0]):
        fade = torch.exp(x.g[t])                                                     # [H] or [H, dk]
        S = S * (fade[None, :, None, None] if decay == "head" else fade[None, :, :, None])
        kv = torch.einsum("rhij,hi->rhj", S, x.k[t])
        v = torch.zeros(n_slots, *x.v[t].shape, device=S.device)
        v[int(labels[t])] = x.v[t]
        delta = x.beta[t][None, :, None] * (v - kv)
        S = S + x.k[t][None, :, :, None] * delta[:, :, None, :]
        outs.append(torch.einsum("rhij,hi->rhj", S, x.q[t]))
    return torch.stack(outs), S
