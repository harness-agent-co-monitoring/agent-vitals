"""Reducing captured attention rows to section shares, sparse top-K rows, and per-token logit statistics."""

from __future__ import annotations

import numpy as np

try:
    import torch as _torch
except ImportError:  # the per-row numpy paths stay complete without torch
    _torch = None

SPARSE_K_GEN = 256   # top-K over generated positions: reasoning traces are long


def pool_row(row: np.ndarray, spans, gen_start: int) -> list[float]:
    """Pool one softmax-normalized query row over region token spans + the
    trailing generated bucket -- the live pool_attention convention."""
    shares = [float(row[s:e].sum()) if s is not None and e is not None else 0.0
              for s, e in spans]
    shares.append(float(row[gen_start:].sum()))
    return shares


def layer_head_mean(attn: dict) -> np.ndarray:
    """(T_q, n_kv): head-pooled per layer (uniform mean, or the head-census
    weights when VITALS_HEAD_WEIGHTS is set and the harvest still carries the
    head axis), then layer-mean; 2-D single-row harvests are lifted to one
    query row."""
    from vitals.engine.head_weights import layer_dropped, pool
    kept = {k: a for k, a in attn.items() if not layer_dropped(str(k))} or attn
    rows = np.mean([pool(np.asarray(a, dtype=np.float32), str(k)) for k, a in kept.items()], axis=0)
    return rows[None, :] if rows.ndim == 1 else rows


def logit_stats(logits: np.ndarray, next_token: int, top_k: int) -> dict:
    """Distribution stats of one pre-sampling logits row: entropy, top-2
    margin, surprisal of the forced next token, top-K ids/logprobs."""
    x = logits.astype(np.float64)
    x -= x.max()
    p = np.exp(x)
    p /= p.sum()
    lp = np.log(np.maximum(p, 1e-300))
    idx = np.argpartition(p, -top_k)[-top_k:]
    idx = idx[np.argsort(p[idx])[::-1]]
    return {
        "ent": float(-(p * lp).sum()),
        "m2": float(p[idx[0]] - p[idx[1]]) if len(idx) > 1 else 1.0,
        "surprisal": float(-lp[next_token]),
        "top_ids": [int(i) for i in idx],
        "top_lp": [float(lp[i]) for i in idx],
    }


def sparse_row(prev_att: np.ndarray, n_prompt: int, k: int, step: int, k_gen: int | None = None) -> tuple:
    """One attn_sparse row for a forced token whose attention row is
    ``prev_att`` (prompt positions then generated positions): top-K prompt
    positions + weights, prompt/generated mass totals, and the top-K
    GENERATED positions (as generated-step indices, the axis of
    tokens-*.parquet ``step``) + weights. The generated side is what the
    reasoning-to-answer flow needs: which earlier generated tokens this token
    read, joinable to those tokens' own prompt-side reads."""
    def topk(v, k=k):
        kk = min(k, len(v))
        if kk <= 0:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        top = np.argpartition(v, -kk)[-kk:]
        top = top[np.argsort(v[top])[::-1]]
        top = top[v[top] > 0.0]
        return top.astype(np.int32), v[top].astype(np.float32)
    p, g = prev_att[:n_prompt], prev_att[n_prompt:]
    top, w = topk(p)
    gtop, gw = topk(g, k_gen if k_gen is not None else max(k, SPARSE_K_GEN))
    return (step, top, w, float(p.sum()), float(g.sum()), gtop, gw)


def stats_device() -> str:
    """Where the vectorized logit statistics run: the GPU when torch sees one."""
    return "cuda" if _torch is not None and _torch.cuda.is_available() else "cpu"


def chunk_logit_stats(rows, next_tokens, top_k: int) -> list[dict]:
    """``logit_stats`` for a whole chunk at once: ``rows`` is (n, n_vocab) with
    row i the distribution that sampled ``next_tokens[i]``. Computed in float64
    like the per-row form, in torch (on the GPU when available); without torch
    it is the per-row loop."""
    if _torch is None:
        return [logit_stats(np.asarray(r), int(t), top_k) for r, t in zip(rows, next_tokens, strict=True)]
    x = rows if isinstance(rows, _torch.Tensor) else _torch.from_numpy(np.ascontiguousarray(rows))
    if not x.is_cuda and _torch.cuda.is_available():
        x = x.to("cuda")
    x = x.double()
    n = x.shape[0]
    lp = _torch.log_softmax(x, dim=1)
    p = lp.exp()
    ent = -_torch.special.xlogy(p, p).sum(1)
    toks = _torch.as_tensor([int(t) for t in next_tokens], device=x.device)
    sur = -lp[_torch.arange(n, device=x.device), toks]
    k = min(int(top_k), x.shape[1])
    top = _torch.topk(lp, k, dim=1)                          # sorted, highest first
    tp = top.values.exp()
    m2 = (tp[:, 0] - tp[:, 1]) if k > 1 else _torch.ones_like(ent)
    ent, sur, m2 = ent.tolist(), sur.tolist(), m2.tolist()
    ids, lps = top.indices.tolist(), top.values.tolist()
    return [{"ent": ent[i], "m2": m2[i], "surprisal": sur[i], "top_ids": ids[i], "top_lp": lps[i]}
            for i in range(n)]


def pad_cols(rows, width: int):
    """Right-pad (q, w) rows with zeros to ``width`` columns: the masked-future
    cells of an attention row are exact zeros."""
    w = rows.shape[1]
    return rows if w >= width else _torch.nn.functional.pad(rows, (0, width - w))


def pool_chunk(rows, spans, n_prompt: int, sparse_k: int, step0: int):
    """``pool_row`` and ``sparse_row`` over every row of one chunk at once, on
    the device the rows live on. ``rows`` is a torch (q, W) tensor, row i the
    attention row that sampled forced token ``step0 + i``. Returns (shares,
    recv, sparse): the per-row region shares (region order plus the trailing
    generated bucket), the rows' summed attention over the prompt positions as
    a float64 tensor (the received-attention contribution), and the
    attn_sparse rows in ``sparse_row``'s form. Sums are float64."""
    q, W = rows.shape
    r64 = rows.double()
    zeros = _torch.zeros(q, dtype=_torch.float64, device=rows.device)
    cs = _torch.zeros((q, W + 1), dtype=_torch.float64, device=rows.device)
    cs[:, 1:] = r64.cumsum(1)

    def seg(s, e):
        s, e = min(int(s), W), min(int(e), W)
        return cs[:, e] - cs[:, s] if e > s else zeros

    cols = [seg(s, e) if s is not None and e is not None else zeros for s, e in spans]
    cols.append(seg(n_prompt, W))
    shares = _torch.stack(cols, 1).tolist()
    recv = r64[:, :n_prompt].sum(0)
    sparse: list[tuple] = []
    if sparse_k:
        p, g = rows[:, :n_prompt], rows[:, n_prompt:]
        kk = min(int(sparse_k), p.shape[1])
        kg = min(int(max(sparse_k, SPARSE_K_GEN)), g.shape[1])
        pm, gm = r64[:, :n_prompt].sum(1).tolist(), r64[:, n_prompt:].sum(1).tolist()

        def topk(v, k):
            if k <= 0:
                return np.zeros((q, 0), np.float32), np.zeros((q, 0), np.int64)
            w, i = _torch.topk(v, k, dim=1)                  # sorted, highest first
            return w.cpu().numpy(), i.cpu().numpy()

        pw, pi = topk(p, kk)
        gw, gi = topk(g, kg)
        for i in range(q):
            keep, gkeep = pw[i] > 0.0, gw[i] > 0.0
            sparse.append((step0 + i, pi[i][keep].astype(np.int32), pw[i][keep].astype(np.float32),
                           float(pm[i]), float(gm[i]),
                           gi[i][gkeep].astype(np.int32), gw[i][gkeep].astype(np.float32)))
    return shares, recv, sparse


def take_device_rows(eng):
    """The (q, n_kv) device rows of the last decode when the engine pooled heads
    and layers on the device (the "layer_mean" capture mode), else None."""
    take = getattr(eng, "take_layer_mean", None)
    return take(eng.n_past) if take is not None else None


def logit_rows(eng, n: int) -> np.ndarray:
    """The last decode call's first ``n`` logit rows, (n, n_vocab)."""
    rows = getattr(eng, "logits_rows", None)
    return rows(n) if rows is not None else np.stack([eng.logits(j) for j in range(n)])


