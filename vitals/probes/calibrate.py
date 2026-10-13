"""One-time attention calibrations (per model): retrieval heads, induction
heads and the positional baseline, artifacts consumed by attention-based
signals:

- `retrieval_heads`: needle-in-haystack probes; per-head attention mass on the
  needle span at the answer position identifies the heads that do retrieval
  (vs positional/sink heads). Each probe also records whether the model
  actually retrieves (greedy next token = the code; `answer_acc`), the same
  head's mass on an equal-length control span of filler (`control_mass`), its
  sink mass (positions 0-3) and recency mass (last 8 positions), the heads to
  drop, and a sparsity summary (top-5/top-20 share of all needle mass, heads
  with >10x their control). Filtering span attention to these heads gives a
  higher-SNR attention-to-instruction signal.
- `induction_heads`: a random token sequence repeated twice; per-head attention
  from the second copy to the token after the first occurrence (Olsson et al.)
  ranks induction heads, with the previous-token and duplicate-token patterns
  scored from the same matrices.
- `position_bias`: the same marker span planted at several depths in neutral
  filler; final-position attention share vs depth is the U-shaped baseline
  that turns raw span shares into positional lift:
  lift = share / baseline(depth).

All run fa-off (diagnostic mode) on a fresh engine and emit JSON-ready dicts.
"""

from __future__ import annotations

import numpy as np

_WORDS = ("system logs rotate nightly", "the build cache lives on the scratch disk",
          "deploys happen from the release branch", "metrics land in the shared dashboard",
          "the staging cluster mirrors production", "backups run before every migration",
          "the linter config is checked in", "feature flags default to off",
          "the oncall handbook covers paging", "test fixtures live beside the tests")


def _filler(rng, n_sent: int) -> str:
    return " ".join(f"Note: {_WORDS[int(rng.integers(len(_WORDS)))]}." for _ in range(n_sent))


def _token_span(eng, text: str, sub: str) -> tuple[int, int] | None:
    cs = text.find(sub)
    if cs < 0:
        return None
    toks = eng.tokenize(text)
    offs, cur = [], 0
    for t in toks:
        offs.append(cur)
        cur += len(eng.detokenize([t]))
    offs.append(cur)
    ce = cs + len(sub)
    ts = next((i for i in range(len(toks)) if offs[i + 1] > cs), 0)
    te = next((i + 1 for i in range(len(toks)) if offs[i + 1] >= ce), len(toks))
    return ts, te


def _last_row_attn(eng, prompt: str, names: set[str], want_logits: bool = False):
    """Prefill all but the last token, then decode it alone -- returns each
    captured node's clean single-step attention (heads, kv), and with
    want_logits also the next-token logits of that last position."""
    toks = eng.tokenize(prompt)
    eng.reset_context(capture_names=names, flash_attn=0,
                      capture_modes={n: "last_row" for n in names})
    eng.decode(toks[:-1])
    eng._captured_parts = {}
    eng.decode([toks[-1]])
    cap = eng.collect_captured()
    attn = {n: np.asarray(a, dtype=np.float64).reshape(-1, np.asarray(a).shape[-1])
            for n, a in cap.items() if a is not None}
    if want_logits:
        return attn, eng.logits(-1)
    return attn


def _control_span(rng, ts: int, te: int, n_tok: int) -> tuple[int, int]:
    """An equal-length span of filler away from the needle (before it when
    there is room, else after): the per-head baseline a retrieval head must
    beat."""
    n = te - ts
    lo, hi = 4, ts - n - 4          # keep clear of the sink and of the needle
    if hi > lo:
        s = int(rng.integers(lo, hi))
    else:
        s = int(rng.integers(te + 4, max(te + 5, n_tok - n - 8)))
    return s, s + n


def retrieval_heads(gguf: str, *, layers: list[int] | None = None, n_probes: int = 8,
                    n_ctx: int = 4096, n_filler: int = 60, seed: int = 5,
                    engine_kwargs: dict | None = None) -> dict:
    """Rank (layer, head) pairs by mean attention mass on a planted needle at
    the forced answer position, across ``n_probes`` synthetic prompts."""
    from vitals.engine.binding import LlamaEngine
    from vitals.engine.nodes import for_model, graph_layers

    eng = LlamaEngine(gguf, n_ctx=n_ctx, **(engine_kwargs or {}))
    nm = for_model(eng.arch, graph_layers(eng))
    layers = layers or list(nm.full_attn_layers)
    names = {f"kq_soft_max-{lay}" for lay in layers}
    rng = np.random.default_rng(seed)

    scores: dict[tuple[int, int], list[float]] = {}
    control: dict[tuple[int, int], list[float]] = {}
    sink: dict[tuple[int, int], list[float]] = {}     # mass on positions 0-3 (attention sink)
    local: dict[tuple[int, int], list[float]] = {}    # mass on the last 8 positions (recency)
    delims: dict[tuple[int, int], list[float]] = {}   # mass on punctuation / whitespace tokens (structural)
    answers = []
    for _ in range(n_probes):
        key = f"K{int(rng.integers(100, 999))}"
        val = f"{int(rng.integers(10000, 99999))}"
        needle = f"The access code for locker {key} is {val}."
        pre = _filler(rng, int(n_filler * rng.uniform(0.2, 0.8)))
        post = _filler(rng, n_filler - len(pre.split("Note:")) + 1)
        prompt = (f"{pre} {needle} {post}\n\nQuestion: what is the access code "
                  f"for locker {key}? The access code for locker {key} is ")
        span = _token_span(eng, prompt, val)
        if span is None:
            continue
        ts, te = span
        toks_p = eng.tokenize(prompt)
        n_tok = len(toks_p)
        cs, ce = _control_span(rng, ts, te, n_tok)
        attn, logits = _last_row_attn(eng, prompt, names, want_logits=True)
        delim = np.array([not any(ch.isalnum() for ch in eng.detokenize([t])) for t in toks_p], dtype=bool)
        # the probe is only a retrieval probe if the model retrieves: greedy next token = the code's first token
        first = eng.tokenize(val, add_special=False)[0]
        answers.append(bool(int(np.argmax(logits)) == int(first)))
        for name, a in attn.items():
            layer = int(name.rsplit("-", 1)[1])
            a = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-9)
            for h in range(a.shape[0]):
                scores.setdefault((layer, h), []).append(float(a[h, ts:te].sum()))
                control.setdefault((layer, h), []).append(float(a[h, cs:ce].sum()))
                # captured rows are padded on the kv axis (KV-cache padding): index by n_tok, never from the end
                sink.setdefault((layer, h), []).append(float(a[h, :4].sum()))
                local.setdefault((layer, h), []).append(float(a[h, max(0, n_tok - 8):n_tok].sum()))
                dm = np.zeros(a.shape[1], dtype=bool)
                dm[:min(n_tok, a.shape[1])] = delim[:min(n_tok, a.shape[1])]
                dm[:4] = False                                   # not double-counting the sink
                delims.setdefault((layer, h), []).append(float(a[h, dm].sum()))
    eng.close()

    ranked = sorted(((lay, h, float(np.mean(v)), float(np.mean(control[(lay, h)])))
                     for (lay, h), v in scores.items()), key=lambda x: -x[2])
    masses = np.array([m for _, _, m, _ in ranked]) if ranked else np.zeros(0)
    total = float(masses.sum()) if len(masses) else 0.0
    return {"model": gguf, "layers": layers, "n_probes": n_probes,
            "answer_acc": (float(np.mean(answers)) if answers else None),
            "summary": {"n_heads": int(len(masses)),
                        "top5_share": (float(masses[:5].sum() / total) if total else None),
                        "top20_share": (float(masses[:20].sum() / total) if total else None),
                        "median_needle_mass": (float(np.median(masses)) if len(masses) else None),
                        "n_heads_over_10x_control": int(sum(1 for _, _, m, c in ranked if m > 10 * max(c, 1e-6)))},
            "ranked_heads": [{"layer": lay, "head": h, "needle_mass": round(m, 4),
                              "control_mass": round(c, 4),
                              "sink_mass": round(float(np.mean(sink[(lay, h)])), 4),
                              "local_mass": round(float(np.mean(local[(lay, h)])), 4),
                              "delim_mass": round(float(np.mean(delims[(lay, h)])), 4)}
                             for lay, h, m, c in ranked]}


def position_bias(gguf: str, *, depths: list[float] | None = None, layers: list[int] | None = None,
                  n_ctx: int = 4096, n_filler: int = 80, seed: int = 7,
                  engine_kwargs: dict | None = None) -> dict:
    """Attention share on a fixed marker span vs its fractional depth in
    neutral filler -- the positional baseline for lift normalization."""
    from vitals.engine.binding import LlamaEngine
    from vitals.engine.nodes import for_model, graph_layers

    depths = depths or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    eng = LlamaEngine(gguf, n_ctx=n_ctx, **(engine_kwargs or {}))
    nm = for_model(eng.arch, graph_layers(eng))
    layers = layers or list(nm.full_attn_layers)      # the read map pools all of them
    names = {f"kq_soft_max-{lay}" for lay in layers}
    rng = np.random.default_rng(seed)

    marker = ("Reminder: the quarterly report template changed this cycle; use "
              "the new column order when filling it in.")
    sents = [f"Note: {_WORDS[int(rng.integers(len(_WORDS)))]}." for _ in range(n_filler)]
    curve = []
    for d in depths:
        k = int(d * len(sents))
        body = " ".join(sents[:k] + [marker] + sents[k:])
        prompt = body + "\n\nSummarize the notes above in one sentence: "
        span = _token_span(eng, prompt, marker)
        if span is None:
            continue
        ts, te = span
        shares = []
        for a in _last_row_attn(eng, prompt, names).values():
            a = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-9)
            shares.append(float(a[:, ts:te].sum(axis=1).mean()))
        n_kv = len(eng.tokenize(prompt))
        share = float(np.mean(shares))
        curve.append({"depth": d, "share": round(share, 5),
                      "uniform": round((te - ts) / n_kv, 5),
                      "ratio": round(share / ((te - ts) / n_kv), 3)})
    eng.close()
    if not curve:
        raise ValueError("the marker sentence was not found as a token span at any depth")
    return {"model": gguf, "layers": layers, "marker_tokens": te - ts, "curve": curve}


def induction_heads(gguf: str, *, layers: list[int] | None = None, n_probes: int = 4,
                    seq_len: int = 64, n_ctx: int = 4096, seed: int = 3,
                    engine_kwargs: dict | None = None) -> dict:
    """Rank (layer, head) by the induction score of Olsson et al.: on a random
    token sequence repeated twice, attention from the second copy's token X_i
    to the token that FOLLOWED X_i's first occurrence (position i - n + 1).
    The same full-matrix capture also scores the previous-token pattern
    (position q-1) and the duplicate-token pattern (position q-n), the two
    heads induction is built from. Random ids from the plain-vocabulary range;
    one prefill per probe with full attention matrices captured."""
    from vitals.engine.binding import LlamaEngine
    from vitals.engine.nodes import for_model, graph_layers

    eng = LlamaEngine(gguf, n_ctx=n_ctx, **(engine_kwargs or {}))
    nm = for_model(eng.arch, graph_layers(eng))
    layers = layers or list(nm.full_attn_layers)
    names = {f"kq_soft_max-{lay}" for lay in layers}
    rng = np.random.default_rng(seed)
    ind: dict[tuple[int, int], list[float]] = {}
    prev: dict[tuple[int, int], list[float]] = {}
    dup: dict[tuple[int, int], list[float]] = {}
    selfm: dict[tuple[int, int], list[float]] = {}   # mass on the query position itself
    unif: dict[tuple[int, int], list[float]] = {}    # row entropy / log(row length): 1 = uniform
    lo, hi = 1000, min(30000, eng.n_vocab - 1)
    for _ in range(n_probes):
        seq = [int(x) for x in rng.integers(lo, hi, size=seq_len)]
        toks = [seq[0]] + seq + seq            # a leading token absorbs the sink
        n = seq_len
        eng.reset_context(capture_names=names, flash_attn=0, capture_modes={nm_: "full" for nm_ in names})
        eng.decode(toks)
        cap = eng.collect_captured()
        for name, a in cap.items():
            if a is None:
                continue
            a = np.asarray(a, dtype=np.float64)            # (heads, q, kv)
            layer = int(name.rsplit("-", 1)[1])
            a = a / np.maximum(a.sum(axis=2, keepdims=True), 1e-9)
            qs = np.arange(1 + n + 1, 1 + 2 * n)             # second copy, from its 2nd token
            for h in range(a.shape[0]):
                ind.setdefault((layer, h), []).append(float(a[h, qs, qs - n + 1].mean()))
                prev.setdefault((layer, h), []).append(float(a[h, qs, qs - 1].mean()))
                dup.setdefault((layer, h), []).append(float(a[h, qs, qs - n].mean()))
                selfm.setdefault((layer, h), []).append(float(a[h, qs, qs].mean()))
                rows = a[h, qs, :]
                ent = -(rows * np.log(np.maximum(rows, 1e-12))).sum(axis=1) / np.log(qs + 1)
                unif.setdefault((layer, h), []).append(float(ent.mean()))
    eng.close()
    ranked = sorted(((lay, h, float(np.mean(v)), float(np.mean(prev[(lay, h)])), float(np.mean(dup[(lay, h)])),
                      float(np.mean(selfm[(lay, h)])), float(np.mean(unif[(lay, h)])))
                     for (lay, h), v in ind.items()), key=lambda x: -x[2])
    masses = np.array([m for _, _, m, _, _, _, _ in ranked])
    return {"model": gguf, "layers": layers, "n_probes": n_probes, "seq_len": seq_len,
            "summary": {"n_heads": int(len(masses)),
                        "n_induction_over_0.3": int((masses > 0.3).sum()),
                        "top20_share": (float(masses[:20].sum() / masses.sum()) if masses.sum() else None),
                        "n_prev_token_over_0.5": int(sum(1 for r in ranked if r[3] > 0.5)),
                        "n_self_over_0.5": int(sum(1 for r in ranked if r[5] > 0.5)),
                        "n_uniform_over_0.9": int(sum(1 for r in ranked if r[6] > 0.9))},
            "ranked_heads": [{"layer": lay, "head": h, "induction": round(m, 4),
                              "prev_token": round(pv, 4), "duplicate": round(dv, 4),
                              "self_mass": round(sm, 4), "uniformity": round(un, 4)}
                             for lay, h, m, pv, dv, sm, un in ranked]}
