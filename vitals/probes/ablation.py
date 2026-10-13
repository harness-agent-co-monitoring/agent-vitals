"""Exact section ablation at decisions, with prefix reuse.

A decision is a position whose next-token distribution chooses token a over token b, with margin
m = z_a - z_b. Replacing a section with equal-length filler and re-decoding everything after it gives the
section's total effect on the decision, Δm = m(original) - m(replaced): every path counts, through later
tokens, the model's own reasoning and the MLPs. Positive Δm means the section pushes toward a.

The original sequence is decoded once, in n_batch chunks, on sequence 0. Each ablation forks it at the chunk
boundary at or before its first replaced position and re-decodes only from there, in the same chunk layout,
so a no-op replacement reproduces the original logits exactly. The backend needs n_seq_max >= 2.

``SectionAblation.effects`` ablates sections one at a time (or as groups); ``localize`` ablates groups first
and then the members of the strongest groups; ``spans_from_labels`` turns per-position section labels into
the token spans to replace.
"""

from __future__ import annotations

import math
import random

import numpy as np

from vitals.core.sections import spans_from_labels  # noqa: F401  (re-exported)

_CONS, _VOW = "bdfgklmnprstvz", "aeiou"


def filler_tokens(engine, n: int, rng: random.Random) -> list[int]:
    """``n`` tokens of pronounceable pseudo-words: no content, ordinary token statistics."""
    if n <= 0:
        return []
    out: list[int] = []
    while len(out) < n:
        words = " ".join("".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(rng.choice([1, 2, 2, 3])))
                         for _ in range(2 * n))
        out += engine.tokenize(" " + words, add_special=False, parse_special=False)
    return out[:n]


def prob(m: float) -> float:
    """Probability of a over b, renormalized to the two tokens."""
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, m))))


def decode_layout(decisions: list[int], nb: int) -> list[tuple[int, int]]:
    """The chunks [start, end) every pass decodes, so a fork at a chunk start re-decodes exactly as the original
    did: ``nb``-token chunks from 0, cut at each decision, and each decision position alone (its logits are read)."""
    chunks, cur = [], 0
    for p in decisions:
        while cur < p:
            e = min(cur + nb, p)
            chunks.append((cur, e))
            cur = e
        chunks.append((p, p + 1))
        cur = p + 1
    return chunks


class SectionAblation:
    """Exact section ablation at one or more decisions of a token sequence (see the module docstring)."""

    def __init__(self, backend, *, seed: int = 0):
        self.eng = backend.engine
        opts = self.eng._ctx_opts
        if int(opts.get("n_seq_max", 1)) < 2:
            raise ValueError("section ablation forks the decoded prefix: load the backend with n_seq_max >= 2")
        self.nb = int(opts["n_batch"])
        self.rng = random.Random(seed)

    def effects(self, tokens: list[int], decisions: dict[str, tuple[int, int, int]],
                ablations: dict[str, list[tuple[int, int]]], replacements: dict[str, list[list[int]]] | None = None) -> dict:
        """``decisions``: name -> (position, a, b). ``ablations``: name -> spans [start, end) replaced together
        (a section, or a group of sections). ``replacements`` optionally gives the replacement tokens per span
        (default: filler). Returns the original margins and, per ablation, the replaced margins and Δm. A decision
        before an ablation's first span keeps its margin (Δm = 0: attention is causal)."""
        toks = list(tokens)
        dec = {k: (int(p), int(a), int(b)) for k, (p, a, b) in decisions.items()}
        last = max(p for p, _, _ in dec.values())
        if last >= len(toks):
            raise ValueError("decision positions must lie inside the token sequence")
        chunks = decode_layout(sorted({p for p, _, _ in dec.values()}), self.nb)
        plan = []
        for name, spans in ablations.items():
            spans = sorted((int(s), int(e)) for s, e in spans)
            if not spans or spans[0][0] < 0 or spans[-1][1] > last:
                raise ValueError(f"ablation {name!r}: spans must lie before the last decision")
            reps = (replacements or {}).get(name) or [filler_tokens(self.eng, e - s, self.rng) for s, e in spans]
            if any(len(r) != e - s for r, (s, e) in zip(reps, spans, strict=True)):
                raise ValueError(f"ablation {name!r}: replacements must keep each span's length")
            fork = max(i for i, (s, _) in enumerate(chunks) if s <= spans[0][0])
            plan.append((fork, name, spans, reps))
        by_pos: dict[int, list[str]] = {}
        for k, (p, _, _) in dec.items():
            by_pos.setdefault(p, []).append(k)

        def run(seq: int, seqtoks: list[int], i0: int, i1: int, out: dict) -> None:
            """Decode chunks [i0, i1) on ``seq``, reading the margins at the decisions among them."""
            for s, e in chunks[i0:i1]:
                self.eng.decode(seqtoks[s:e], pos0=s, seq_id=seq)
                if e - s == 1 and s in by_pos:
                    z = np.asarray(self.eng.logits(-1), dtype=np.float64)
                    for k in by_pos[s]:
                        _, a, b = dec[k]
                        out[k] = float(z[a] - z[b])

        self.eng.reset_context(capture_names=set(), capture_modes={})
        base: dict[str, float] = {}
        replaced, cursor = {}, 0
        for fork, name, spans, reps in sorted(plan, key=lambda x: x[0]):
            run(0, toks, cursor, fork, base)      # the shared prefix, on sequence 0
            cursor = fork
            alt = list(toks)
            for (s, e), r in zip(spans, reps, strict=True):
                alt[s:e] = list(r)
            self.eng.branch_copy(1, 0)
            replaced[name] = {}
            run(1, alt, fork, len(chunks), replaced[name])
            self.eng.branch_drop(1)
        self.eng.branch_copy(1, 0)
        run(1, toks, cursor, len(chunks), base)
        self.eng.branch_drop(1)
        for r in replaced.values():
            for k in dec:
                r.setdefault(k, base[k])          # decided before the fork: unchanged
        return {"base": base, "replaced": replaced,
                "delta": {n: {k: base[k] - r[k] for k in dec} for n, r in replaced.items()}}

    def localize(self, tokens: list[int], decision: tuple[int, int, int], groups: dict[str, dict[str, list[tuple[int, int]]]],
                 *, top_groups: int = 2) -> list[dict]:
        """Ablate each group of sections as a whole, then each member of the ``top_groups`` groups with the
        largest |Δm|. Returns rows (level, group, section, delta), groups first."""
        whole = {g: [sp for spans in members.values() for sp in spans] for g, members in groups.items() if members}
        res = self.effects(tokens, {"d": decision}, whole)
        rows = [{"level": "group", "group": g, "section": None, "delta": res["delta"][g]["d"]} for g in whole]
        top = sorted(whole, key=lambda g: -abs(res["delta"][g]["d"]))[:top_groups]
        members = {s: spans for g in top for s, spans in groups[g].items() if len(groups[g]) > 1}
        if members:
            res2 = self.effects(tokens, {"d": decision}, members)
            owner = {s: g for g in top for s in groups[g]}
            rows += [{"level": "section", "group": owner[s], "section": s, "delta": res2["delta"][s]["d"]} for s in members]
        return rows
