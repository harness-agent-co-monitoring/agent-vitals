"""Candidate-vs-parent read delta: what one edit did to the worker's reading.

A candidate in an evolutionary loop is an EDIT of an existing design, so its
rendered context on a given example shares most of its text verbatim with the
parent's. Instead of aligning sections by the names each harness chose (which
need not agree), the two prompts are aligned per paired example by a
token-level diff, and every hunk is classed as

  retained  the same tokens in both prompts (paired: read in parent vs child)
  removed   parent-only text (what the edit deleted, and how it was read)
  added     child-only text (what the edit wrote, and how the worker reads it)

Section names annotate hunks (the section covering the hunk in the system that
owns it); they never define them. Lift uses the digest's convention
(`attn_share / length_share` under the turn's attention vector), so the numbers
sit beside the per-section table. The retained class is the control: read the
added text against retained-in-child and the removed text against
retained-in-parent; retained parent -> child is the global drift (context
length, reasoning share).

Outcomes are paired on every episode both systems ran (classification: the
same validation examples; terminal-bench: the same tasks), lifts on the
episodes that carry attention.

Classification pairs episodes by `example_idx` (final call of a multi-call
prediction) and diffs the single prompt. Terminal-bench pairs trials by task,
diffs the first turn's prompt (the scaffold + task description; observations
are trajectory-specific) and pools each hunk's lift over every replayed turn in
which that prefix is still in place.

Parent: an explicit name (the loop passes the candidate's declared
`base_system`), else the store whose prompt shares the most tokens with the
candidate's on a paired example (siblings of the same iteration excluded).

CLI:
  vitals metaharness diff --vitals-dir <dir> --child <system> [--parent <system>]
      [--dataset LawBench] [--exclude sibling ...] [--jobs-dir <harbor jobs>] [--out <md>]
The digests call `delta_block()` and append the result to each candidate's
`vitals_digest.md` as "## Delta vs parent".
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vitals.addons.metaharness.digest.common import store_aggregate, task_stats

MIN_ROW_TOK = 8        # (class, section) rows below this many tokens/episode pool into "(other)"
MAX_PARENT_PROBE = 2   # paired episodes used to score parent candidates


# --------------------------------------------------------------------- loading
def _meta(ep: Path) -> dict:
    try:
        return json.loads((ep / "manifest.json").read_text()).get("meta", {})
    except (OSError, ValueError):
        return {}


def _episodes(store: Path, need_attn: bool) -> dict:
    """{pair_key: episode dir}. Classification keys by example_idx and keeps the
    final call; terminal-bench keys by task (example_idx) and keeps the first
    trial. Restricted to episodes with an attention table when need_attn."""
    out: dict = {}
    for ep in sorted(store.glob("ep-*")):
        if need_attn and not (ep / "attn_tokens.parquet").exists():
            continue
        m = _meta(ep)
        if "example_idx" not in m:
            continue
        key = m["example_idx"]
        call = int(m.get("call_idx", 0))
        prev = out.get(key)
        if prev is None or (isinstance(key, int) and call > prev[1]):
            out[key] = (ep, call, m)
    return {k: (v[0], v[2]) for k, v in out.items()}


def _prompt(ep: Path):
    """(token ids of the first user message, {turn: attention vec}, {turn: L},
    spans [(name, s, e)] of turn 0, end of the user message). Attention vectors
    are the digest's per-position vectors (attn_tokens.attn)."""
    t = pd.read_parquet(ep / "attn_tokens.parquet", columns=["turn", "idx", "token_id", "attn"])
    regs = {}
    for line in (ep / "regions.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
            regs[int(r["turn"])] = r["regions"]
        except (ValueError, KeyError):
            continue
    t0 = int(t.turn.min())
    r0 = regs.get(t0, [])
    end = next((int(r["token_end"]) for r in r0 if r.get("id") == f"user_{t0}"), None)
    first = t[t.turn == t0].sort_values("idx")
    ids = first.token_id.to_numpy()
    if end is None:
        end = len(ids)
    spans = [(str(r["id"]).removeprefix("span:"), int(r["token_start"]), int(r["token_end"]))
             for r in r0 if str(r.get("id", "")).startswith("span:")
             and r.get("token_start") is not None and r.get("token_end") is not None]
    vecs, Ls = {}, {}
    for turn, g in t.groupby("turn"):
        g = g.sort_values("idx")
        if int(turn) != t0 and not np.array_equal(g.token_id.to_numpy()[:end], ids[:end]):
            continue          # the scaffold prefix was rewritten at this turn: not comparable
        vecs[int(turn)] = g.attn.to_numpy(dtype=float)
        Ls[int(turn)] = len(g)
    return ids[:end], vecs, Ls, spans, end


def _base(vecs: dict, Ls: dict, end: int) -> float:
    """Mean digest-lift of the diffed prompt itself (the `rel` denominator:
    1 = read like the average of this prompt, whatever its length)."""
    vals = [float(v[:end].mean()) * Ls[t] / (float(v.sum()) or 1.0) for t, v in vecs.items()]
    return float(np.mean(vals)) if vals else float("nan")


def _section_of(spans, s: int, e: int) -> str:
    """Declared span covering most of [s, e) (ties: the smallest span), else '(unnamed)'."""
    best, cover, size = "(unnamed)", 0, 10 ** 9
    for name, a, b in spans:
        ov = min(b, e) - max(a, s)
        if ov > cover or (ov == cover and ov > 0 and (b - a) < size):
            best, cover, size = name, ov, b - a
    return best if cover > 0 else "(unnamed)"


def _family(name: str) -> str:
    """Per-label content sections (`demo:<label>#<k>`, `hallmarks:<label>`) pool by
    prefix; scaffold names (`directive:k`, `fb:<name>`, `obs...`) stay themselves."""
    if ":" in name and not name.startswith(("directive:", "span:", "fb:", "obs")):
        return name.split(":", 1)[0] + ":*"
    return name


def _lift(vecs: dict, Ls: dict, s: int, e: int) -> float:
    """Digest-convention lift of [s, e), mean over the turns it is in place
    (attention renormalized over the context: / its share of the decode)."""
    vals = []
    for turn, v in vecs.items():
        if e <= len(v):
            vals.append(float(v[s:e].mean()) * Ls[turn] / (float(v.sum()) or 1.0))
    return float(np.mean(vals)) if vals else float("nan")


def _mass(vecs: dict, s: int, e: int) -> float:
    vals = [float(v[s:e].sum()) for v in vecs.values() if e <= len(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ------------------------------------------------------------------ the diff
def pair_hunks(ep_p: Path, ep_c: Path, key) -> tuple[list[dict], dict]:
    """Hunk rows of one paired episode plus its context line."""
    ids_p, vec_p, L_p, sp_p, end_p = _prompt(ep_p)
    ids_c, vec_c, L_c, sp_c, end_c = _prompt(ep_c)
    base = {"parent": _base(vec_p, L_p, end_p), "child": _base(vec_c, L_c, end_c)}
    sm = difflib.SequenceMatcher(None, ids_p.tolist(), ids_c.tolist(), autojunk=False)
    rows = []

    def add(cls, side, ids, vecs, Ls, spans, s, e, twin=None):
        sec = _section_of(spans, s, e)
        lift = _lift(vecs, Ls, s, e)
        rows.append({"example": key, "class": cls, "side": side,
                     "section": sec, "family": _family(sec), "n": e - s,
                     "pos": (s + e) / 2 / max(1, len(ids)),
                     "mass": _mass(vecs, s, e), "lift": lift,
                     "rel": lift / base[side] if base[side] else float("nan"),
                     "twin": twin})

    for k, (tag, i1, i2, j1, j2) in enumerate(sm.get_opcodes()):
        if tag == "equal":
            add("retained", "parent", ids_p, vec_p, L_p, sp_p, i1, i2, twin=k)
            add("retained", "child", ids_c, vec_c, L_c, sp_c, j1, j2, twin=k)
        else:
            if i2 > i1:
                add("removed", "parent", ids_p, vec_p, L_p, sp_p, i1, i2)
            if j2 > j1:
                add("added", "child", ids_c, vec_c, L_c, sp_c, j1, j2)
    ctx = {"example": key, "tok_p": len(ids_p), "tok_c": len(ids_c),
           "L_p": float(np.mean(list(L_p.values()))), "L_c": float(np.mean(list(L_c.values()))),
           "pmass_p": float(np.mean([v.sum() for v in vec_p.values()])),
           "pmass_c": float(np.mean([v.sum() for v in vec_c.values()])),
           "shared_c": sum(j2 - j1 for t, _, _, j1, j2 in sm.get_opcodes() if t == "equal") / max(1, len(ids_c)),
           "turns_p": len(vec_p), "turns_c": len(vec_c)}
    return rows, ctx


def _overlap(ep_a: Path, ep_b: Path) -> float:
    ia, *_ = _prompt(ep_a)
    ib, *_ = _prompt(ep_b)
    sm = difflib.SequenceMatcher(None, ia.tolist(), ib.tolist(), autojunk=False)
    return sum(k for _, _, k in sm.get_matching_blocks()) / max(1, len(ia))


def _name_in(hint: str | None, names: list[str]) -> str | None:
    """First system name mentioned in a free-text `base_system` field."""
    if not hint:
        return None
    for tok in re.findall(r"[A-Za-z0-9_]+", hint):
        if tok in names:
            return tok
    return None


def resolve_parent(vitals_dir: Path, child: str, dataset: str, exclude=(), hint: str | None = None):
    """(parent system, share of the child's prompt tokens found verbatim in it,
    'declared'|'overlap'|None)."""
    stores = {d.name.rsplit("--", 1)[0]: d / "store" for d in vitals_dir.glob(f"*--{dataset}")
              if (d / "store").is_dir()}
    names = [n for n in stores if n != child and n not in set(exclude)]
    if not names:
        return None, 0.0, None
    child_eps = _episodes(stores[child], need_attn=True) if child in stores else {}
    declared = _name_in(hint, names)
    if declared:
        parent_eps = _episodes(stores[declared], need_attn=True)
        common = [k for k in child_eps if k in parent_eps][:1]
        ov = _overlap(child_eps[common[0]][0], parent_eps[common[0]][0]) if common else float("nan")
        return declared, ov, "declared"
    # cheap 8-gram containment ranks the candidates; the exact diff runs on the top few
    def grams(ep):
        ids = _prompt(ep)[0].tolist()
        return {tuple(ids[i:i + 8]) for i in range(0, max(0, len(ids) - 7))}
    ranked = []
    child_grams = {k: grams(v[0]) for k, v in list(child_eps.items())[:MAX_PARENT_PROBE]}
    cand_eps = {}
    for n in names:
        peps = _episodes(stores[n], need_attn=True)
        common = [k for k in child_grams if k in peps]
        if not common:
            continue
        cand_eps[n] = (peps, common)
        c = float(np.mean([len(child_grams[k] & grams(peps[k][0])) / max(1, len(child_grams[k])) for k in common]))
        ranked.append((c, n))
    scored = []
    for _, n in sorted(ranked, reverse=True)[:3]:
        peps, common = cand_eps[n]
        scored.append((float(np.mean([_overlap(child_eps[k][0], peps[k][0]) for k in common])), n))
    scored.sort(reverse=True)
    if not scored:
        return None, 0.0, None
    alts = "; ".join(f"`{n}` {ov:.0%}" for ov, n in scored[1:])
    return scored[0][1], scored[0][0], "overlap" + (f" (next: {alts})" if alts else "")


# ------------------------------------------------------------- the report
def _fmt(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _k(x):
    return f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}"


def system_delta(vitals_dir: Path, child: str, parent: str, dataset: str,
                 jobs_dir: Path | None = None) -> tuple[str, pd.DataFrame]:
    """Markdown block + hunk rows for child vs parent on one dataset."""
    tb = dataset == "tb2"
    sc, sp = vitals_dir / f"{child}--{dataset}" / "store", vitals_dir / f"{parent}--{dataset}" / "store"
    ec_all, ep_all = _episodes(sc, False), _episodes(sp, False)
    ec, epp = _episodes(sc, True), _episodes(sp, True)
    common = [k for k in ec if k in epp]
    if not common:
        return f"## Delta vs parent (`{parent}`)\n\n(no paired episodes with attention)\n", pd.DataFrame()
    rows, ctxs = [], []
    for k in common:
        r, c = pair_hunks(epp[k][0], ec[k][0], k)
        rows.extend(r)
        ctxs.append(c)
    H = pd.DataFrame(rows)
    C = pd.DataFrame(ctxs)
    n = len(common)

    # class summary: tokens/ep, share of the owning prompt, lift
    def cls_stats(cls, side):
        h = H[(H["class"] == cls) & (H.side == side)]
        tok = h.n.sum() / n
        tot = C.tok_c.mean() if side == "child" else C.tok_p.mean()
        lift = (h.n * h.lift).sum() / h.n.sum() if h.n.sum() else float("nan")
        rel = (h.n * h.rel).sum() / h.n.sum() if h.n.sum() else float("nan")
        return tok, tok / max(1, tot), lift, rel
    rp, rc = cls_stats("retained", "parent"), cls_stats("retained", "child")
    rm, ad = cls_stats("removed", "parent"), cls_stats("added", "child")
    out = [f"## Delta vs parent (`{parent}`)",
           f"Token-level diff of the rendered prompt against `{parent}` on {n} paired "
           f"{'tasks' if tb else 'examples'} with attention"
           f"{' (first turn; lifts pooled over the replayed turns that keep the prefix)' if tb else ''}: "
           f"{C.shared_c.mean():.0%} of this design's prompt is verbatim in the parent. "
           f"Prompt tokens {_k(C.tok_p.mean())} → {_k(C.tok_c.mean())}"
           + (f"; context per turn {_k(C.L_p.mean())} → {_k(C.L_c.mean())}" if tb else "")
           + f"; share of attention on the prompt (vs the worker's own output) "
           f"{C.pmass_p.mean():.2f} → {C.pmass_c.mean():.2f}.\n",
           "| text | tokens/ep | share of prompt | lift in parent | lift in child | rel parent → child |",
           "|---|---|---|---|---|---|",
           f"| retained (same tokens in both) | {_k(rc[0])} | {rp[1]:.2f} → {rc[1]:.2f} | {_fmt(rp[2])} | {_fmt(rc[2])} | {_fmt(rp[3])} → {_fmt(rc[3])} |",
           f"| removed (parent only) | {_k(rm[0])} | {rm[1]:.2f} | {_fmt(rm[2])} | – | {_fmt(rm[3])} → – |",
           f"| added (child only) | {_k(ad[0])} | {ad[1]:.2f} | – | {_fmt(ad[2])} | – → {_fmt(ad[3])} |",
           "",
           "`lift` is the digest's convention (it grows with context length, so it is not comparable "
           "across the two designs when their contexts differ); `rel` is the same text's lift over the "
           "mean lift of its own prompt (1 = read like this prompt's average), comparable across designs. "
           "Read `added` against `retained` in the child and `removed` against `retained` in the parent; "
           "retained parent → child is the drift the edit did not cause.\n"]

    # per-section rows for added / removed
    def by_section(cls, side, ref_lift):
        h = H[(H["class"] == cls) & (H.side == side)]
        if h.empty:
            return "(none)"
        g = (h.groupby("family").apply(lambda x: pd.Series({
            "tok": x.n.sum() / n, "lift": (x.n * x.lift).sum() / x.n.sum() if x.n.sum() else np.nan,
            "eps": x.example.nunique(), "k": x.section.nunique()}), include_groups=False).reset_index())
        big, small = g[g.tok >= MIN_ROW_TOK].sort_values("tok", ascending=False), g[g.tok < MIN_ROW_TOK]
        lines = ["| section | tokens/ep | lift | vs retained | in eps |", "|---|---|---|---|---|"]
        for r in big.itertuples():
            nm = r.family[:40] + (f" ({int(r.k)} sections)" if r.family.endswith(":*") else "")
            lines.append(f"| {nm} | {_k(r.tok)} | {_fmt(r.lift)} | "
                         f"{_fmt(r.lift / ref_lift if ref_lift else np.nan)}x | {int(r.eps)}/{n} |")
        if len(small):
            hs = h[h.family.isin(small.family)]
            lines.append(f"| (other, {int(hs.section.nunique())} sections) | {_k(hs.n.sum() / n)} | "
                         f"{_fmt((hs.n * hs.lift).sum() / hs.n.sum() if hs.n.sum() else np.nan)} | – | – |")
        return "\n".join(lines)
    out.append("**Added text by section** (this design's section names):\n\n" + by_section("added", "child", rc[2]) + "\n")
    out.append("**Removed text by section** (the parent's section names):\n\n" + by_section("removed", "parent", rp[2]) + "\n")

    # retained text, paired per section (child's names): lift p -> c on the same tokens
    ret = H[H["class"] == "retained"]
    pr = ret[ret.side == "parent"].set_index(["example", "twin"])
    cr = ret[ret.side == "child"].set_index(["example", "twin"])
    j = cr.join(pr[["lift", "rel", "mass", "pos", "section"]], rsuffix="_p")
    if len(j):
        j = j.reset_index()
        g = (j.groupby("family").apply(lambda x: pd.Series({
            "tok": x.n.sum() / n, "k": x.section.nunique(),
            "lift_p": (x.n * x.rel_p).sum() / x.n.sum(), "lift_c": (x.n * x.rel).sum() / x.n.sum(),
            "pos_p": (x.n * x.pos_p).sum() / x.n.sum(), "pos_c": (x.n * x.pos).sum() / x.n.sum(),
            "up": sum(1 for _, y in x.groupby("example")
                      if (y.n * y.rel).sum() > (y.n * y.rel_p).sum()),
            "eps": x.example.nunique()}), include_groups=False).reset_index())
        g = g[g.tok >= MIN_ROW_TOK].sort_values("tok", ascending=False)
        lines = ["| section | tokens/ep | rel parent → child | Δ | eps read harder | position |", "|---|---|---|---|---|---|"]
        for r in g.itertuples():
            nm = r.family[:40] + (f" ({int(r.k)} sections)" if r.family.endswith(":*") else "")
            lines.append(f"| {nm} | {_k(r.tok)} | {_fmt(r.lift_p)} → {_fmt(r.lift_c)} | "
                         f"{r.lift_c - r.lift_p:+.2f} | {int(r.up)}/{int(r.eps)} | {r.pos_p:.2f} → {r.pos_c:.2f} |")
        out.append("**Retained text by section** (same tokens in both prompts, `rel` lift; `position` = "
                   "fraction of the prompt where the text sits):\n\n" + "\n".join(lines) + "\n")

    # section-level digest numbers side by side (names shared / new / gone)
    ap_, ac_ = _section_agg(sp), _section_agg(sc)
    if ap_ is not None and ac_ is not None:
        both = sorted(sorted(set(ap_.index) & set(ac_.index)), key=lambda s: -ac_.footprint_tok.get(s, 0))
        new = sorted(sorted(set(ac_.index) - set(ap_.index)), key=lambda s: -ac_.footprint_tok.get(s, 0))
        gone = sorted(sorted(set(ap_.index) - set(ac_.index)), key=lambda s: -ap_.footprint_tok.get(s, 0))
        def named(xs):
            return [s for s in xs if not (":" in s and not s.startswith(("directive:", "span:", "fb:", "obs")))]
        lines = ["| section | footprint parent → child | lift | consult ratio |", "|---|---|---|---|"]
        for s in named(both)[:16]:
            a, b = ap_.loc[s], ac_.loc[s]
            lines.append(f"| {s[:40]} | {int(a.footprint_tok)} → {int(b.footprint_tok)} | {a.lift:.2f} → {b.lift:.2f} | "
                         f"{_fmt(a.consult_ratio)} → {_fmt(b.consult_ratio)} |")
        for s in named(new)[:8]:
            b = ac_.loc[s]
            lines.append(f"| {s[:40]} (new) | – → {int(b.footprint_tok)} | – → {b.lift:.2f} | – → {_fmt(b.consult_ratio)} |")
        for s in named(gone)[:8]:
            a = ap_.loc[s]
            lines.append(f"| {s[:40]} (removed) | {int(a.footprint_tok)} → – | {a.lift:.2f} → – | {_fmt(a.consult_ratio)} → – |")
        out.append("**Sections by name** (the digest's per-section numbers, where both designs declare the "
                   "same name; `(new)`/`(removed)` are names only one declares):\n\n" + "\n".join(lines) + "\n")

    # outcomes: every paired episode; on the attention panel, the flipped ones with their read
    out.append(_outcomes(ec_all, ep_all, H, common, tb, jobs_dir, child, parent))
    H.insert(0, "parent", parent)
    H.insert(0, "child", child)
    return "\n".join(out) + "\n", H


def _section_agg(store: Path):
    """The digest's per-section aggregate of a store (built and cached by the
    digest; built here when a store was never digested)."""
    try:
        df = store_aggregate(store)
    except Exception:
        return None
    if df is None or "section" not in df:
        return None
    return df.drop_duplicates("section").set_index("section")


def _outcomes(ec_all, ep_all, H, panel, tb, jobs_dir, child, parent) -> str:
    keys = [k for k in ec_all if k in ep_all]
    if tb and jobs_dir is not None:
        try:
            sc, spar = task_stats(jobs_dir, child), task_stats(jobs_dir, parent)
            tasks = sorted(set(sc) & set(spar))
            if tasks:
                lines = [f"**Outcomes** on {len(tasks)} tasks both ran (all trials): parent "
                         f"{np.mean([np.mean(spar[t]['ok']) for t in tasks]):.0%} → child "
                         f"{np.mean([np.mean(sc[t]['ok']) for t in tasks]):.0%} pass; timeouts "
                         f"{sum(spar[t]['timeouts'] for t in tasks)} → {sum(sc[t]['timeouts'] for t in tasks)}.",
                         "", "| task | parent pass, turns, secs | child pass, turns, secs | added text rel | retained rel p → c |",
                         "|---|---|---|---|---|"]
                for t in tasks:
                    a, b = spar[t], sc[t]
                    lines.append(f"| {t[:32]} | {np.mean(a['ok']):.0%}, {np.mean(a['turns']) if a['turns'] else float('nan'):.0f}t, "
                                 f"{np.mean(a['secs']) if a['secs'] else float('nan'):.0f}s | "
                                 f"{np.mean(b['ok']):.0%}, {np.mean(b['turns']) if b['turns'] else float('nan'):.0f}t, "
                                 f"{np.mean(b['secs']) if b['secs'] else float('nan'):.0f}s | "
                                 f"{_read(H, t, 'added', 'child')} | {_read(H, t, 'retained', 'parent')} → {_read(H, t, 'retained', 'child')} |")
                return "\n".join(lines) + "\n"
        except Exception:
            pass
    rp = {k: float(ep_all[k][1].get("reward", 0) or 0) for k in keys}
    rc = {k: float(ec_all[k][1].get("reward", 0) or 0) for k in keys}
    won = [k for k in keys if rc[k] > rp[k]]
    lost = [k for k in keys if rc[k] < rp[k]]
    unit = "tasks" if tb else "examples"
    lines = [f"**Outcomes** on {len(keys)} paired {unit}: parent {sum(v > 0 for v in rp.values())}/{len(keys)} → "
             f"child {sum(v > 0 for v in rc.values())}/{len(keys)}; ✗→✓ {len(won)}"
             f"{' (' + ', '.join(str(k) for k in won[:12]) + ')' if won else ''}; ✓→✗ {len(lost)}"
             f"{' (' + ', '.join(str(k) for k in lost[:12]) + ')' if lost else ''}."]
    flipped = [k for k in panel if k in won or k in lost]
    if flipped:
        lines += ["", "On the attention panel, the flipped " + unit + ":", "",
                  "| " + unit[:-1] + " | flip | added text rel | removed text rel | retained rel parent → child |",
                  "|---|---|---|---|---|"]
        for k in flipped:
            lines.append(f"| {k} | {'✗→✓' if k in won else '✓→✗'} | {_read(H, k, 'added', 'child')} | "
                         f"{_read(H, k, 'removed', 'parent')} | {_read(H, k, 'retained', 'parent')} → {_read(H, k, 'retained', 'child')} |")
    return "\n".join(lines) + "\n"


def _read(H, key, cls, side) -> str:
    """Pooled `rel` lift of one class on one episode."""
    h = H[(H.example == key) & (H["class"] == cls) & (H.side == side)]
    return _fmt((h.n * h.rel).sum() / h.n.sum()) if h.n.sum() else "–"


LEGEND = ("**How to read the delta.** The candidate's rendered prompt is diffed token by token against the "
          "parent's on the same example; `retained` text is read in both (its parent → child change is the "
          "drift the edit did not cause), `removed` is what the edit deleted (read as the parent read it), "
          "`added` is what the edit wrote (read as this design reads it). `vs retained` = the text's lift over "
          "the retained text's lift in the same prompt: below 1 the edit's text is under-read relative to what "
          "surrounds it. Correlational, as everywhere in the digest.\n")


def delta_block(vitals_dir: Path, child: str, dataset: str, exclude=(), hint: str | None = None,
                jobs_dir: Path | None = None, parent: str | None = None):
    """Resolve the parent and build the block. A `base_system` hint that names
    an existing store wins; else the given `parent`; else the closest design by
    prompt overlap. Returns (markdown, hunks df, parent)."""
    how, ov = "given", float("nan")
    if hint:
        p2, ov2, how2 = resolve_parent(vitals_dir, child, dataset, exclude=exclude, hint=hint)
        if how2 == "declared":
            parent, ov, how = p2, ov2, how2
    if parent is None:
        parent, ov, how = resolve_parent(vitals_dir, child, dataset, exclude=exclude)
    if parent is None or parent == child:
        return "", pd.DataFrame(), None
    md, H = system_delta(vitals_dir, child, parent, dataset, jobs_dir=jobs_dir)
    head = (f"Parent `{parent}` " + ("as declared by the proposer" if how == "declared" else "as given" if how == "given"
            else f"by prompt overlap, the closest existing design at {ov:.0%}" + how.removeprefix("overlap")) + ".\n\n")
    md = md.replace("\n", "\n\n" + head, 1) if md.startswith("## ") else head + md
    return md + "\n" + LEGEND, H, parent


def parse_hints(items) -> dict:
    """`name=base_system text` CLI items -> {name: text}."""
    out = {}
    for it in items or []:
        if "=" in it:
            k, v = it.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def demote(md: str) -> str:
    """`## ` headings -> `### ` (the block inside a consolidated report)."""
    return "\n".join(("#" + line if line.startswith("## ") else line) for line in md.splitlines())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vitals-dir", type=Path, required=True)
    ap.add_argument("--child", required=True)
    ap.add_argument("--parent", default=None, help="parent system (default: declared hint, else closest by prompt overlap)")
    ap.add_argument("--hint", default=None, help="free-text base_system field to resolve the parent from")
    ap.add_argument("--dataset", default=None, help="dataset suffix of the stores (default: every dataset the child has)")
    ap.add_argument("--exclude", nargs="*", default=[], help="systems never taken as parent (the iteration's siblings)")
    ap.add_argument("--jobs-dir", type=Path, default=None, help="terminal-bench: harbor jobs dir for all-trial outcomes")
    ap.add_argument("--out", type=Path, default=None, help="write the markdown here (default: stdout)")
    ap.add_argument("--hunks", type=Path, default=None, help="write the hunk rows as parquet")
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else sorted(
        d.name.rsplit("--", 1)[1] for d in args.vitals_dir.glob(f"{args.child}--*") if (d / "store").is_dir())
    blocks, hunks = [], []
    for ds in datasets:
        md, H, parent = delta_block(args.vitals_dir, args.child, ds, exclude=args.exclude, hint=args.hint,
                                    jobs_dir=args.jobs_dir, parent=args.parent)
        if not md:
            blocks.append(f"## {args.child} / {ds}\n\n(no parent candidate)\n")
            continue
        blocks.append(f"# {args.child} / {ds}\n\n" + md)
        if len(H):
            hunks.append(H.assign(dataset=ds))
    text = "\n".join(blocks)
    if args.out:
        args.out.write_text(text)
        print(f"delta: wrote {args.out}")
    else:
        print(text)
    if args.hunks and hunks:
        pd.concat(hunks, ignore_index=True).to_parquet(args.hunks, index=False)


if __name__ == "__main__":
    main()
