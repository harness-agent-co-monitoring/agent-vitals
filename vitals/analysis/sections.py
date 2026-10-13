"""Section-level outcome diagnostics across one or more replayed stores.

Three outputs over one long table:

  section_table(roots)          episode x section rows with reading stats and
                                the manifest's outcome fields
  outcome_contrast(table)       per named section: errors-vs-correct contrast
                                of within-design z-scored read intensity
  answer_relation(table)        per-label sections split by their relation to
                                the episode's answer (predicted / true /
                                other): the answer-faithfulness gradient

Conventions: reading intensity is the span read lift (per-token attention vs
uniform; see ``reading.episode_turns``). z-scoring is within (design, dataset,
section) so contrasts are within-design and never smuggle in between-design
differences. Outcomes come from manifest meta (reward / prediction / target,
as the ingest stamps them). Attention is correlational: these are markers of
failure modes, not causes.

CLI: ``vitals sections <replay-root-or-store>``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from vitals.core.store import episode_dirs


def _welch_p(a, b) -> tuple[float, float]:
    """Welch t and two-sided p (normal approximation; fine at these n)."""
    import numpy as np

    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb + 1e-300)
    p = math.erfc(abs(t) / math.sqrt(2))
    return float(t), float(p)


def _norm(s) -> str:
    return str(s).lower().strip()


def _stores_under(root: Path) -> list[Path]:
    root = Path(root)
    if episode_dirs(root):
        return [root]
    out = [d for d in sorted(root.glob("*/store")) if episode_dirs(d)]
    return out or [d for d in sorted(root.glob("*")) if d.is_dir() and episode_dirs(d)]


def section_table(roots):
    """Episode x section long table across one or more stores.

    Columns: episode_id, design, dataset, section, lift, W, mass_share,
    footprint_tok, L, reward, prediction, target. design/dataset come from
    manifest meta (memory/task) with the store name as fallback.
    """
    import pandas as pd

    from vitals.core.store import read_regions

    from .reading import episode_turns

    if isinstance(roots, (str, Path)):
        roots = [roots]
    rows = []
    for root in roots:
        for store in _stores_under(Path(root)):
            fallback = store.parent.name
            for ep in episode_dirs(store):
                man = ep / "manifest.json"
                try:
                    meta = json.loads(man.read_text()).get("meta") or {}
                except (OSError, json.JSONDecodeError):
                    meta = {}
                turns = episode_turns(ep)
                if not turns:
                    continue
                foot = {}
                for _t, regs in read_regions(ep).items():
                    for r in regs:
                        rid = str(r.get("id", ""))
                        s, e = r.get("token_start"), r.get("token_end")
                        if rid.startswith("span:") and s is not None and e is not None:
                            foot[rid.removeprefix("span:")] = int(e) - int(s)
                for tr in turns:
                    for name, d in tr["sections"].items():
                        rows.append({
                            "episode_id": ep.name,
                            "design": meta.get("memory") or fallback,
                            "dataset": meta.get("task") or "",
                            "section": name,
                            "lift": d["lift"], "W": d["W"],
                            "footprint_tok": foot.get(name, 0),
                            "mass_share": d["W"] * foot.get(name, 0),
                            "L": tr["L"],
                            "reward": meta.get("reward"),
                            "prediction": meta.get("prediction"),
                            "target": meta.get("target"),
                        })
    return pd.DataFrame(rows)


def _zlift(df):
    g = df.groupby(["design", "dataset", "section"]).lift
    return (df.lift - g.transform("mean")) / g.transform("std").replace(0, float("nan"))


def outcome_contrast(table, min_errors: int = 15, min_correct: int = 30):
    """Per named section: errors-vs-correct contrast of within-design z-lift.

    Positive delta = errors read the section MORE than successes. Returns a
    frame sorted by p with columns dataset, section, delta, p, n_err, n_corr,
    n_designs. Sections carrying a per-label suffix are best analyzed with
    answer_relation instead, but are included here by full name.
    """
    import pandas as pd

    columns = ["dataset", "section", "delta", "t", "p", "n_err", "n_corr", "n_designs"]
    df = table.dropna(subset=["reward"]).copy()
    df["zl"] = _zlift(df)
    df = df.dropna(subset=["zl"])
    out = []
    for (ds, name), g in df.groupby(["dataset", "section"]):
        e = g[g.reward == 0.0].zl
        c = g[g.reward == 1.0].zl
        if len(e) < min_errors or len(c) < min_correct:
            continue
        t, p = _welch_p(e, c)
        out.append({"dataset": ds, "section": name,
                    "delta": round(float(e.mean() - c.mean()), 4),
                    "t": round(t, 2), "p": p,
                    "n_err": len(e), "n_corr": len(c),
                    "n_designs": g.design.nunique()})
    return pd.DataFrame(out, columns=columns).sort_values("p").reset_index(drop=True)


def answer_relation(table, normalize=None, match: str = "exact"):
    """The answer-faithfulness gradient over per-label sections.

    Sections whose name carries a ':<label>' suffix are classified by the
    label's relation to the episode's answer -- pred(=true) / predicted /
    true / other -- and the mean within-design z-lift of each group is
    reported for correct and error episodes separately. ``normalize`` is an
    optional str->str callable applied to labels/predictions/targets (task
    prefixes etc.); ``match`` is "exact" or "contains".
    """
    import pandas as pd

    norm = normalize or _norm
    df = table.dropna(subset=["reward"]).copy()
    df = df[df.section.str.contains(":")]
    if not len(df):
        return pd.DataFrame()
    df["zl"] = _zlift(df)
    df = df.dropna(subset=["zl"])
    df["label"] = df.section.str.split(":", n=1).str[1].map(norm)
    df["pred"] = df.prediction.map(norm)
    df["targ"] = df.target.map(norm)
    def hit(a, b) -> bool:
        if match == "contains":
            return bool(a) and bool(b) and (a in b or b in a)
        return bool(a) and a == b

    isp = [hit(lab, p) for lab, p in zip(df.label, df.pred, strict=True)]
    ist = [hit(lab, t) for lab, t in zip(df.label, df.targ, strict=True)]
    df["relation"] = ["pred(=true)" if a and b else "predicted" if a
                      else "true" if b else "other"
                      for a, b in zip(isp, ist, strict=True)]
    out = []
    for (ds, rel), g in df.groupby(["dataset", "relation"]):
        for outcome, gg in (("correct", g[g.reward == 1.0]),
                            ("error", g[g.reward == 0.0])):
            if len(gg) > 20:
                out.append({"dataset": ds, "relation": rel,
                            "episodes_of": outcome,
                            "mean_zlift": round(float(gg.zl.mean()), 3),
                            "n": len(gg)})
    return pd.DataFrame(out)


def report(root, min_errors: int = 15, min_correct: int = 30,
           top: int = 12, match: str = "exact") -> str:
    """Human-readable section diagnostic for a store or replay root."""
    table = section_table(root)
    if not len(table):
        return f"no replayed section data under {root}"
    lines = [f"section diagnostic: {table.episode_id.nunique()} episodes, "
             f"{table.design.nunique()} designs, "
             f"{table.section.nunique()} named sections"]
    oc = outcome_contrast(table, min_errors, min_correct)
    # per-label sections (suffix matches an answer somewhere in the corpus)
    # belong to the gradient below: their raw contrasts confound with label
    # base rates, so keep the contrast list to structural sections
    answers = {a for a in (set(table.prediction.dropna().map(_norm))
                           | set(table.target.dropna().map(_norm))) if a}

    def _sfx_is_answer(name: str) -> bool:
        _, _, sfx = name.partition(":")
        sfx = _norm(sfx)
        return len(sfx) >= 2 and any(sfx in a or a in sfx for a in answers)

    # a FAMILY (prefix before ':') is per-label if any of its suffixes matches
    # an answer; then every member is (labels outside the answer set too)
    label_fams = {n.partition(":")[0] for n in oc.section
                  if ":" in n and _sfx_is_answer(n)}
    oc = oc[~oc.section.map(
        lambda n: ":" in n and n.partition(":")[0] in label_fams).astype(bool)]
    for ds, g in oc.groupby("dataset"):
        lines.append(f"\n[{ds or 'default'}] errors-vs-correct z-lift by "
                     f"section (+ = errors read it MORE):")
        for r in g.head(top).itertuples():
            mark = "**" if r.p < 1e-3 else ("*" if r.p < 1e-2 else "")
            lines.append(f"  {r.section[:40]:40} {r.delta:+.3f}  p={r.p:.1e}"
                         f"{mark}  (err/corr n={r.n_err}/{r.n_corr})")
    ar = answer_relation(table, match=match)
    if len(ar):
        lines.append("\nanswer-faithfulness gradient (per-label sections, "
                     "mean within-design z-lift):")
        for r in ar.sort_values(["dataset", "relation"]).itertuples():
            lines.append(f"  [{r.dataset or 'default'}] {r.relation:12} "
                         f"{r.episodes_of:8} {r.mean_zlift:+.3f} (n={r.n})")
    return "\n".join(lines)
