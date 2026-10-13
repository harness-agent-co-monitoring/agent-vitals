"""In-loop per-section digest for the read-informed optimizer run (classification).

Turns the raw replay stores of an iteration's systems into a compact per-section
table the proposer can read in its normal flow (it works in bash: cat / sed /
python3 over val.json). For each candidate, one row per fine-grained section --
keyed by the SAME names the harness declared in prompt_sections() -- carrying:

  footprint_tok  tokens the section costs (the budget lever)
  mass_share     length-inclusive attention it draws
  lift           read lift: per-token read intensity vs fair share
  lift_z_role    lift vs the role norm (the interpretable form)
  consult_ratio  consult ratio: entropy-weighted share over plain share
                 (> 1 = read more when the worker is unsure)
  consult_z_role log2 consult ratio vs the role norm

The role norm is the accumulating reference frame: mean/std over every store
replayed so far in the run (baselines seed it), so a single design's numbers are
legible rather than noise. The measures are outcome-free, so raising them buys the
proposer nothing directly (accuracy stays the sole score). LEGEND below defines every column.

No GPU: everything is pandas over the stored parquets. Per-store aggregates are
cached (section_agg.parquet) so the norm is cheap to recompute each iteration.

Writes, for each requested system x dataset:
  <logs>/<dataset>/<system>/<model>/vitals_digest.md      (cattable)
  <logs>/<dataset>/<system>/<model>/vitals_digest.parquet (python3-sliceable)
and a consolidated <logs>/reports/vitals_iteration_<N>.md.

Usage:
  vitals metaharness digest --logs-dir <.../logs/<run>> \
      --vitals-dir <.../logs/<run>/vitals> --systems a b --iteration 3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vitals.addons.metaharness.digest.common import headlines, named_sections, store_aggregate
from vitals.addons.metaharness.digest.delta import delta_block, demote, parse_hints

# Prepended to every per-system digest and the consolidated report: the
# proposer-facing definition of the measures, faithful to what the code
# computes (vitals.analysis.reading: episode_turns and token_uptake_turns).
LEGEND = """\
Per declared section `s`, from teacher-forced replay of the worker's logged
outputs through the frozen BF16 reference (attention rows softmax-normalized):

- READ LIFT (`lift` column) `= attn_share(s) / length_share(s)`, with the
  attention renormalized over the prompt (the worker's own generated tokens
  are not part of any section, so lift does not fall when it reasons longer) — 1 = read
  like average context, >1 studied, <1 under-read. Read with `footprint_tok`: a
  tiny section can score high lift at negligible `mass_share` (its
  attn_share).
- CONSULT RATIO (`consult_ratio` column)
  `= (sum_t H_t a_t(s) / sum_t H_t) / mean_t a_t(s)` over generated tokens
  `t`: the section's attention share averaged with each token weighted by the
  worker's next-token entropy `H_t`, over its plain average share. 1 = read
  the same whether or not the worker is unsure; >1 = consulted when unsure
  (2.0 = twice the share at uncertain moments); <1 = read while confident;
  n/a = <8 tokens or the section drew no attention.
- CONSULT CONTRAST (`consult_contrast` column) — the ratio's tail form: the
  section's share at the 10% most uncertain generated tokens over its share at
  the 10% most confident (needs 16+ tokens on each side, else n/a). The ratio
  says whether the worker reaches for a section when unsure; the contrast says
  how much (2.0 = twice the share at its hardest moments).
- `*_z_role` — z-score vs all designs so far with the same section role
  (the ratio is z-scored on log2).
- PEAK (`peak` column) — the mean lift of the section's 8 most-read content
  tokens (punctuation, brackets and count markers excluded: delimiter heads
  park mass there), for sections of 32+ tokens (shorter ones are already
  fine-grained: n/a). A low `lift` with a high `peak` is a block the worker
  mostly skips but reads a few lines of hard; the headline names the line.

Averaged over the replayed episode subsample. Correlational, not causal
(read != useful). Nothing here is scored; the objective is what the task
prompt states. Sections named `call<k>/...` belong to an earlier call of a
multi-call prediction; unprefixed sections are the final call's.
"""


PHASE_LEGEND = ("A reasoning worker reads the harness on three paths, and one pooled number hides them: "
                "**harness←thinking** (the think block's attention to the section), **harness←answer** (the "
                "answer tokens' attention to it), and **harness←…←answer** (all hops: the answer's attention "
                "followed back through the worker's own reasoning to the sections that reasoning read — ~80% of "
                "an answer token's attention is on its reasoning, so this is where the answer's dependence sits). "
                "Each of the two measures is given on all three paths. `all hops / thinking` compares the "
                "all-hops answer column with the same all-hops trace applied to the think tokens: above 1 the "
                "section takes a larger part of the reading behind the answer than behind the thinking; below "
                "1 it was deliberation-time reading. The all-hops columns need the generated-side joint (n/a "
                "on older stores) and follow attention weights, not causal contributions.\n\n")


def _phase_table(df: pd.DataFrame) -> str:
    """Named sections, one small table per measure, columns = the three paths."""
    if "lift_think" not in df.columns:
        return ""
    d = named_sections(df)
    d = d[d.lift_think.notna()].sort_values("lift_think", ascending=False)
    if d.empty:
        return ""

    def f(v):
        return "n/a" if v is None or pd.isna(v) else f"{v:.2f}"

    def ex(v):
        return (2 ** v) if v is not None and pd.notna(v) else None

    lift = ["**Read lift** (attention to the section over its length share):", "",
            "| section | harness←thinking | harness←answer | harness←…←answer (all hops) | all hops / thinking |",
            "|---|---|---|---|---|"]
    cons = ["**Consult ratio** (read more when the next token is uncertain; 1 = no preference):", "",
            "| section | harness←thinking | harness←answer | harness←…←answer (all hops) |",
            "|---|---|---|---|"]
    for r in d.itertuples():
        tot, th = getattr(r, "flow_total", None), getattr(r, "flow_think", None)
        ratio = (tot / th) if tot is not None and th and pd.notna(tot) and pd.notna(th) else None
        lift.append(f"| {r.section[:40]} | {f(r.lift_think)} | {f(r.lift_answer)} | {f(tot)} | {f(ratio)} |")
        cons.append(f"| {r.section[:40]} | {f(ex(r.lcr_think))} | {f(ex(r.lcr_answer))} | "
                    f"{f(ex(getattr(r, 'flow_cr_answer', None)))} |")
    out = lift + [""] + cons
    tr = d.flow_traced.dropna() if "flow_traced" in d else pd.Series(dtype=float)
    if len(tr):
        out.append(f"\nAll-hops columns: {tr.mean():.2f} of the answer steps' attention traces back to the prompt "
                   f"without renormalization; the uncaptured top-K tail is renormalized.")
    return "\n".join(out)


def _fmt_table(df: pd.DataFrame) -> str:
    cols = ["section", "role", "footprint_tok", "mass_share", "lift", "peak",
            "lift_z_role", "consult_ratio", "consult_contrast", "consult_z_role"]
    df = df.sort_values(["role", "lift"], ascending=[True, False])
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in df.itertuples():
        pk = getattr(r, "peak_lift", None)
        cc = getattr(r, "consult_contrast", None)
        out.append("| {} | {} | {:d} | {:.4f} | {:.2f} | {} | {} | {} | {} | {} |".format(
            r.section[:40], r.role, int(r.footprint_tok), r.mass_share, r.lift,
            "n/a" if pk is None or pd.isna(pk) else f"{pk:.2f}",
            "n/a" if pd.isna(r.lift_z_role) else f"{r.lift_z_role:+.2f}",
            "n/a" if pd.isna(r.consult_ratio) else f"{r.consult_ratio:.2f}",
            "n/a" if cc is None or pd.isna(cc) else f"{cc:.2f}",
            "n/a" if pd.isna(r.consult_z_role) else f"{r.consult_z_role:+.2f}"))
    return "\n".join(out)


PE_LEGEND = ("One row per replayed example. `read hardest`: the label whose section drew the most "
             "attention per token. `pred` / `gold`: each answer with the read lift of its own section, "
             "or `absent` if no section on the page carried that label. So for every error: `absent` = "
             "the right answer was never on the page (supply); a low lift next to a higher one = it was "
             "there and a rival was read harder (reading). `consulted most`: the scaffold section whose "
             "share rose most at this example's uncertain tokens (its consult ratio), i.e. what the worker "
             "reached for when unsure here. `hot spot`: inside the under-read long sections, the one with the sharpest peak (mean lift of its 8 most-read content tokens) and the line that carries it; on an inventory page this is usually the entry of the label the worker went on to emit. `H_tail`: entropy over the last 16 generated "
             "tokens, i.e. how unsure the worker was when it committed. A design that makes several "
             "worker calls per prediction is read on its final (answer-producing) call; `calls` is how "
             "many it made, and the per-section table lists the earlier calls' sections under a "
             "`call<k>/` prefix.\n\n")


def _label_tag(r, multi: bool) -> str:
    return f"{r.lift:.2f}" + (f"·{r.prefix}" if multi else "")


def _label_reads(units: list[str], labels: pd.DataFrame, multi: bool) -> str:
    """Each answer unit with the read lift of its own label section, or `absent`."""
    parts = []
    for u in units:
        hit = labels[labels.label == u].sort_values("lift", ascending=False)
        parts.append(f"{u} {_label_tag(hit.iloc[0], multi)}" if len(hit) else f"{u} absent")
    return "; ".join(parts) or "–"


def _per_example_map(vdir: Path, val_json: Path | None) -> str:
    """Per-episode read map for one system x dataset store."""
    rows_p = vdir / "section_rows.parquet"
    if not rows_p.exists() or val_json is None:
        return ""
    rows = pd.read_parquet(rows_p)
    try:
        preds = json.loads(val_json.read_text()).get("predictions", [])
    except (OSError, json.JSONDecodeError):
        return ""
    ep_idx = re.compile(r"-(\d{4})-c(\d+)$")

    def units(t):
        t = re.sub(r"<eoa>.*", "", str(t or ""))
        m = re.search(r"\[[^\]]*\](.*)$", t)
        if m:
            t = m.group(1)
        t = t.strip()
        for sep in (";", "；", ",", "，"):
            if sep in t:
                return [u.strip() for u in t.split(sep) if u.strip()]
        return [t] if t else []

    multi_call = "n_calls" in rows and int(rows.n_calls.max()) > 1
    hdr = "| example | ok | read hardest | pred: lift | gold: lift | consulted most | hot spot | H_tail |"
    sep = "|---|---|---|---|---|---|---|---|"
    if multi_call:
        hdr, sep = hdr + " calls |", sep + "---|"
    out = [hdr, sep]
    # the answer comes from a prediction's final call: read that one
    final_call = {}
    for ep in rows.episode.unique():
        m = ep_idx.search(ep)
        if m:
            i, k = int(m.group(1)), int(m.group(2))
            final_call[i] = max(final_call.get(i, -1), k)
    for ep, g in rows.groupby("episode"):
        m = ep_idx.search(ep)
        if not m or int(m.group(2)) != final_call.get(int(m.group(1)), 0):
            continue
        i = int(m.group(1))
        if i >= len(preds):
            continue
        p = preds[i]
        vi = p.get("val_index", i)
        labels = g[g.section.str.contains(":")].copy()
        labels["prefix"] = labels.section.str.split(":", n=1).str[0]
        labels["label"] = labels.section.str.split(":", n=1).str[1]
        labels = labels[~labels.section.str.startswith(("directive:", "span:"))]
        multi = labels.prefix.nunique() > 1     # e.g. cases:<label> and ballot:<label>
        top = labels.sort_values("lift", ascending=False).head(1)
        hardest = f"{top.label.iloc[0]} {_label_tag(top.iloc[0], multi)}" if len(top) else "–"
        named = named_sections(g)
        named = named[named.lcr.notna()] if "lcr" in named else named.iloc[0:0]
        if len(named):
            top_c = named.sort_values("lcr", ascending=False).iloc[0]
            consulted = f"{top_c.section[:30]} {2 ** top_c.lcr:.2f}"
        else:
            consulted = "–"
        hot = "–"
        if "peak_lift" in g.columns:
            hs = g[(g.lift < 1) & g.peak_lift.notna() & (g.peak_lift >= 5 * g.lift)]
            if len(hs):
                r0 = hs.assign(conc=hs.peak_lift / hs.lift).sort_values("conc", ascending=False).iloc[0]
                hot = f"{r0.section[:20]} {r0.peak_lift:.1f}x" + (f" «{str(r0.hot_line)[:40]}»" if isinstance(r0.hot_line, str) else "")
        ht = "n/a"
        try:
            store = vdir / "store" / ep
            parts = sorted(store.glob("tokens-*.parquet"))
            toks = pd.concat([pd.read_parquet(x) for x in parts])
            gen = toks[toks.region_id == "gen"].sort_values("step") if "region_id" in toks else toks
            if len(gen):
                ht = f"{gen.ent.tail(16).mean():.2f}"
        except Exception:
            pass
        out.append(f"| {vi} | {'✓' if p.get('was_correct') else '✗'} | {hardest} | "
                   f"{_label_reads(units(p.get('prediction')), labels, multi)} | "
                   f"{_label_reads(units(p.get('target')), labels, multi)} | {consulted} | {hot} | {ht} |"
                   + (f" {final_call.get(i, 0) + 1} |" if multi_call else ""))
    return "\n".join(out) if len(out) > 2 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", type=Path, required=True)
    ap.add_argument("--vitals-dir", type=Path, required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--norm-dir", type=Path, action="append", default=[],
                    help="additional vitals dirs whose stores enter the role norms but are not digested "
                         "(a probe is scored against the run's accumulated norms)")
    ap.add_argument("--report-name", default=None,
                    help="consolidated report file name (default vitals_iteration_NN.md)")
    ap.add_argument("--per-example", action="store_true",
                    help="append a per-example read map per system (which label sections were read, "
                         "was the gold label's section on the page, commit entropy)")
    ap.add_argument("--parent-hint", action="append", default=[], metavar="NAME=BASE_SYSTEM",
                    help="a candidate's declared base_system (free text; the first existing design it "
                         "names is its parent for the delta block)")
    ap.add_argument("--no-delta", action="store_true", help="skip the candidate-vs-parent delta block")
    args = ap.parse_args()
    hints = parse_hints(args.parent_hint)

    # 1) aggregate every store present (the accumulating reference frame)
    per_store = {}   # (system, dataset) -> agg df
    for store in sorted(args.vitals_dir.glob("*--*/store")):
        system, dataset = store.parent.name.rsplit("--", 1)
        agg = store_aggregate(store)
        if agg is not None and len(agg):
            per_store[(system, dataset)] = agg
    if not per_store:
        print("inloop_digest: no replayed stores yet; nothing to write")
        return
    norm_store = dict(per_store)
    for nd in args.norm_dir:
        for store in sorted(nd.glob("*--*/store")):
            system, dataset = store.parent.name.rsplit("--", 1)
            agg = store_aggregate(store)
            if agg is not None and len(agg):
                norm_store.setdefault((f"norm::{system}", dataset), agg)

    # 2) role norms per (dataset, role) over everything accumulated
    allsec = pd.concat(
        [a.assign(dataset=ds) for (sy, ds), a in norm_store.items()],
        ignore_index=True)
    norm = (allsec.groupby(["dataset", "role"])
            .agg(lift_mu=("lift", "mean"), lift_sd=("lift", "std"),
                 lcr_mu=("lcr", "mean"), lcr_sd=("lcr", "std"),
                 n_designs=("lift", "size"))
            .reset_index())

    def zrole(df, dataset):
        n = norm[norm.dataset == dataset].set_index("role")
        df = df.copy()
        df["lift_z_role"] = df.apply(
            lambda r: (r.lift - n.lift_mu.get(r.role, np.nan))
            / (n.lift_sd.get(r.role, np.nan) or np.nan), axis=1)
        df["consult_z_role"] = df.apply(
            lambda r: (r.lcr - n.lcr_mu.get(r.role, np.nan))
            / (n.lcr_sd.get(r.role, np.nan) or np.nan)
            if pd.notna(r.lcr) else np.nan, axis=1)
        return df

    # 3) per requested system: write digest next to its val.json
    written = []
    consolidated = [f"# Vitals digest, iteration {args.iteration}\n", LEGEND]
    for system in args.systems:
        for (sy, dataset), agg in sorted(per_store.items()):
            if sy != system:
                continue
            tbl = zrole(agg, dataset)
            vjs = list(args.logs_dir.glob(f"{dataset}/{system}/*/val.json"))
            pe = ""
            if args.per_example:
                pe = _per_example_map(args.vitals_dir / f"{system}--{dataset}", vjs[0] if vjs else None)
            pe_block = (f"## Per-example map\n\n{PE_LEGEND}{pe}\n\n") if pe else ""
            hl = headlines(tbl)
            hl_block = (f"## Section headlines (both measures)\n\n{hl}\n\n") if hl else ""
            ph = _phase_table(tbl)
            ph_block = (f"## Reading by phase\n\n{PHASE_LEGEND}{ph}\n\n") if ph else ""
            # what this design's edit did to the reading, against the design it edited
            dl, dl_H = "", None
            if not args.no_delta:
                try:
                    dl, dl_H, _ = delta_block(args.vitals_dir, system, dataset, exclude=args.systems,
                                              hint=hints.get(system))
                except Exception as e:      # never lose the digest to the delta
                    print(f"inloop_digest: delta {system}/{dataset} failed: {e!r}")
            dl_block = (dl + "\n") if dl else ""
            for vj in vjs:
                tbl.to_parquet(vj.parent / "vitals_digest.parquet", index=False)
                if dl_H is not None and len(dl_H):
                    dl_H.to_parquet(vj.parent / "vitals_delta.parquet", index=False)
                (vj.parent / "vitals_digest.md").write_text(
                    f"# {system} / {dataset} — read lift and consult ratio\n\n" + hl_block + ph_block + dl_block + pe_block
                    + "## Per-section table\n\n" + _fmt_table(tbl) + "\n\n## How to read the columns\n\n" + LEGEND)
                written.append(str(vj.parent / "vitals_digest.md"))
            consolidated.append(f"\n## {system} / {dataset}\n\n"
                                + (f"### section headlines\n\n{hl}\n\n" if hl else "")
                                + (f"### reading by phase\n\n{ph}\n\n" if ph else "")
                                + (demote(dl) + "\n\n" if dl else "")
                                + (f"### per-example map\n\n{pe}\n\n" if pe else "")
                                + "### per-section table\n\n" + _fmt_table(tbl))

    # role-norm table at the end of the consolidated report
    consolidated.append("\n## role norms (across all designs so far)\n")
    consolidated.append(
        "| dataset | role | lift_mu | consult_ratio_mu | n_designs |\n|---|---|---|---|---|")
    for r in norm.sort_values(["dataset", "role"]).itertuples():
        consolidated.append(
            f"| {r.dataset} | {r.role} | {r.lift_mu:.2f} | "
            f"{'n/a' if pd.isna(r.lcr_mu) else f'{2 ** r.lcr_mu:.2f}'} | {int(r.n_designs)} |")

    reports = args.logs_dir / "reports"
    reports.mkdir(exist_ok=True)
    cons_path = reports / (args.report_name or f"vitals_iteration_{args.iteration:02d}.md")
    cons_path.write_text("\n".join(consolidated) + "\n")
    print(f"inloop_digest: wrote {len(written)} per-system digests + "
          f"{cons_path}")


if __name__ == "__main__":
    main()
