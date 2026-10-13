"""In-loop per-section digest for the read-informed optimizer run (terminal-bench).

The terminal-bench counterpart of digest/inloop.py. For each candidate, one
row per fine-grained section of the worker's context, carrying the two
per-section measures pooled over the replayed trials' turns:

  footprint_tok   tokens the section costs (median over turns)
  mass_share      length-inclusive attention it draws (mean over turns)
  lift            read lift: attention share / length share (mean over turns)
  lift_z_role     lift vs the norm for that role across all designs so far
  consult_ratio   consult ratio: entropy-weighted share / plain share
                  (geometric mean over turns; > 1 = read more when unsure)
  consult_z_role  log2 consult ratio vs the role norm
  n_turns         turns the section was in context

Sections: the scaffold sections the harness declared in prompt_sections()
(role `scaffold`, keyed by the names it chose), harness-injected feedback from
feedback_sections() pooled by template (`fb:<name>`, role `feedback`), and the
environment's observations pooled by age (`obs[fresh]`, `obs[1-3]`, `obs[4+]`,
role `observation`). A second, smaller table gives the scaffold sections'
lift and consult ratio by turn bucket, since on this benchmark the reading of
a section changes over the episode.

Written as vitals_digest.md/.parquet inside the candidate's harbor job dir
and consolidated into <reports>/vitals_iteration_NN.md. Pure pandas, no GPU.

Usage:
  vitals metaharness digest-tb --vitals-dir <logs/<run>/vitals> \
      --jobs-dir <jobs/<run>> --reports-dir <logs/<run>/reports> \
      --systems <agent> [<agent> ...] --iteration N
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vitals.addons.metaharness.analysis.tb_readmaps import BUCKETS, SCAFF, episode_table, pooled_section
from vitals.addons.metaharness.digest.common import (
    frontier_agent,
    headlines,
    norm_table,
    result_fields,
    task_stats,
    zrole,
)
from vitals.addons.metaharness.digest.delta import delta_block, demote, parse_hints
from vitals.addons.metaharness.digest.inloop import LEGEND as _TC_LEGEND

# the call<k> sentence is classification-specific (multi-call predictions);
# TB episodes are multi-turn, not multi-call
LEGEND = "\n".join(line for line in _TC_LEGEND.splitlines()
                   if "call<k>" not in line and "multi-call prediction" not in line
                   and "unprefixed sections are the final" not in line)
LEGEND += ("\n`peak` (PEAK LIFT) = mean lift of the section's 8 most-read content tokens (sections >= 32 "
           "tokens): a cold average with a hot peak is a lookup, not a skip. `contrast` (CONSULT CONTRAST) "
           "= the section's attention share at the worker's 10% most-uncertain decode steps over its share "
           "at the 10% most-confident -- the tail version of the consult ratio.")

TB_NOTE = """\
Terminal-bench pooling: rows are the scaffold sections the harness declared
(role `scaffold`), its injected feedback by template (`fb:<name>`, role
`feedback`), and the environment's observations by age in turns (role
`observation`); values pool over every replayed turn in which the section was
in context (the loop replays a sample of each trial's turns -- the first N and
last M -- so per-turn tables cover replayed turns only). The second table follows the scaffold sections over the episode.
`persists` (feedback rows): in how many replayed trials the harness re-injected
the SAME span on 3+ consecutive turns, and the mean longest run. A span the
worker reads every turn while its trigger never clears was read and not acted
on; read lift says nothing about compliance, this column does.
"""


def _agent_of_store(store_parent: str) -> str:
    name = re.sub(r"--tb2$", "", store_parent)
    name = re.sub(r"^evolve-", "", name)
    return re.sub(r"-t\d+$", "", name)


def _pool_name(section: str, turn: int) -> tuple[str, str]:
    """(pooled name, role)."""
    role = ("feedback" if section.startswith("fb:") else
            "observation" if section.startswith("obs:") else "scaffold")
    return pooled_section(section, turn), role


def _store_rows(store: Path) -> pd.DataFrame | None:
    """Per-(episode, turn, section) rows for one candidate's store (cached)."""
    cache = store.parent / "section_rows.parquet"
    eps = [e for e in sorted(store.glob("ep-*")) if (e / "attn_sparse.parquet").exists()]
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            if int(df.attrs.get("n_ep", -1)) == len(eps) and "consult_contrast" in df.columns:
                return df
        except Exception:
            pass
    if not eps:
        return None
    rows = []
    for ep in eps:
        try:
            rows += episode_table(ep)
        except Exception:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.attrs["n_ep"] = len(eps)
    try:
        df.to_parquet(cache, index=False)
    except Exception:
        pass
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    pooled = [_pool_name(s, t) for s, t in zip(d.section, d.turn, strict=True)]
    d["pooled"] = [p for p, _ in pooled]
    d["role"] = [r for _, r in pooled]
    d["mass_share"] = d.lift * d.span_tok / d.prompt_tok.clip(lower=1)
    d["lcr"] = np.log2(d.consult_ratio.where(d.consult_ratio > 0))
    agg = (d.groupby(["pooled", "role"])
           .agg(footprint_tok=("span_tok", "median"), mass_share=("mass_share", "mean"),
                lift=("lift", "mean"), peak=("peak", "mean"), lcr=("lcr", "mean"),
                contrast=("consult_contrast", "mean"), n_turns=("lift", "size"))
           .reset_index().rename(columns={"pooled": "section"}))
    agg["consult_ratio"] = 2 ** agg.lcr
    return agg


def _turn_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.section.isin(SCAFF)].copy()
    d["b"] = [next(n for lo, hi, n in BUCKETS if lo <= t <= hi) for t in d.turn]
    d["lcr"] = np.log2(d.consult_ratio.where(d.consult_ratio > 0))
    L = d.pivot_table(index="section", columns="b", values="lift", aggfunc="mean")
    C = 2 ** d.pivot_table(index="section", columns="b", values="lcr", aggfunc="mean")
    cols = [b[2] for b in BUCKETS if b[2] in L.columns]
    # C may lack buckets whose consult ratios are all n/a (short turns): reindex,
    # never index, so a missing bucket prints n/a instead of raising
    return L.reindex(index=SCAFF, columns=cols), C.reindex(index=SCAFF, columns=cols)


def _persistence(rows: pd.DataFrame) -> dict:
    """Per pooled feedback section: how often the harness re-injected the SAME
    span on consecutive turns. A span injected at observation k is fresh at
    turn k+1 (same rule as obs[fresh]); a streak of fresh injections means the
    condition that triggers it never cleared, i.e. the worker read it and did
    not act on it. Returns {name: (trials_with_streak>=3, trials_seen, mean_longest_streak)}."""
    out = {}
    fb = rows[rows.section.str.startswith("fb:")].copy()
    if not len(fb):
        return out
    fb["k"] = fb.section.str.split(":", n=2).str[1].astype(int)
    fb["name"] = "fb:" + fb.section.str.split(":", n=2).str[2]
    fresh = fb[fb.turn - fb.k - 1 == 0]
    for name, g in fresh.groupby("name"):
        streaks = []
        for _, ge in g.groupby("episode_id"):
            ts = sorted(set(int(t) for t in ge.turn))
            best = run = 1
            for a, b in zip(ts, ts[1:], strict=False):
                run = run + 1 if b == a + 1 else 1
                best = max(best, run)
            streaks.append(best)
        out[name] = (sum(1 for x in streaks if x >= 3), len(streaks), float(np.mean(streaks)))
    return out


def _fmt_main(df: pd.DataFrame, persist: dict | None = None) -> str:
    order = {"scaffold": 0, "feedback": 1, "observation": 2}
    df = df.assign(_o=df.role.map(order)).sort_values(["_o", "lift"], ascending=[True, False])
    persist = persist or {}
    out = ["| section | role | footprint_tok | mass_share | lift | peak | lift_z_role | consult_ratio | contrast | consult_z_role | n_turns | persists |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        pe = persist.get(r.section)
        pcol = f"{pe[0]}/{pe[1]} trials, run {pe[2]:.1f}" if pe else "–"
        out.append("| {} | {} | {:d} | {:.4f} | {:.2f} | {} | {} | {} | {} | {} | {:d} | {} |".format(
            r.section[:40], r.role, int(r.footprint_tok), r.mass_share, r.lift,
            "n/a" if pd.isna(getattr(r, "peak", float("nan"))) else f"{r.peak:.1f}",
            "n/a" if pd.isna(r.lift_z_role) else f"{r.lift_z_role:+.2f}",
            "n/a" if pd.isna(r.consult_ratio) else f"{r.consult_ratio:.2f}",
            "n/a" if pd.isna(getattr(r, "contrast", float("nan"))) else f"{r.contrast:.2f}",
            "n/a" if pd.isna(r.consult_z_role) else f"{r.consult_z_role:+.2f}", int(r.n_turns), pcol))
    return "\n".join(out)


def _fmt_turns(L: pd.DataFrame, C: pd.DataFrame) -> str:
    cols = list(L.columns)
    out = ["| scaffold section | " + " | ".join(f"lift {c}" for c in cols) + " | " + " | ".join(f"consult {c}" for c in cols) + " |",
           "|---|" + "---|" * (2 * len(cols))]
    for s in L.index:
        cells = [("n/a" if pd.isna(L.at[s, c]) else f"{L.at[s, c]:.2f}") for c in cols] + \
                [("n/a" if pd.isna(C.at[s, c]) else f"{C.at[s, c]:.2f}") for c in cols]
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(out)


PT_LEGEND = ("One row per replayed trial: outcome, turns, wall-clock seconds, whether it hit the agent "
             "timeout, and the parent agent's mean pass / turns / seconds / timeouts on the same task "
             "(so cost changes sit next to reading changes). `scaffold`, `obs[fresh]`, `feedback`: read lift "
             "pooled over the trial's turns for your declared prompt sections, the newest observation, and "
             "the text your harness injected. `scaffold consult`: the scaffold's consult ratio pooled over "
             "the trial's turns (>1 = reached for when unsure); `consulted most`: the scaffold or feedback "
             "section with the highest consult ratio in this trial. `H early` / `H late`: the worker's "
             "entropy in the first and last third of the trial; rising entropy toward the end has tracked "
             "failure.\n\n")


def _trial_cost(jobs_dir: Path, agent: str, ep: str) -> dict:
    """Wall-clock seconds, timeout flag and outcome of one trial from its harbor result.json."""
    name = ep[3:] if ep.startswith("ep-") else ep
    for jd in list(jobs_dir.glob(f"*{agent}-t*")) + list(jobs_dir.glob(f"probe-{agent}")):
        rj = jd / name / "result.json"
        if rj.exists():
            try:
                secs, timeout, rew = result_fields(json.loads(rj.read_text()))
                return {"secs": secs, "timeout": timeout, "ok": (rew or 0) > 0}
            except Exception:
                return {}
    return {}


def _pooled_lift(x: pd.DataFrame) -> str:
    """Token-weighted mean lift of a set of section rows."""
    return f"{(x.lift * x.span_tok).sum() / x.span_tok.sum():.2f}" if len(x) and x.span_tok.sum() else "–"


def _per_trial_map(rows: pd.DataFrame, vitals_dir: Path, jobs_dir: Path | None = None,
                   agent: str | None = None, parent: str | None = None) -> str:
    d = rows.copy()
    pstats = task_stats(jobs_dir, parent) if (jobs_dir and parent) else {}
    pooled = [_pool_name(s, t) for s, t in zip(d.section, d.turn, strict=True)]
    d["role"] = [r for _, r in pooled]
    d["pooled"] = [p for p, _ in pooled]
    hdr = "| trial | task | ok | turns | secs | timeout | parent (ok, turns, secs, timeouts) | scaffold | obs[fresh] | feedback | scaffold consult | consulted most | H early | H late |"
    out = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    has_cr = "consult_ratio" in d.columns
    for ep, g in d.groupby("episode_id"):
        sc = _pooled_lift(g[g.role == "scaffold"])
        ob = _pooled_lift(g[g.pooled == "obs[fresh]"])
        fb = _pooled_lift(g[g.role == "feedback"])
        # consult ratio at the same level as lift: pooled over the scaffold, and
        # the scaffold/feedback section the worker reached for most when unsure
        scc, most = "–", "–"
        if has_cr:
            v = g[(g.role == "scaffold") & (g.consult_ratio > 0)].consult_ratio
            if len(v):
                scc = f"{2 ** np.log2(v).mean():.2f}"
            cm = (g[g.role.isin(["scaffold", "feedback"]) & (g.consult_ratio > 0)]
                  .groupby("section").consult_ratio.apply(lambda s: 2 ** np.log2(s).mean()))
            if len(cm):
                most = f"{str(cm.idxmax())[:24]} {cm.max():.2f}"
        turns = int(g.turn.nunique())
        he = hl = "n/a"
        try:
            store = next(vitals_dir.glob(f"*--tb2/store/{ep}"))
            parts = sorted(store.glob("tokens-*.parquet"))
            toks = pd.concat([pd.read_parquet(x) for x in parts])
            gen = toks[toks.region_id == "gen"] if "region_id" in toks else toks
            ts = sorted(gen.turn.unique())
            if len(ts) >= 3:
                k = max(1, len(ts) // 3)
                he = f"{gen[gen.turn.isin(ts[:k])].ent.mean():.2f}"
                hl = f"{gen[gen.turn.isin(ts[-k:])].ent.mean():.2f}"
        except Exception:
            pass
        task = str(g.task.iloc[0]) if "task" in g else "?"
        ok = "✓" if bool(g.passed.iloc[0]) else "✗"
        cost = _trial_cost(jobs_dir, agent, ep) if (jobs_dir and agent) else {}
        secs = f"{cost['secs']:.0f}" if cost.get("secs") is not None else "–"
        to = "TIMEOUT" if cost.get("timeout") else "–"
        ps = pstats.get(task)
        pcol = (f"{np.mean(ps['ok']):.0%}, {np.mean(ps['turns']):.0f}t, {np.mean(ps['secs']):.0f}s, {ps['timeouts']}/{ps['n']}"
                if ps and ps["turns"] and ps["secs"] else "–")
        out.append(f"| {ep[-16:]} | {task} | {ok} | {turns} | {secs} | {to} | {pcol} | {sc} | {ob} | {fb} | {scc} | {most} | {he} | {hl} |")
    return "\n".join(out) if len(out) > 2 else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vitals-dir", type=Path, required=True)
    ap.add_argument("--jobs-dir", type=Path, required=True, help="the run's harbor jobs dir (digest lands in <jobs-dir>/<job of agent>/)")
    ap.add_argument("--reports-dir", type=Path, required=True)
    ap.add_argument("--systems", nargs="+", required=True, help="agent names")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--norm-dir", type=Path, action="append", default=[],
                    help="additional vitals dirs whose stores enter the role norms but are not digested "
                         "(a probe is scored against the run's accumulated norms)")
    ap.add_argument("--report-name", default=None, help="consolidated report file name")
    ap.add_argument("--parent", default=None,
                    help="agent whose per-task outcome/turns/seconds are shown beside each trial in the "
                         "per-trial map (default: the run's frontier agent from logs/<run>/frontier_val.json)")
    ap.add_argument("--per-trial", action="store_true",
                    help="append a per-trial read map per agent: task, outcome, turns replayed, pooled lift of "
                         "the scaffold / freshest observation / injected feedback, entropy early vs late")
    ap.add_argument("--parent-hint", action="append", default=[], metavar="NAME=BASE_SYSTEM",
                    help="a candidate's declared base_system (free text; the first existing agent it names "
                         "is its parent for the delta block, else --parent, else the closest by prompt overlap)")
    ap.add_argument("--no-delta", action="store_true", help="skip the candidate-vs-parent delta block")
    args = ap.parse_args()
    hints = parse_hints(args.parent_hint)
    parent = args.parent if args.parent is not None else frontier_agent(args.reports_dir)

    # 1) every store present = the accumulating reference frame
    per_store: dict[str, pd.DataFrame] = {}
    for store in sorted(args.vitals_dir.glob("*--tb2/store")):
        rows = _store_rows(store)
        if rows is not None and len(rows):
            per_store[_agent_of_store(store.parent.name)] = _aggregate(rows)
    if not per_store:
        print("inloop_digest_tb: no replayed stores yet; nothing to write")
        return
    norm_store = dict(per_store)
    for nd in args.norm_dir:
        for store in sorted(nd.glob("*--tb2/store")):
            rows = _store_rows(store)
            if rows is not None and len(rows):
                norm_store.setdefault("norm::" + _agent_of_store(store.parent.name), _aggregate(rows))
    allsec = pd.concat([a.assign(agent=ag) for ag, a in norm_store.items()], ignore_index=True)
    norm = (allsec.groupby("role").agg(lift_mu=("lift", "mean"), lift_sd=("lift", "std"),
                                       lcr_mu=("lcr", "mean"), lcr_sd=("lcr", "std"),
                                       n_designs=("agent", "nunique")).reset_index().set_index("role"))

    written = []
    consolidated = [f"# Vitals digest (terminal-bench), iteration {args.iteration}\n", LEGEND, TB_NOTE]
    for agent in args.systems:
        agg = per_store.get(agent)
        if agg is None:
            consolidated.append(f"\n## {agent}\n\n(no replayed trials yet)\n")
            continue
        tbl = zrole(agg, norm)
        rows = None
        for store in args.vitals_dir.glob("*--tb2/store"):
            if _agent_of_store(store.parent.name) == agent:
                rows = _store_rows(store)
        L, C = _turn_table(rows) if rows is not None else (None, None)
        persist = _persistence(rows) if rows is not None else {}
        pt = _per_trial_map(rows, args.vitals_dir, args.jobs_dir, agent, parent) if (args.per_trial and rows is not None) else ""
        pt_block = (f"## Per-trial map\n\n{PT_LEGEND}{pt}\n\n") if pt else ""
        hl = headlines(tbl)
        hl_block = (f"## Section headlines (both measures)\n\n{hl}\n\n") if hl else ""
        # what this agent's edit did to the reading of the scaffold, against the agent it edited
        dl, dl_H = "", None
        if not args.no_delta:
            try:
                dl, dl_H, _ = delta_block(args.vitals_dir, agent, "tb2", exclude=args.systems, hint=hints.get(agent),
                                          jobs_dir=args.jobs_dir, parent=None if parent == agent else parent)
            except Exception as e:          # never lose the digest to the delta
                print(f"inloop_digest_tb: delta {agent} failed: {e!r}")
        dl_block = (dl + "\n") if dl else ""
        body = (f"# {agent} — read lift and consult ratio (terminal-bench)\n\n" + hl_block + dl_block + pt_block
                + "## Per-section table\n\n" + _fmt_main(tbl, persist)
                + "\n\n**Scaffold sections over the episode** (by turn bucket):\n\n"
                + (_fmt_turns(L, C) if L is not None else "(n/a)")
                + "\n\n## How to read the columns\n\n" + LEGEND + TB_NOTE)
        targets = ([d for d in args.jobs_dir.glob(f"*{agent}-t*") if d.is_dir()]
                   + [d for d in args.jobs_dir.glob(f"probe-{agent}") if d.is_dir()]) or [args.jobs_dir / agent]
        for jd in targets:
            jd.mkdir(parents=True, exist_ok=True)
            (jd / "vitals_digest.md").write_text(body)
            tbl.to_parquet(jd / "vitals_digest.parquet", index=False)
            if dl_H is not None and len(dl_H):
                dl_H.to_parquet(jd / "vitals_delta.parquet", index=False)
            written.append(str(jd / "vitals_digest.md"))
        consolidated.append(f"\n## {agent}\n\n"
                            + (f"### section headlines\n\n{hl}\n\n" if hl else "")
                            + (demote(dl) + "\n\n" if dl else "")
                            + (f"### per-trial map\n\n{pt}\n\n### per-section table\n\n" if pt else "")
                            + _fmt_main(tbl, persist) + "\n\nScaffold by turn bucket:\n\n"
                            + (_fmt_turns(L, C) if L is not None else "(n/a)"))
    consolidated += norm_table(norm)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out = args.reports_dir / (args.report_name or f"vitals_iteration_{args.iteration:02d}.md")
    out.write_text("\n".join(consolidated) + "\n")
    print(f"inloop_digest_tb: wrote {len(written)} per-candidate digests + {out}")


if __name__ == "__main__":
    main()
