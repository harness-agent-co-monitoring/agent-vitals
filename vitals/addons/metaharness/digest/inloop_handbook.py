"""Per-section read digest for the read-informed run on HANDBOOK.md (mirror of
digest/inloop_tb.py). Reads the replayed stores under
<vitals-dir>/<agent>--handbook/store and writes, per benchmarked agent, a
`vitals_digest.md` (+ `.parquet`) into its harbor job dir and a consolidated
`reports/vitals_iteration_NN.md` for the next propose step.

Sections are the fine-grained parts of the worker's context: the harness
sentence by sentence (the candidate's `prompt_sections()` declaration, else a
sentence split), the task instruction, the 84 tool schemas, tool results,
the worker's own earlier messages, and the handbook's passages once the
worker has read it in. Four measures per section (read lift, peak lift,
consult ratio, consult contrast), a per-trial map (score, handbook reading
at the turns that act, hottest passage, read-once flag), and a
`Delta vs parent` block for the candidate's edit.

  vitals metaharness digest-handbook --vitals-dir ... --jobs-dir ... --reports-dir ... \
      --systems A B --iteration N [--per-trial] [--parent-hint NAME=BASE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vitals.addons.metaharness.analysis.chat_readmaps import episode_table
from vitals.addons.metaharness.digest.common import frontier_agent, norm_table, zrole
from vitals.addons.metaharness.digest.delta import delta_block, demote, parse_hints
from vitals.addons.metaharness.ingest.handbook import trial_meta

DATASET = "handbook"
ACT = ("write", "exec")
WRITE_PREFIX = ("google_mail__send", "google_mail__reply", "google_mail__move", "google_mail__forward",
                "google_mail__mark", "google_mail__delete", "google_mail__label", "google_mail__create",
                "slack__post", "slack__send", "google_calendar__create", "google_calendar__update",
                "google_calendar__delete", "jira__create", "jira__update", "jira__add", "jira__transition",
                "shopify__create", "shopify__update", "shopify__cancel", "syntara__writeFile")
EXEC_PREFIX = ("syntara__executePython", "syntara__executeBash")
LEGEND = (
    "**Columns.** `footprint_tok` = tokens the section costs in the prompt (median over turns); "
    "`share` = fraction of the worker's prompt attention it drew; `lift` (READ LIFT) = attention share / "
    "length share (1 = read like average text, <1 under-read, >1 studied); `peak` (PEAK LIFT) = mean lift of "
    "the section's 8 hottest content tokens (sections >= 32 tokens: is any line of it studied even when the "
    "average is cold); `consult` (CONSULT RATIO) = entropy-weighted share / plain share (1 = no preference, "
    ">1 read more when the worker is unsure); `contrast` = share at the 10% most-uncertain decode steps / "
    "share at the 10% most-confident (the same question, tails only); `lift_z_role` / `consult_z_role` = vs "
    "the norm for that role across every design replayed so far. Roles: harness sentences (`sys:*`), the "
    "task instruction, tool schemas, tool results (`obs`), the worker's own earlier messages (`self`), and "
    "the handbook's passages (`doc:*`, present only after the worker read the file in). Nothing here is "
    "scored; the rubric score is the only objective.\n"
)
HB_NOTE = (
    "**HANDBOOK note.** The handbook is not in the harness: the worker must find and read it with its "
    "tools, so handbook rows exist only from the turn it did. Turns that act (`write` = mail/slack/jira/"
    "calendar/file writes, `exec` = python/bash) are where a rule has to be applied; `hb_lift@act` is the "
    "handbook's read lift pooled over those turns and `read_once` flags trials that read the document on "
    "entry and never went back (act-turn lift < 0.5 with entry lift > 1). The 65-word harness dilutes past "
    "~30k tokens of context; reading happens mostly while writing the answer, not in the think block.\n"
)
PT_LEGEND = (
    "Per replayed trial: task, rubric score (parent's score on the same task), turns replayed, prompt "
    "tokens at the last turn, pooled lift of the harness / the instruction / the handbook at act turns, "
    "the hottest handbook passage at an act turn with its hot line, `read_once`, and the worker's mean "
    "next-token entropy over the first third vs the last third of the turns.\n\n"
)


def gmean(x: pd.Series) -> float:
    x = x.dropna()
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def kind(calls, action: str) -> str:
    calls = list(calls) if calls is not None and not isinstance(calls, float) else [action]
    calls = [c for c in calls if c]
    if not calls or action == "message":
        return "message"
    if "finish" in calls:
        return "finish"
    if any(c.startswith(WRITE_PREFIX) for c in calls):
        return "write"
    if any(c.startswith(EXEC_PREFIX) for c in calls):
        return "exec"
    if all(c == "think" for c in calls):
        return "think"
    return "read"


def _agent_of_store(name: str) -> str:
    return name.removesuffix(f"--{DATASET}")


def _store_rows(store: Path) -> pd.DataFrame | None:
    cache = store.parent / "section_rows.parquet"
    eps = [e for e in sorted(store.glob("ep-*")) if (e / "attn_tokens.parquet").exists()]
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            if int(df.attrs.get("n_ep", -1)) == len(eps) and "peak" in df.columns:
                return df
        except Exception:
            pass
    if not eps:
        return None
    rows = []
    for ep in eps:
        try:
            rows += episode_table(ep)
        except Exception as exc:
            print(f"inloop_digest_handbook: SKIP {ep.name}: {str(exc)[:80]}")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["kind"] = [kind(c, a) for c, a in zip(df.calls, df.action, strict=True)]
    df["role"] = df.group
    df["turn_id"] = df.episode_id + ":" + df.turn.astype(str)
    df.attrs["n_ep"] = len(eps)
    try:
        df.to_parquet(cache, index=False)
    except Exception:
        pass
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per role, plus one row per harness sentence (role 'system')."""
    def agg(g: pd.DataFrame) -> dict:
        return {"footprint_tok": float(g.span_tok.median()), "share": float(g.groupby("turn_id").share.sum().mean()),
                "lift": float(g.lift.mean()), "peak": float(g.peak.mean()) if g.peak.notna().any() else np.nan,
                "lcr": float(np.log2(gmean(g.consult_ratio))) if g.consult_ratio.notna().any() else np.nan,
                "contrast": float(g.consult_contrast.mean()), "n_turns": int(g.turn_id.nunique())}
    rows = []
    for role, g in df.groupby("role"):
        rows.append({"section": role, "role": role, **agg(g)})
    for sec, g in df[df.role == "system"].groupby("section"):
        rows.append({"section": sec, "role": "system", **agg(g)})
    out = pd.DataFrame(rows)
    out["consult_ratio"] = 2 ** out.lcr
    return out


def _fmt(x, d=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _fmt_main(tbl: pd.DataFrame) -> str:
    order = ["system", "instruction", "handbook", "obs", "self", "user", "tools"]
    tbl = tbl.copy()
    tbl["_o"] = [order.index(r) if r in order else 99 for r in tbl.role]
    tbl["_s"] = (tbl.section != tbl.role).astype(int)
    tbl = tbl.sort_values(["_o", "_s", "lift"], ascending=[True, True, False])
    out = ["| section | role | footprint_tok | share | lift | peak | consult | contrast | lift_z_role | consult_z_role | n_turns |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tbl.itertuples():
        name = r.section if r.section == r.role else f"  {r.section}"
        out.append(f"| {name} | {r.role} | {r.footprint_tok:.0f} | {_fmt(r.share, 3)} | {_fmt(r.lift)} | {_fmt(r.peak)} | "
                   f"{_fmt(r.consult_ratio)} | {_fmt(r.contrast)} | {_fmt(getattr(r, 'lift_z_role', np.nan), 1)} | "
                   f"{_fmt(getattr(r, 'consult_z_role', np.nan), 1)} | {r.n_turns} |")
    return "\n".join(out)


def _headlines(df: pd.DataFrame) -> str:
    sy = df[df.role == "system"].groupby("section").lift.mean().sort_values()
    hb = df[(df.role == "handbook") & df.kind.isin(ACT)]
    lines = []
    if len(sy):
        lines.append(f"- harness sentence read hardest: `{sy.index[-1]}` ({sy.iloc[-1]:.1f}x); under-read most: `{sy.index[0]}` ({sy.iloc[0]:.1f}x)")
    if len(hb):
        top = hb.sort_values("peak", ascending=False).iloc[0]
        lines.append(f"- hottest handbook passage at an act turn: `{str(top.section).split('/')[-1][:48]}` "
                     f"(peak {top.peak:.0f}x, lift {top.lift:.1f}x) — \"{str(top.hot_line)[:70]}\"")
        per_ep = hb.groupby("episode_id").lift.mean()
        lines.append(f"- handbook lift at act turns, per trial: {', '.join(f'{v:.2f}' for v in per_ep.round(2).tolist())} "
                     f"(read_once in {int(_read_once(df).sum())}/{df.episode_id.nunique()} trials)")
    return "\n".join(lines)


def _read_once(df: pd.DataFrame) -> pd.Series:
    out = {}
    for eid, e in df.groupby("episode_id"):
        hb = e[e.role == "handbook"]
        if not len(hb):
            out[eid] = False
            continue
        first = hb.turn.min()
        entry = hb[hb.turn == first].lift.mean()
        acts = hb[(hb.turn > first) & hb.kind.isin(ACT)]
        out[eid] = bool(len(acts) and acts.lift.mean() < 0.5 and entry > 1.0)
    return pd.Series(out)


def _entropy_early_late(ep: Path) -> tuple[float, float]:
    parts = sorted(ep.glob("tokens-*.parquet"))
    if not parts:
        return np.nan, np.nan
    tk = pd.concat([pd.read_parquet(p, columns=["turn", "ent", "region_id"]) for p in parts])
    tk = tk[tk.region_id == "gen"]
    turns = sorted(tk.turn.unique())
    if len(turns) < 3:
        return float(tk.ent.mean()), float(tk.ent.mean())
    k = max(1, len(turns) // 3)
    return float(tk[tk.turn.isin(turns[:k])].ent.mean()), float(tk[tk.turn.isin(turns[-k:])].ent.mean())


def _parent_scores(jobs_dir: Path, parent: str | None) -> dict[str, float]:
    if not parent:
        return {}
    out: dict[str, list[float]] = {}
    for jd in jobs_dir.glob(f"*{parent}-t*"):
        for trial in jd.iterdir():
            rf = trial / "result.json"
            if trial.is_dir() and rf.exists():
                try:
                    r = ((json.loads(rf.read_text()).get("verifier_result") or {}).get("rewards") or {}).get("reward")
                except (json.JSONDecodeError, OSError):
                    r = None
                if r is not None:
                    out.setdefault(trial_meta(trial)["example_idx"], []).append(float(r))
    return {t: float(np.mean(v)) for t, v in out.items()}


def _per_trial_map(df: pd.DataFrame, store: Path, jobs_dir: Path, parent: str | None) -> str:
    pscores = _parent_scores(jobs_dir, parent)
    ro = _read_once(df)
    out = ["| trial | task | score | parent | turns | prompt_tok | harness | instruction | hb_lift@act | hottest passage @act | read_once | H early→late |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for eid, e in df.groupby("episode_id"):
        task = str(e.task.iloc[0])
        score = e.reward.iloc[0]
        acts = e[e.kind.isin(ACT)]
        hb = acts[acts.role == "handbook"]
        hot = hb.sort_values("peak", ascending=False).iloc[0] if len(hb) and hb.peak.notna().any() else None
        he, hl = _entropy_early_late(store / eid)
        out.append(f"| {eid[-14:]} | {task[:34]} | {_fmt(score)} | {_fmt(pscores.get(task))} | {e.turn.nunique()} | "
                   f"{int(e.prompt_tok.max()):,} | {_fmt(e[e.role == 'system'].lift.mean(), 1)} | "
                   f"{_fmt(e[e.role == 'instruction'].lift.mean(), 1)} | {_fmt(hb.lift.mean()) if len(hb) else 'not read'} | "
                   + (f"{str(hot.section).split('/')[-1][:30]} ({hot.peak:.0f}x) \"{str(hot.hot_line)[:40]}\"" if hot is not None else "–")
                   + f" | {'yes' if ro.get(eid) else 'no'} | {_fmt(he)}→{_fmt(hl)} |")
    return "\n".join(out) if len(out) > 2 else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vitals-dir", type=Path, required=True)
    ap.add_argument("--jobs-dir", type=Path, required=True)
    ap.add_argument("--reports-dir", type=Path, required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--report-name", default=None)
    ap.add_argument("--parent", default=None)
    ap.add_argument("--per-trial", action="store_true")
    ap.add_argument("--parent-hint", action="append", default=[], metavar="NAME=BASE_SYSTEM")
    ap.add_argument("--no-delta", action="store_true")
    args = ap.parse_args()
    hints = parse_hints(args.parent_hint)
    parent = args.parent if args.parent is not None else frontier_agent(args.reports_dir)

    per_store: dict[str, tuple[pd.DataFrame, pd.DataFrame, Path]] = {}
    for store in sorted(args.vitals_dir.glob(f"*--{DATASET}/store")):
        rows = _store_rows(store)
        if rows is not None and len(rows):
            per_store[_agent_of_store(store.parent.name)] = (rows, _aggregate(rows), store)
    if not per_store:
        print("inloop_digest_handbook: no replayed stores yet; nothing to write")
        return
    allsec = pd.concat([a.assign(agent=ag) for ag, (_, a, _) in per_store.items()], ignore_index=True)
    allsec = allsec[allsec.section == allsec.role]
    norm = allsec.groupby("role").agg(lift_mu=("lift", "mean"), lift_sd=("lift", "std"), lcr_mu=("lcr", "mean"),
                                      lcr_sd=("lcr", "std"), n_designs=("agent", "nunique"))

    written = []
    consolidated = [f"# Vitals digest (HANDBOOK.md), iteration {args.iteration}\n", LEGEND, HB_NOTE]
    for agent in args.systems:
        if agent not in per_store:
            consolidated.append(f"\n## {agent}\n\n(no replayed trials yet)\n")
            continue
        rows, agg, store = per_store[agent]
        tbl = zrole(agg, norm)
        hl = _headlines(rows)
        pt = _per_trial_map(rows, store, args.jobs_dir, None if parent == agent else parent) if args.per_trial else ""
        dl, dl_H = "", None
        if not args.no_delta:
            try:
                dl, dl_H, _ = delta_block(args.vitals_dir, agent, DATASET, exclude=args.systems, hint=hints.get(agent),
                                          jobs_dir=args.jobs_dir, parent=None if parent == agent else parent)
            except Exception as e:
                print(f"inloop_digest_handbook: delta {agent} failed: {e!r}")
        body = (f"# {agent} — read lift, peak lift, consult ratio, consult contrast (HANDBOOK.md)\n\n"
                + (f"## Section headlines\n\n{hl}\n\n" if hl else "")
                + ((dl + "\n") if dl else "")
                + (f"## Per-trial map\n\n{PT_LEGEND}{pt}\n\n" if pt else "")
                + "## Per-section table\n\n" + _fmt_main(tbl)
                + "\n\n## How to read the columns\n\n" + LEGEND + HB_NOTE)
        targets = [d for d in args.jobs_dir.glob(f"*{agent}-t*") if d.is_dir()] or [args.jobs_dir / agent]
        for jd in targets:
            jd.mkdir(parents=True, exist_ok=True)
            (jd / "vitals_digest.md").write_text(body)
            tbl.to_parquet(jd / "vitals_digest.parquet", index=False)
            if dl_H is not None and len(dl_H):
                dl_H.to_parquet(jd / "vitals_delta.parquet", index=False)
            written.append(str(jd / "vitals_digest.md"))
        consolidated.append(f"\n## {agent}\n\n" + (f"### section headlines\n\n{hl}\n\n" if hl else "")
                            + (demote(dl) + "\n\n" if dl else "")
                            + (f"### per-trial map\n\n{pt}\n\n" if pt else "")
                            + "### per-section table\n\n" + _fmt_main(tbl))
    consolidated += norm_table(norm)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out = args.reports_dir / (args.report_name or f"vitals_iteration_{args.iteration:02d}.md")
    out.write_text("\n".join(consolidated) + "\n")
    print(f"inloop_digest_handbook: wrote {len(written)} per-candidate digests + {out}")


if __name__ == "__main__":
    main()
