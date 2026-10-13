"""Helpers shared by the classification, Terminal-Bench and HANDBOOK.md digests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from vitals.addons.metaharness.analysis.roles import role_of, span_footprints
from vitals.analysis.phases import answer_flow, section_phase_rows
from vitals.analysis.reading import episode_turns, token_uptake_turns
from vitals.analysis.readmap import PEAK_K, PEAK_MIN_TOK, section_peaks
from vitals.core.store import read_regions, read_token_rows


def _log2(v):
    return None if not v or v <= 0 else float(np.log2(v))


def store_aggregate(store: Path) -> pd.DataFrame | None:
    """Per-section aggregate for one system x dataset store (cached)."""
    cache = store.parent / "section_agg.parquet"
    eps = [e for e in sorted(store.glob("ep-*"))
           if (e / "attn_received.jsonl").exists()]
    if cache.exists() and (store.parent / "section_rows.parquet").exists():
        try:
            df = pd.read_parquet(cache)
            if (int(df.attrs.get("n_ep", -1)) == len(eps) and "lcr" in df.columns and "lcc" in df.columns
                    and df.attrs.get("lift_norm") == "context"):
                return df
        except Exception:
            pass
    if not eps:
        return None
    rows = []
    for ep in eps:
        toks = read_token_rows(ep)
        if toks is None:
            continue
        turns = episode_turns(ep, tokens=toks)
        if not turns:
            continue
        tr = turns[0]                       # classification: single turn
        foot = span_footprints(ep)
        ut = {r["section"]: r for r in token_uptake_turns(ep, tokens=toks) if r["turn"] == 0}
        # which call of its prediction this episode is (multi-call designs):
        # sections of a non-final call are reported under a "call<k>/" prefix
        call, n_calls = 0, 1
        try:
            meta = json.loads((ep / "manifest.json").read_text()).get("meta", {})
            call, n_calls = int(meta.get("call_idx", 0)), int(meta.get("n_calls", 1))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        pre = f"call{call}/" if n_calls > 1 and call < n_calls - 1 else ""
        # the three parts of the read: harness<-thinking, harness<-answer, and
        # (with the generated-side joint) harness<-thinking<-answer
        ph = {r["section"]: r for r in section_phase_rows(ep, tokens=toks) if r["turn"] == 0}
        flow = answer_flow(ep, tokens=toks)
        fl = flow.get("sections", {})
        fl_traced = (flow.get("answer_row") or {}).get("traced")
        pk = section_peaks(ep, read_regions(ep).get(0, []), tr["L"])
        for name, d in tr["sections"].items():
            fp = foot.get(name, 0)
            cr = (ut.get(name) or {}).get("consult_ratio")
            cc = (ut.get(name) or {}).get("consult_contrast")
            p, f = ph.get(name, {}), fl.get(name, {})
            rows.append({"episode": ep.name, "section": pre + name, "role": role_of(name),
                         "call": call, "n_calls": n_calls, "ctx_share": tr.get("ctx_share"),
                         "lift": d["lift"], "mass_share": d["W"] * fp,
                         "footprint_tok": fp,
                         "lcr": _log2(cr),
                         "lcc": _log2(cc),   # log2 consult contrast
                         "lift_think": p.get("lift_think"), "lift_answer": p.get("lift_answer"),
                         "lcr_think": _log2(p.get("consult_think")), "lcr_answer": _log2(p.get("consult_answer")),
                         "flow_direct": f.get("direct"), "flow_indirect": f.get("indirect"),
                         "flow_total": f.get("reach"), "flow_think": f.get("reach_think"), "flow_traced": fl_traced,
                         "flow_cr_think": _log2(f.get("consult_think_traced")),
                         "flow_cr_answer": _log2(f.get("consult_answer_traced")),
                         "peak_lift": pk.get(name, (None, None))[0], "hot_line": pk.get(name, (None, None))[1]})
    if not rows:
        return None
    r = pd.DataFrame(rows)
    try:
        r.to_parquet(store.parent / "section_rows.parquet", index=False)  # per-episode rows (read maps)
    except Exception:
        pass
    agg = (r.groupby(["section", "role"])
           .agg(footprint_tok=("footprint_tok", "median"),
                mass_share=("mass_share", "mean"),
                lift=("lift", "mean"),
                lcr=("lcr", "mean"),          # log2 consult ratio (geometric mean)
                lcc=("lcc", "mean"),          # log2 consult contrast (geometric mean)
                n_ep=("lift", "size"),
                lift_think=("lift_think", "mean"), lift_answer=("lift_answer", "mean"),
                lcr_think=("lcr_think", "mean"), lcr_answer=("lcr_answer", "mean"),
                flow_direct=("flow_direct", "mean"), flow_indirect=("flow_indirect", "mean"),
                flow_total=("flow_total", "mean"), flow_think=("flow_think", "mean"),
                flow_traced=("flow_traced", "mean"),
                flow_cr_think=("flow_cr_think", "mean"), flow_cr_answer=("flow_cr_answer", "mean"),
                peak_lift=("peak_lift", "mean"),
                hot_line=("hot_line", lambda v: (v.dropna().mode().iloc[0] if v.notna().any() else None)),
                hot_line_n=("hot_line", lambda v: int((v == v.dropna().mode().iloc[0]).sum()) if v.notna().any() else 0))
           .reset_index())
    agg["consult_ratio"] = 2 ** agg.lcr
    agg["consult_contrast"] = 2 ** agg.lcc
    agg.attrs["n_ep"] = len(eps)
    agg.attrs["lift_norm"] = "context"
    try:
        agg.to_parquet(cache, index=False)
    except Exception:
        pass
    return agg


def named_sections(df: pd.DataFrame) -> pd.DataFrame:
    """Sections that are harness scaffold rather than one-per-label content
    (`cases:<label>`, `ballot:<label>` ...); `directive:k` and `span:` stay."""
    s = df.section.astype(str).str.replace(r"^call\d+/", "", regex=True)
    label_like = s.str.contains(":") & ~s.str.startswith(("directive:", "span:"))
    return df[~label_like]


def headlines(df: pd.DataFrame, min_n: int = 3) -> str:
    """Both measures at the same level, before any table: the sections read
    hardest and the ones under-read at the highest cost (read lift), and the
    sections consulted when unsure and the ones read only while confident
    (consult ratio), over the named (non-label) sections."""
    ncol = "n_ep" if "n_ep" in df.columns else ("n_turns" if "n_turns" in df.columns else None)
    d = named_sections(df)
    if ncol:
        d = d[d[ncol] >= min_n]
    if d.empty:
        return ""

    def z(v):
        return "n/a" if pd.isna(v) else f"{v:+.1f}"

    def lift_row(r):
        return f"`{r.section[:36]}` {r.lift:.2f}x (z {z(r.lift_z_role)}, {int(r.footprint_tok)} tok)"

    def cons_row(r):
        return f"`{r.section[:36]}` {r.consult_ratio:.2f}x (z {z(r.consult_z_role)})"

    def lst(rows, f):
        return "; ".join(f(r) for r in rows.itertuples()) or "none"

    studied = d.sort_values("lift", ascending=False).head(3)
    costly = d[(d.footprint_tok >= 40) & (d.lift < 1)].sort_values("lift").head(3)
    c = d[d.consult_ratio.notna()]
    consulted = c[c.consult_ratio >= 1.05].sort_values("consult_ratio", ascending=False).head(3)
    confident = c[c.consult_ratio <= 0.95].sort_values("consult_ratio").head(3)
    hot = ""
    if "peak_lift" in d.columns:
        h = d[(d.lift < 1) & d.peak_lift.notna() & (d.peak_lift >= 5 * d.lift) & (d.peak_lift >= 1)]
        h = h.assign(conc=h.peak_lift / h.lift).sort_values("conc", ascending=False).head(3)
        def hot_row(r):
            n_ep = int(getattr(r, ncol)) if ncol else 0
            base = f"`{r.section[:36]}` peak {r.peak_lift:.2f}x vs {r.lift:.2f}x mean"
            if isinstance(r.hot_line, str) and n_ep and int(r.hot_line_n) >= max(3, n_ep // 4):
                return base + f" — «{r.hot_line}» ({int(r.hot_line_n)}/{n_ep} eps)"
            return base + " — the hot line varies by example (per-example map)"
        hot = ("\n- **Hot spots** — lines read hard inside under-read sections (peak = mean lift of the "
               f"{PEAK_K} most-read tokens; sections ≥ {PEAK_MIN_TOK} tokens): " + lst(h, hot_row) + ".")
    return ("- **Read lift** — read hardest: " + lst(studied, lift_row)
            + ".\n  Under-read at the highest cost (large footprint, low lift): " + lst(costly, lift_row) + ".\n"
            "- **Consult ratio** — consulted when unsure: " + lst(consulted, cons_row)
            + ".\n  Read only while confident: " + lst(confident, cons_row) + "." + hot)


def result_fields(d: dict) -> tuple[float | None, bool, float | None]:
    """(wall-clock seconds, timed out, reward) from a harbor result.json."""
    ae = d.get("agent_execution") or {}
    t0, t1 = ae.get("started_at") or d.get("started_at"), ae.get("finished_at") or d.get("finished_at")
    secs = ((datetime.fromisoformat(t1.replace("Z", "+00:00"))
             - datetime.fromisoformat(t0.replace("Z", "+00:00"))).total_seconds() if t0 and t1 else None)
    exc = (d.get("exception_info") or {}).get("exception_type") or ""
    vr = d.get("verifier_result") or {}
    rew = list(vr.get("rewards", {}).values())[0] if isinstance(vr, dict) and vr.get("rewards") else d.get("reward")
    return secs, "Timeout" in exc, rew


def task_stats(jobs_dir: Path, agent: str) -> dict:
    """Per task for one agent: mean pass, mean turns, mean seconds, timeouts/trials (all its trials)."""
    stats = {}
    dirs = list(jobs_dir.glob(f"*{agent}-t*"))
    if agent.startswith("baseline_"):     # Phase-0 jobs are named <name>-baseline-t<k>
        dirs += list(jobs_dir.glob(f"{agent.removeprefix('baseline_')}-baseline-t*"))
    for jd in dirs:
        for rj in jd.glob("*/result.json"):
            task = rj.parent.name.rsplit("__", 1)[0]
            try:
                d = json.loads(rj.read_text())
            except Exception:
                continue
            secs, timeout, rew = result_fields(d)
            turns = None
            tj = rj.parent / "agent" / "trajectory.json"
            if tj.exists():
                try:
                    turns = sum(1 for st in json.loads(tj.read_text()).get("steps", []) if st.get("source") == "agent")
                except Exception:
                    pass
            st = stats.setdefault(task, {"ok": [], "turns": [], "secs": [], "timeouts": 0, "n": 0})
            st["ok"].append(1 if (rew or 0) > 0 else 0)
            st["n"] += 1
            if turns is not None:
                st["turns"].append(turns)
            if secs is not None:
                st["secs"].append(secs)
            st["timeouts"] += 1 if timeout else 0
    return stats


def frontier_agent(reports_dir: Path) -> str | None:
    """The run's best agent recorded in <run>/frontier_val.json, if any."""
    try:
        return json.loads((reports_dir.parent / "frontier_val.json").read_text()).get("_best", {}).get("agent")
    except Exception:
        return None


def zrole(df: pd.DataFrame, norm: pd.DataFrame) -> pd.DataFrame:
    """Add lift_z_role and consult_z_role (on log2) against per-role norms indexed by role."""
    df = df.copy()
    df["lift_z_role"] = [(r.lift - norm.lift_mu.get(r.role, np.nan)) / (norm.lift_sd.get(r.role, np.nan) or np.nan)
                         for r in df.itertuples()]
    df["consult_z_role"] = [((r.lcr - norm.lcr_mu.get(r.role, np.nan)) / (norm.lcr_sd.get(r.role, np.nan) or np.nan))
                            if pd.notna(r.lcr) else np.nan for r in df.itertuples()]
    return df


def norm_table(norm: pd.DataFrame) -> list[str]:
    """The role-norm section that ends a consolidated report."""
    out = ["\n## role norms (across all designs so far)\n",
           "| role | lift_mu | consult_ratio_mu | n_designs |\n|---|---|---|---|"]
    for role, r in norm.iterrows():
        out.append(f"| {role} | {r.lift_mu:.2f} | {'n/a' if pd.isna(r.lcr_mu) else f'{2 ** r.lcr_mu:.2f}'} | {int(r.n_designs)} |")
    return out
