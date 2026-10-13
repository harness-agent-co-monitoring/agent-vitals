"""Terminal-Bench read maps: per-episode turn x section trajectories of the
section measures (read lift and consult ratio).

Sections are the individual fine-grained spans baked into the replay stores:
the declared scaffold sections (task_framing, autonomy_policy,
completion_check, task_description), each observation obs:<k> separately,
and each harness-feedback span fb:<k>:<name>. Nothing is pooled by role.

Stage 1  build    -> <replay>/readmaps.parquet, one row per
                     (episode, turn, section): lift, peak, consult ratio,
                     consult contrast, chi, span tokens.
         lift(t,s) = attn_share(s) / length_share(s) from attn_tokens.parquet
                     (regions baked at replay time).
         consult_ratio(t,s) = entropy-weighted mean share over plain mean
                     share over the forced tokens (>= 8 tokens), from the
                     dense per-token section shares (attn_sparse.parquet for
                     stores replayed without them); chi(t,s) = the within-turn
                     Pearson kept for reference (same sign).
Stage 2  views    (all read the table; build runs automatically if missing)
         matrices  print turn x section matrices (lift and consult ratio) for a
                   matched pass/fail pair with confirmations, or for --episodes.
         atlas     write an HTML gallery of every episode's lift heatmap with
                   salient consult-ratio glyphs (fill = lift, diverging around 1x).
         traj      role-pooled lift / consult ratio by turn bucket and by outcome.
         events    two-channel event taxonomy: scaffold flares, task rereads,
                   confirmation-checklist exposure dynamics, by outcome.

Usage:
    vitals metaharness readmaps build    --replay <dir with the replay stores>
    vitals metaharness readmaps matrices --replay <dir>
    vitals metaharness readmaps atlas    --replay <dir> --out tb_read_atlas.html
    vitals metaharness readmaps events   --replay <dir>
"""

from __future__ import annotations

import argparse
import html as H
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vitals.analysis.reading import consult_contrast, consult_ratio, token_uptake_turns
from vitals.analysis.readmap import PEAK_K, PEAK_MIN_TOK, content_token

SCAFF = ["task_framing", "autonomy_policy", "completion_check", "task_description"]
BUCKETS = [(0, 0, "t0"), (1, 2, "t1-2"), (3, 5, "t3-5"), (6, 10, "t6-10"), (11, 10**6, "t11+")]


# ───────────────────────────── stage 1: build ─────────────────────────────

def _peak_lift(vec, toks, st, en, L):
    """Mean lift of the span's PEAK_K most-read content tokens (>= PEAK_MIN_TOK
    span tokens, else None) -- the peak-lift measure of the digest."""
    n = en - st
    if n < PEAK_MIN_TOK:
        return None
    tl = vec[st:en] * L
    cand = np.flatnonzero([content_token(t) for t in toks[st:en]])
    if len(cand) < PEAK_K:
        return None
    top = cand[np.argpartition(tl[cand], -PEAK_K)[-PEAK_K:]]
    return float(tl[top].mean())


def episode_table(ep: Path) -> list[dict]:
    """Per-(turn, section) lift and chi for one replayed episode."""
    man = json.loads((ep / "manifest.json").read_text())["meta"]
    at = pd.read_parquet(ep / "attn_tokens.parquet",
                         columns=["turn", "idx", "region_id", "attn", "token_str"])
    tk_all = pd.concat([pd.read_parquet(p) for p in sorted(ep.glob("tokens-*.parquet"))])
    tk = tk_all[tk_all.region_id == "gen"][["turn", "step", "ent"]]
    # consult measures need the per-step section shares. The dense per-region
    # shares (attn_share, exact) are the source; the sparse top-K table is only
    # a fallback for stores replayed without them -- it keeps 64 positions per
    # step (~1/3 of the prompt mass) and reads a cold section as zero at most
    # steps, which biased the ratio of diffuse sections down by up to 2x.
    dense = "attn_share" in tk_all.columns and tk_all[tk_all.region_id == "gen"].attn_share.map(
        lambda s: s is not None and len(s) > 0).all()
    sp = None if dense else pd.read_parquet(ep / "attn_sparse.parquet")
    rows: dict[tuple, dict] = {}
    for t, g in at.groupby("turn"):
        t = int(t)
        tot, n = g.attn.sum() or 1e-9, len(g)
        secmap: dict[int, str] = {}
        spans = g[g.region_id.fillna("").str.startswith("span:")]
        g_sorted = g.sort_values("idx")
        vec_all = g_sorted.attn.to_numpy(dtype=float) / tot
        toks_all = g_sorted.token_str.tolist()
        pos_of = {int(i): k for k, i in enumerate(g_sorted.idx)}
        for rid, gg in spans.groupby("region_id"):
            s = rid[5:]
            ps = sorted(pos_of[int(i)] for i in gg.idx)
            peak = _peak_lift(vec_all, toks_all, ps[0], ps[-1] + 1, n)
            rows[(t, s)] = {"turn": t, "section": s, "span_tok": int(len(gg)),
                            "prompt_tok": int(n),
                            "lift": float((gg.attn.sum() / tot) / (len(gg) / n)),
                            "peak": np.nan if peak is None else peak,
                            "chi": np.nan, "consult_ratio": np.nan, "consult_contrast": np.nan}
            for i in gg.idx:
                secmap[int(i)] = s
        if dense:
            continue
        st = sp[sp.turn == t].merge(tk[tk.turn == t], on=["turn", "step"])
        if len(st) < 8:
            continue
        shares: dict[str, dict[int, float]] = {}
        for r in st.itertuples(index=False):
            tot_row = r.prompt_mass + r.gen_mass
            acc: dict[str, float] = {}
            for p, w in zip(r.pos, r.w, strict=True):
                s = secmap.get(int(p))
                if s:
                    acc[s] = acc.get(s, 0.0) + float(w)
            for s, v in acc.items():
                shares.setdefault(s, {})[int(r.step)] = v / max(tot_row, 1e-9)
        steps = st.step.astype(int).tolist()
        ents = st.ent.to_numpy()
        if ents.std() == 0:
            continue
        for s, d in shares.items():
            ys = np.array([d.get(x, 0.0) for x in steps])
            if (t, s) not in rows:
                continue
            if ys.std() > 0:
                rows[(t, s)]["chi"] = float(np.corrcoef(ents, ys)[0, 1])
            cr = consult_ratio(ents, ys)
            rows[(t, s)]["consult_ratio"] = np.nan if cr is None else float(cr)
            cc = consult_contrast(ents, ys)
            rows[(t, s)]["consult_contrast"] = np.nan if cc is None else float(cc)
    if dense:
        for r in token_uptake_turns(ep, tk_all):
            key = (int(r["turn"]), r["section"])
            if key in rows:
                rows[key]["chi"] = np.nan if r["chi_tok"] is None else float(r["chi_tok"])
                rows[key]["consult_ratio"] = np.nan if r["consult_ratio"] is None else float(r["consult_ratio"])
                rows[key]["consult_contrast"] = np.nan if r["consult_contrast"] is None else float(r["consult_contrast"])
    for r in rows.values():
        r.update(episode_id=ep.name, agent=man.get("memory"), task=man.get("example_idx"),
                 reward=man.get("reward"), passed=bool((man.get("reward") or 0) > 0))
    return list(rows.values())


def build(replay: Path) -> pd.DataFrame:
    eps = [e for e in sorted(replay.glob("*/store/ep-*"))
           if (e / "attn_tokens.parquet").exists()
           and ((e / "attn_sparse.parquet").exists() or list(e.glob("tokens-*.parquet")))]
    out = []
    for ep in eps:
        try:
            out += episode_table(ep)
        except Exception as exc:  # one bad store must not kill the table
            print(f"  SKIP {ep.parent.parent.name}/{ep.name}: {str(exc)[:80]}")
    df = pd.DataFrame(out)
    path = replay / "readmaps.parquet"
    df.to_parquet(path, index=False)
    print(f"built {path}: {df.episode_id.nunique()} episodes, {len(df)} (turn, section) rows")
    return df


def load(replay: Path, rebuild: bool = False) -> pd.DataFrame:
    path = replay / "readmaps.parquet"
    if rebuild or not path.exists():
        return build(replay)
    return pd.read_parquet(path)


# ───────────────────────────── helpers ─────────────────────────────

def sec_key(s: str):
    if s in SCAFF:
        return (0, SCAFF.index(s), 0)
    if s.startswith(("obs:", "fb:")):
        return (1, int(s.split(":")[1]), 0 if s.startswith("obs:") else 1)
    # any other declared/structural scaffold section (e.g. the terminus-2
    # stock blocks field_spec / command_structure) sorts after the known
    # scaffold, before the per-turn spans, alphabetically
    return (0, len(SCAFF), 0, s)


def matrix(df: pd.DataFrame, ep: str, value: str) -> pd.DataFrame:
    d = df[df.episode_id == ep].pivot(index="section", columns="turn", values=value)
    return d.reindex(sorted(d.index, key=sec_key))


def pick_pair(df: pd.DataFrame, lo: int = 8, hi: int = 14):
    """A matched pass/fail pair: moderate length, both with a confirmation."""
    meta = (df.groupby("episode_id").agg(passed=("passed", "first"), nt=("turn", "max"),
                                         fb=("section", lambda s: s.str.startswith("fb:").any()))
            .reset_index())
    ok = meta[(meta.nt + 1 >= lo) & (meta.nt + 1 <= hi) & meta.fb]
    p = ok[ok.passed].episode_id.tolist()
    f = ok[~ok.passed].episode_id.tolist()
    return (p[0] if p else None), (f[0] if f else None)


# ───────────────────────────── view: matrices ─────────────────────────────

def view_matrices(df: pd.DataFrame, episodes: list[str] | None) -> None:
    if not episodes:
        episodes = [e for e in pick_pair(df) if e]
    for ep in episodes:
        d = df[df.episode_id == ep]
        if d.empty:
            print(f"(no rows for {ep})")
            continue
        tag = "PASS" if d.passed.iloc[0] else "FAIL"
        for value, fmt in [("lift", "{:4.1f}"), ("consult_ratio", "{:4.2f}")]:
            m = matrix(df, ep, value)
            print(f"\n=== {ep} | {d.agent.iloc[0]} | {tag} | {m.shape[1]} turns | {value} per turn ===")
            print(f"{'section':<26}" + "".join(f" t{t:<4}" for t in m.columns))
            for sec, row in m.iterrows():
                cells = "".join("   .  " if pd.isna(v) else fmt.format(v).rjust(5) + " " for v in row)
                print(f"{sec:<26}{cells}")


# ───────────────────────────── view: atlas (HTML) ─────────────────────────────

def _cls(v):
    if v is None or np.isnan(v):
        return "na"
    x = np.clip(np.log2(max(v, 1e-3)), -2.5, 2.5)
    return f"m{int(round((x + 2.5) / 5 * 10))}"


_CSS = """
:root { color-scheme: light;
 --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e6e5e1; --card:#f6f6f4;
 --good:#0ca30c; --crit:#d03b3b; --gi:#fff;
 --m0:#2a78d6; --m1:#5c93da; --m2:#8fb4e4; --m3:#b7cfee; --m4:#d9e4f5;
 --m5:#f0efec; --m6:#f7dcd9; --m7:#f2bcb8; --m8:#ec9791; --m9:#e57069; --m10:#d03b3b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
 --bg:#1a1a19; --ink:#f2f1ec; --ink2:#c3c2b7; --line:#33332f; --card:#222220;
 --good:#0ca30c; --crit:#e66767; --gi:#111;
 --m0:#3987e5; --m1:#3473c0; --m2:#2f5f9c; --m3:#2c4c74; --m4:#2c3b52;
 --m5:#383835; --m6:#4a2f2d; --m7:#633936; --m8:#8c4642; --m9:#b95552; --m10:#e66767; } }
:root[data-theme="dark"] { color-scheme: dark;
 --bg:#1a1a19; --ink:#f2f1ec; --ink2:#c3c2b7; --line:#33332f; --card:#222220;
 --good:#0ca30c; --crit:#e66767; --gi:#111;
 --m0:#3987e5; --m1:#3473c0; --m2:#2f5f9c; --m3:#2c4c74; --m4:#2c3b52;
 --m5:#383835; --m6:#4a2f2d; --m7:#633936; --m8:#8c4642; --m9:#b95552; --m10:#e66767; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--ink); margin:0; font-family:"Instrument Sans",system-ui,sans-serif; }
main { max-width:1200px; margin:0 auto; padding:28px 20px 80px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:15px; text-transform:uppercase; letter-spacing:0.08em; margin:36px 0 12px;
 color:var(--ink2); border-bottom:1px solid var(--line); padding-bottom:6px; }
.sub { color:var(--ink2); font-size:13px; margin:0 0 18px; }
.legend { display:flex; flex-wrap:wrap; gap:18px; align-items:center; font-size:12px; color:var(--ink2);
 background:var(--card); border:1px solid var(--line); border-radius:6px; padding:10px 14px; }
.sw { display:inline-block; width:13px; height:13px; vertical-align:-2px; }
.ep { margin:0 0 22px; }
.ephead { display:flex; gap:10px; align-items:baseline; font-size:12.5px; margin:0 0 4px;
 font-family:"IBM Plex Mono",ui-monospace,monospace; }
.task { font-weight:600; } .agent,.nt { color:var(--ink2); }
.chip { font-size:10px; font-weight:700; letter-spacing:0.06em; padding:1px 7px; border-radius:3px; color:#fff; }
.chip.pass { background:var(--good); } .chip.fail { background:var(--crit); }
.scroll { overflow-x:auto; }
table { border-collapse:separate; border-spacing:1px; }
td { width:13px; height:13px; min-width:13px; font-size:9px; line-height:13px; text-align:center;
 color:var(--ink); border-radius:2px; }
td.m0,td.m1,td.m9,td.m10 { color:var(--gi); }
td.na { background:transparent; }
""" + "".join(f"td.m{i}{{background:var(--m{i});}}" for i in range(11)) + """
th { font-family:"IBM Plex Mono",ui-monospace,monospace; font-weight:400; font-size:9px;
 color:var(--ink2); text-align:left; padding:0; }
th.rl { text-align:right; padding-right:6px; max-width:190px; overflow:hidden; white-space:nowrap; font-size:9.5px; }
"""


def _block(df: pd.DataFrame, ep: str) -> str:
    d = df[df.episode_id == ep]
    L, C = matrix(df, ep, "lift"), matrix(df, ep, "consult_ratio")
    hdr = "".join(f"<th>{f't{t}' if t % 5 == 0 else ''}</th>" for t in L.columns)
    rows = []
    for s in L.index:
        cells = []
        for t in L.columns:
            v, c = L.at[s, t], C.at[s, t]
            if pd.isna(v):
                cells.append('<td class="na"></td>')
                continue
            g = "&#9652;" if (not pd.isna(c) and c >= 1.3) else ("&#9662;" if (not pd.isna(c) and c <= 0.77) else "")
            tip = f"{s} &middot; t{t} &middot; lift {v:.2f}" + ("" if pd.isna(c) else f" &middot; consult {c:.2f}&times;")
            cells.append(f'<td class="{_cls(v)}" title="{tip}">{g}</td>')
        nm = H.escape(s if len(s) <= 26 else s[:25]) + ("&hellip;" if len(s) > 26 else "")
        rows.append(f'<tr><th class="rl" title="{H.escape(s)}">{nm}</th>{"".join(cells)}</tr>')
    passed = bool(d.passed.iloc[0])
    chip = '<span class="chip pass">PASS</span>' if passed else '<span class="chip fail">FAIL</span>'
    return (f'<div class="ep"><div class="ephead">{chip}<span class="task">{H.escape(str(d.task.iloc[0]))}</span>'
            f'<span class="agent">{H.escape(str(d.agent.iloc[0]))}</span><span class="nt">{L.shape[1]} turns</span></div>'
            f'<div class="scroll"><table><tr><th class="rl"></th>{hdr}</tr>{"".join(rows)}</table></div></div>')


def view_atlas(df: pd.DataFrame, out: Path) -> None:
    meta = df.groupby("episode_id").agg(passed=("passed", "first"), agent=("agent", "first"),
                                        nt=("turn", "max")).reset_index()
    meta = meta.sort_values(["agent", "nt"], ascending=[True, False])
    fails = meta[~meta.passed].episode_id.tolist()
    passes = meta[meta.passed].episode_id.tolist()
    legend = ('<div class="legend"><span><b style="color:var(--ink)">fill = read lift</b> (per turn): '
              + "".join(f'<span class="sw" style="background:var(--m{i})"></span>' for i in range(11))
              + ' &nbsp;under-read &le;&frac14;&times; &larr; 1&times; fair &rarr; &ge;4&times; studied</span>'
              '<span><b style="color:var(--ink)">&#9652;</b> consulted when uncertain (consult ratio &ge; 1.3&times;) '
              '&nbsp;<b style="color:var(--ink)">&#9662;</b> read while calm (&le; 0.77&times;)</span>'
              '<span>rows: scaffold sections, then each obs:k / fb:k in arrival order &middot; hover any cell</span></div>')
    doc = ('<title>Terminal-Bench read atlas</title>'
           '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap">'
           f'<style>{_CSS}</style><main><h1>Terminal-Bench read atlas</h1>'
           f'<p class="sub">Every replayed episode&rsquo;s full turn &times; section read-map &mdash; fine-grained harness '
           f'sections, harness feedback (fb), and each observation individually. Snapshot: {len(meta)} episodes '
           f'({len(passes)} pass / {len(fails)} fail), generated {time.strftime("%Y-%m-%d %H:%M")}.</p>{legend}'
           f'<h2>Failing episodes ({len(fails)})</h2>' + "".join(_block(df, e) for e in fails)
           + f'<h2>Passing episodes ({len(passes)})</h2>' + "".join(_block(df, e) for e in passes) + '</main>')
    out.write_text(doc)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(meta)} episodes)")


# ───────────────────────────── view: traj (macro) ─────────────────────────────

def pooled_section(sec: str, turn: int) -> str:
    """fb:<k>:<name> -> fb:<name>; obs:<k> -> obs[fresh] / obs[1-3] / obs[4+] by
    its age in turns; any other section unchanged."""
    if sec.startswith("fb:"):
        return "fb:" + sec.split(":", 2)[2]
    if sec.startswith("obs:"):
        age = turn - int(sec.split(":")[1]) - 1
        return "obs[fresh]" if age == 0 else ("obs[1-3]" if age <= 3 else "obs[4+]")
    return sec


def view_traj(df: pd.DataFrame) -> None:
    d = df.copy()
    d["role"] = [pooled_section(s, t) for s, t in zip(d.section, d.turn, strict=True)]
    nt = d.groupby("episode_id").turn.max().rename("nt")
    d = d.join(nt, on="episode_id")
    d["last3"] = d.turn >= d.nt - 2
    d["b"] = [next(n for lo, hi, n in BUCKETS if lo <= t <= hi) for t in d.turn]
    order = SCAFF + ["fb:confirm_task_echo", "fb:confirm_checklist", "obs[fresh]", "obs[1-3]", "obs[4+]"]
    bn = [b[2] for b in BUCKETS]
    print(f"{d.episode_id.nunique()} episodes ({d[d.passed].episode_id.nunique()} pass / "
          f"{d[~d.passed].episode_id.nunique()} fail)\n")
    d["lcr"] = np.log2(d.consult_ratio)
    for value, label in [("lift", "READ LIFT"), ("lcr", "CONSULT RATIO (geometric mean)")]:
        f = (lambda x: x) if value == "lift" else (lambda x: 2 ** x)
        print(f"== {label} by turn bucket ==")
        print(f(d.pivot_table(index="role", columns="b", values=value, aggfunc="mean").reindex(order)[bn]).round(2).to_string())
        p = f(d[d.last3].pivot_table(index="role", columns="passed", values=value, aggfunc="mean").reindex(order))
        p.columns = ["fail", "pass"]
        print(f"\n== {label}, last 3 turns: pass vs fail ==")
        print(p.round(2).to_string(), "\n")


# ───────────────────────────── view: events ─────────────────────────────

def view_events(df: pd.DataFrame, flare_abs: float = 2.5, reread: float = 3.0) -> None:
    ep_rows, exp_rows, flare_rows = [], [], []
    for ep, d in df.groupby("episode_id"):
        passed = bool(d.passed.iloc[0])
        L = d.pivot(index="section", columns="turn", values="lift")
        C = np.log2(d.pivot(index="section", columns="turn", values="consult_ratio"))
        turns = sorted(L.columns)
        sl = L.reindex(SCAFF).mean(axis=0)
        thr = max(flare_abs, 2 * float(np.nanmedian(sl)))
        n_flare = n_reread = 0
        for t in turns:
            if sl[t] >= thr:
                n_flare += 1
                old = [C.at[s, t] for s in C.index if s.startswith("obs:")
                       and t - int(s.split(":")[1]) - 1 >= 2 and not pd.isna(C.at[s, t])]
                scf = [C.at[s, t] for s in SCAFF if s in C.index and not pd.isna(C.at[s, t])]
                flare_rows.append(dict(passed=passed, scaff_chi=np.mean(scf) if scf else np.nan,
                                       oldobs_chi_max=max(old) if old else np.nan))
            if "task_description" in L.index and L.at["task_description", t] >= reread:
                n_reread += 1
        confs = [s for s in L.index if s.startswith("fb:") and s.endswith("confirm_checklist")]
        for name in confs:
            ts = [t for t in turns if not pd.isna(L.at[name, t])]
            for i, t in enumerate(ts):
                exp_rows.append(dict(passed=passed, exposure=min(i + 1, 3), lift=L.at[name, t], chi=C.at[name, t]))
        ep_rows.append(dict(ep=ep, passed=passed, n_turns=len(turns), n_flare=n_flare,
                            n_reread=n_reread, n_conf=len(confs)))
    E, X, F = pd.DataFrame(ep_rows), pd.DataFrame(exp_rows), pd.DataFrame(flare_rows)
    print(f"{len(E)} episodes: {int(E.passed.sum())} pass / {int((~E.passed).sum())} fail\n")
    print("== per-episode event counts (mean per 10 turns) ==")
    for name, col in [("scaffold flares", "n_flare"), (f"task rereads (lift>={reread:g})", "n_reread"),
                      ("confirmations", "n_conf")]:
        p = (E[E.passed][col] / E[E.passed].n_turns * 10).mean()
        f = (E[~E.passed][col] / E[~E.passed].n_turns * 10).mean()
        print(f"  {name:<26} pass {p:.2f}  fail {f:.2f}")
    if len(F):
        print(f"\n== inside flare turns (n={len(F)}): scaffold consult ratio {2 ** F.scaff_chi.mean():.2f}x"
              f" vs old-obs max {2 ** F.oldobs_chi_max.mean():.2f}x ==")
    if len(X):
        print("\n== confirmation checklist by exposure (1st, 2nd, 3rd+ turn in context), by outcome ==")
        g = X.groupby(["passed", "exposure"]).agg(lift=("lift", "mean"), lcr=("chi", "mean"), n=("lift", "size"))
        g["consult_ratio"] = 2 ** g.pop("lcr")
        print(g.round(3).to_string())


# ───────────────────────────── main ─────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("view", choices=["build", "matrices", "atlas", "traj", "events"])
    ap.add_argument("--replay", type=Path, required=True, help="replay root with <job>--tb2/store/ep-*")
    ap.add_argument("--rebuild", action="store_true", help="recompute readmaps.parquet")
    ap.add_argument("--episodes", nargs="*", help="matrices: episode ids (default: a matched pass/fail pair)")
    ap.add_argument("--out", type=Path, help="atlas: output html (default <replay>/tb_read_atlas.html)")
    args = ap.parse_args()
    if args.view == "build":
        build(args.replay)
        return
    df = load(args.replay, rebuild=args.rebuild)
    if args.view == "matrices":
        view_matrices(df, args.episodes)
    elif args.view == "atlas":
        view_atlas(df, args.out or (args.replay / "tb_read_atlas.html"))
    elif args.view == "traj":
        view_traj(df)
    elif args.view == "events":
        view_events(df)


if __name__ == "__main__":
    main()
