// Evolution: a Meta-Harness run's candidates. Score by iteration (and against context size when the run
// logs it), each candidate's section reading from its digest, absolute or against its parent or a reference
// candidate, and per-task scores. The selected candidate's hypothesis, lineage and reading changes on the right.

import { divergingColor, ratioColor, scalePos, sequentialColor } from "./color.js";
import { binOf, binsFor, heatmap, labelWidth, legend } from "./heatmap.js";
import { buildRows, defaultFold, groupChips, groupLabel } from "./rows.js";
import { scatterChart } from "./scatter.js";
import { api, checkbox, fmt, go, h, href, list, route, s, seg, setParams, table, tip } from "./util.js";

const MEASURES = [
  ["lift", "Read lift", "Attention per token over the context mean, averaged over the candidate's episodes."],
  ["peak", "Peak lift", "Lift of the most-read content tokens."],
  ["consult_ratio", "Consult ratio", "Whether attention to the section rises at uncertain steps."],
  ["consult_contrast", "Consult contrast", "Attention at the most uncertain steps over the least uncertain."],
];
const LABEL = Object.fromEntries(MEASURES.map(([k, l]) => [k, l]));
const TASK_BINS = [0, 0.25, 0.5, 0.75, 1].map((lo, i, a) => ({
  id: `t${i}`, lo: i === 0 ? -Infinity : lo, hi: i === a.length - 1 ? Infinity : a[i + 1],
  text: i === a.length - 1 ? "1" : i === 0 ? "< 0.25" : `${lo}–${a[i + 1]}`, t: 0,
}));

function settings(p, sections, datasets = []) {
  const fold = defaultFold(sections);
  for (const d of datasets) fold.delete(`g:${d}`);
  return {
    run: p.run, cand: p.cand || null, sec: p.sec || null,
    color: LABEL[p.color] ? p.color : "lift",
    cmp: p.cmp === "parent" || p.cmp === "ref" ? p.cmp : "abs",
    ref: p.ref || null,
    range: p.range || "auto",
    tex: p.tex === "1",
    view: p.view === "table" ? "table" : "chart",
    q: p.q || "",
    hide: new Set(list(p.hide)),
    fold: p.fold == null ? fold : new Set(list(p.fold).filter((k) => k !== "-")),
    sort: p.sort || "order",
    grouped: p.grouped !== "0",
    bins: new Set(list(p.bins)),
    tbins: new Set(list(p.tbins)),
    i0: p.i0 != null ? Number(p.i0) : null, i1: p.i1 != null ? Number(p.i1) : null,
    lh: new Set(list(p.lh)),
    tsort: p.tsort || "", ttsort: p.ttsort || "",
    task: p.task || null, tq: p.tq || "", trows: p.trows || "name",
  };
}

export function evolutionView(root, ctx) {
  let data = null, hm = null, thm = null, charts = [], hoverBin = null, hoverTBin = null;
  let st = settings(route().params, []);
  const content = h("div", { class: "content" });
  const panel = h("div", { class: "panel" });
  const split = h("div", { class: "split evo" }, content);
  root.replaceChildren(split);

  function update(patch, rerender = true) {
    const before = st.run;
    setParams(Object.fromEntries(Object.entries(patch).map(([k, v]) => [k, v instanceof Set ? [...v].join(",") || null : v])));
    st = settings(route().params, data?.matrix.sections || [], data?.run.datasets);
    if (st.run !== before) load(false);
    else if (rerender) render();
  }

  function render() {
    if (!data) return;
    const scale = data.run.score_scale;
    const fmtScore = (v) => (v == null ? "–" : `${(scale === 1 ? v * 100 : v).toFixed(1)}%`);
    const m = st.color;

    // -- candidates and their digest cells --------------------------------------------------------------
    const all = data.candidates.map((c, i) => ({ ...c, order: i, label: c.name }));
    const byName = new Map(all.map((c) => [c.name, c]));
    for (const c of all) c.children = all.filter((x) => x.parent === c.name).map((x) => x.name);
    const F = data.matrix.fields;
    const cellsBy = new Map(), digestOf = new Map();
    for (const row of data.matrix.cells) {
      const o = Object.fromEntries(F.map((k, i) => [k, row[i]]));
      cellsBy.set(`${o.section}\u0000${o.candidate}`, o);
      if (!digestOf.has(o.candidate)) digestOf.set(o.candidate, []);
      digestOf.get(o.candidate).push(o);
    }
    const inRange = (c) => c.baseline || ((st.i0 == null || c.iteration >= st.i0) && (st.i1 == null || c.iteration <= st.i1));
    const cols = all.filter(inRange);
    const zoomed = st.i0 != null || st.i1 != null;
    const hasParents = all.some((c) => c.parent && byName.has(c.parent));
    const refName = st.ref && byName.has(st.ref) ? st.ref : (all.find((c) => c.baseline) || all[0])?.name;
    const cmp = st.cmp === "parent" && !hasParents ? "abs" : st.cmp;
    const against = (c) => (cmp === "parent" ? (c.parent && byName.has(c.parent) ? c.parent : null) : cmp === "ref" ? refName : null);
    const sel = st.cand && byName.has(st.cand) ? byName.get(st.cand) : null;
    const secs = data.matrix.sections;
    const secBy = new Map(secs.map((x) => [x.name, x]));

    // value shown for (row, candidate): the measure, or its ratio to the comparison candidate's
    const raw = (name, cand) => cellsBy.get(`${name}\u0000${cand}`);
    const aggregate = (row, cand) => {
      if (row.kind === "section") return raw(row.name, cand) || null;
      let tok = 0, mass = 0, n = 0, crw = 0, crm = 0, ccw = 0, ccm = 0, peak = null;
      for (const name of row.members) {
        const c = raw(name, cand);
        if (!c) continue;
        n++;
        const t = c.tokens || 0, w = (c.lift ?? 0) * t;
        tok += t; mass += w;
        if (c.consult_ratio != null) { crw += c.consult_ratio * w; crm += w; }
        if (c.consult_contrast != null) { ccw += c.consult_contrast * w; ccm += w; }
        if (c.peak != null && (peak == null || c.peak > peak)) peak = c.peak;
      }
      return n ? { lift: tok ? mass / tok : null, peak, consult_ratio: crm ? crw / crm : null, consult_contrast: ccm ? ccw / ccm : null, tokens: tok, n } : null;
    };
    const memo = new Map();
    function cellOf(row, c) {
      const k = `${row.key}\u0000${c.name}`;
      if (memo.has(k)) return memo.get(k);
      const own = aggregate(row, c.name);
      let out = null;
      const other = against(c);
      if (!digestOf.has(c.name) || (other && !digestOf.has(other))) { memo.set(k, null); return null; }
      if (cmp === "abs") out = own ? { ...own, v: own[m] } : null;
      else if (!other) out = own ? { ...own, v: null, noRef: true } : null;
      else {
        const base = aggregate(row, other);
        if (own && base) out = { ...own, v: own[m] > 0 && base[m] > 0 ? own[m] / base[m] : null, base: base[m], other };
        else if (own) out = { ...own, v: null, added: true, other };
        else if (base) out = { removed: true, base: base[m], other };
      }
      memo.set(k, out);
      return out;
    }
    const rowStat = new Map();
    const statOf = (row) => {
      if (!rowStat.has(row.key)) {
        const ls = cols.map((c) => cellOf(row, c)?.v).filter((v) => v > 0).map(Math.log2);
        rowStat.set(row.key, ls.length ? {
          mean: 2 ** (ls.reduce((a, b) => a + b, 0) / ls.length), n: ls.length,
          change: cmp === "abs" ? Math.max(...ls) - Math.min(...ls) : Math.max(...ls.map(Math.abs)),
        } : null);
      }
      return rowStat.get(row.key);
    };
    const rows = buildRows(secs, {
      fold: st.fold, hide: st.hide, q: st.q, sort: st.sort, grouped: st.grouped,
      stats: {
        mean: (n) => statOf({ key: `s:${n}`, kind: "section", name: n, members: [n] })?.mean ?? null,
        change: (n) => statOf({ key: `s:${n}`, kind: "section", name: n, members: [n] })?.change ?? null,
      },
    });
    const vals = [];
    for (const r of rows) for (const c of cols) { const v = cellOf(r, c)?.v; if (v > 0) vals.push(Math.abs(Math.log2(v))); }
    vals.sort((a, b) => a - b);
    const p90 = vals.length ? vals[Math.floor(vals.length * 0.9)] : 1;
    const R = st.range === "auto" ? Math.max(1, Math.min(4, Math.ceil(p90))) : Number(st.range);
    const bins = binsFor(R);
    const activeBins = new Set([...st.bins].filter((b) => bins.some((x) => x.id === b)));

    // -- header and controls ----------------------------------------------------------------------------
    const scored = all.filter((c) => c.score != null);
    const best = scored.reduce((b, c) => (!b || c.score > b.score ? c : b), null);
    const iters = new Set(all.filter((c) => c.iteration != null).map((c) => c.iteration));
    const runs = ctx.sources?.runs || [];
    const head = h("div", { class: "head" },
      h("h1", { text: data.run.label }),
      h("div", { class: "facts" },
        h("span", null, h("b", { text: String(all.filter((c) => !c.baseline).length) }), " candidates over ", h("b", { text: String(iters.size) }), " iterations"),
        all.some((c) => c.baseline) ? h("span", null, h("b", { text: String(all.filter((c) => c.baseline).length) }), " baselines") : null,
        best ? h("span", null, "best ", h("b", { text: fmtScore(best.score) }), ` (${best.name})`) : null));
    const controls1 = h("div", { class: "controls" },
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Run" }), h("select", { "aria-label": "Run", onchange: (e) => go("/evolution", { run: e.target.value }) },
        runs.map((r) => h("option", { value: r.id, text: r.label, selected: r.id === st.run })))),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Color" }), seg(MEASURES.filter(([k]) => data.matrix.cells.some((row) => row[F.indexOf(k)] != null)).map(([k, l, d]) => [k, l, d]), m, (v) => update({ color: v === "lift" ? null : v, bins: null }), "Color by")),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Compare" }),
        seg([["abs", "None"], hasParents ? ["parent", "vs parent", "Each candidate against the candidate it was built from"] : null, ["ref", "vs reference"]].filter(Boolean), cmp, (v) => update({ cmp: v === "abs" ? null : v, bins: null }), "Compare"),
        cmp === "ref" ? h("select", { "aria-label": "Reference candidate", onchange: (e) => update({ ref: e.target.value }) },
          all.map((c) => h("option", { value: c.name, text: c.baseline ? `${c.name} (baseline)` : c.name, selected: c.name === refName }))) : null),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Scale" }), seg([["auto", `Auto (${2 ** R}×)`], ["1", "2×"], ["2", "4×"], ["3", "8×"]], st.range, (v) => update({ range: v === "auto" ? null : v, bins: null }), "Scale")),
      h("div", { class: "ctl" }, checkbox("Texture", st.tex, (on) => update({ tex: on ? "1" : null }), "Hatch cells by direction and size, for print and color-vision deficiency")),
      h("div", { style: { flex: 1 } }),
      h("div", { class: "ctl" }, seg([["chart", "Chart"], ["table", "Table"]], st.view, (v) => update({ view: v === "chart" ? null : v }), "View")));
    const fsearch = h("input", { type: "search", placeholder: "Find section", value: st.q, "aria-label": "Find section" });
    let qt;
    fsearch.addEventListener("input", (e) => { clearTimeout(qt); qt = setTimeout(() => update({ q: e.target.value }), 180); });
    const controls2 = h("div", { class: "controls" },
      h("div", { class: "ctl" }, fsearch),
      h("div", { class: "ctl" }, groupChips(secs, st.hide, (hide) => update({ hide }))),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Rows" }),
        seg([["order", "Frequency"], ["change", cmp === "abs" ? "Variation" : "Change"], ["mean", "Mean"], ["name", "Name"]], st.sort, (v) => update({ sort: v === "order" ? null : v }), "Sort rows"),
        checkbox("Grouped", st.grouped, (on) => update({ grouped: on ? null : "0" }))),
      zoomed
        ? h("button", { class: "chip x", title: "Show every iteration", onclick: () => update({ i0: null, i1: null }) }, `iterations ${st.i0 ?? "…"}–${st.i1 ?? "…"}`)
        : h("span", { class: "muted", style: { fontSize: "12px" }, text: "Drag across a chart or the iteration strip to zoom" }));

    // -- score charts --------------------------------------------------------------------------------------
    const showPanelNow = !!sel;
    split.classList.toggle("with-panel", showPanelNow);
    if (showPanelNow && !panel.isConnected) split.append(panel);
    if (!showPanelNow) panel.remove();

    const candTip = (c) => ({
      title: c.name,
      sub: c.baseline ? "baseline" : `iteration ${c.iteration}${c.axis ? ` · ${c.axis}` : ""}`,
      rows: [[fmtScore(c.score), "score"], c.outcome ? [c.outcome, "outcome"] : null, c.ctx_chars ? [fmt.int(c.ctx_chars), "context characters"] : null,
        c.parent ? [c.parent, "parent"] : null].filter(Boolean),
      quote: c.description ? `${c.description.slice(0, 220)}${c.description.length > 220 ? "…" : ""}` : null,
    });
    const chartW = (n) => Math.max(320, (content.clientWidth - 32 - (n - 1) * 16) / n - 26);
    const hasCtx = scored.some((c) => c.ctx_chars);
    const toggle = (k) => { const lh = new Set(st.lh); lh.has(k) ? lh.delete(k) : lh.add(k); update({ lh }); };
    const pick = (key) => update({ cand: key && key !== st.cand ? key : null });
    const hoverCand = (key) => {
      for (const ch of charts) ch.setHover(key);
      const ci = key ? cols.findIndex((c) => c.name === key) : -1;
      hm?.setHover(ci >= 0 ? { ri: null, ci } : null);
      thm?.setHover(ci >= 0 ? { ri: null, ci } : null);
    };
    const nCharts = hasCtx ? 2 : 1;
    const scoreChart = scatterChart({
      width: chartW(nCharts), title: "Score by iteration", xLabel: "iteration", integerX: true, xFmt: (v) => String(Math.round(v)),
      yFmt: fmtScore, frontier: "best",
      points: scored.map((c) => ({ key: c.name, x: c.iteration, y: c.score, label: c.name, order: c.order, tip: candTip(c) })),
      links: all.filter((c) => c.parent && byName.has(c.parent)).map((c) => [c.parent, c.name]),
      hidden: st.lh, onToggle: toggle, onHover: hoverCand, onClick: pick, sel: sel?.name,
      range: zoomed ? [st.i0 ?? -Infinity, st.i1 ?? Infinity] : null,
      onBrush: (a, b) => (a == null ? update({ i0: null, i1: null }) : update({ i0: Math.ceil(a), i1: Math.floor(b) })),
    });
    charts = [scoreChart];
    const chartCards = [h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", { text: "Score by iteration" }),
      h("span", { class: "muted", text: "Each dot is a candidate; the line is the best score so far." })), h("div", { class: "cc-body" }, scoreChart.el))];
    if (hasCtx) {
      const ctxChart = scatterChart({
        width: chartW(nCharts), title: "Score against context size", xLabel: "context characters (log scale)", xLog: true,
        xFmt: (v) => (v >= 1000 ? `${v / 1000}k` : String(v)), yFmt: fmtScore, frontier: "pareto",
        points: scored.filter((c) => c.ctx_chars && inRange(c)).map((c) => ({ key: c.name, x: c.ctx_chars, y: c.score, label: c.name, order: c.order, tip: candTip(c) })),
        hidden: st.lh, onToggle: toggle, onHover: hoverCand, onClick: pick, sel: sel?.name,
      });
      charts.push(ctxChart);
      chartCards.push(h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", { text: "Score against context size" }),
        h("span", { class: "muted", text: "The line joins candidates no other beats on both score and size." })), h("div", { class: "cc-body" }, ctxChart.el)));
    }
    const chartRow = h("div", { class: "card-row" }, chartCards);

    // -- reading heatmap -------------------------------------------------------------------------------------
    const cmpText = cmp === "parent" ? "against its parent" : cmp === "ref" ? `against ${refName}` : "";
    const mapCard = h("div", { class: "card" }, h("div", { class: "card-head" },
      h("h2", { text: `${LABEL[m]} by section and candidate${cmpText ? `, ${cmpText}` : ""}` }),
      h("span", { class: "muted", text: cmp === "abs" ? MEASURES.find((x) => x[0] === m)[2] : "Ratio of the candidate's value to the comparison's: red reads more, blue less." })));
    const smin = Math.min(...scored.map((c) => c.score)), smax = Math.max(...scored.map((c) => c.score));
    const cellTip = (r, c, cell) => {
      const title = r.kind === "section" ? r.name : `${r.label} · ${r.count} sections`;
      const sub = `${c.name}${c.baseline ? " (baseline)" : c.iteration != null ? ` · iteration ${c.iteration}` : ""}`;
      if (!digestOf.has(c.name)) return { title, sub: `${sub} · no digest for this candidate` };
      if (!cell) return { title, sub: `${sub} · not in this candidate's context` };
      if (cell.removed) return { title, sub: `${sub} · removed`, rows: [[fmt.x(cell.base), `${LABEL[m].toLowerCase()} in ${cell.other}`]] };
      const rowsT = [];
      if (cmp !== "abs") {
        if (cell.noRef) rowsT.push(["–", "no parent recorded"]);
        else if (cell.added) rowsT.push(["new", `not in ${cell.other}`]);
        else rowsT.push([fmt.x(cell.v), `of ${cell.other}'s ${LABEL[m].toLowerCase()}`], [fmt.x(cell.base), `in ${cell.other}`]);
      }
      rowsT.push([fmt.x(cell[m]), LABEL[m].toLowerCase()]);
      for (const [k] of MEASURES) if (k !== m && cell[k] != null) rowsT.push([fmt.x(cell[k]), LABEL[k].toLowerCase()]);
      rowsT.push([fmt.int(cell.tokens), "tokens"]);
      if (cell.instances > 1) rowsT.push([String(cell.instances), "instances merged"]);
      return { title, sub, rows: rowsT };
    };
    const openStore = (c, section) => {
      const ss = data.stores[c.name] || [];
      if (!ss.length) return;
      const ds = section && ss.length > 1 ? section.split(":")[0] : null;
      const [sid] = ss.find(([, d]) => d && d === ds) || ss[0];
      const local = section && ds && section.startsWith(`${ds}:`) ? section.slice(ds.length + 1) : section;
      go("/store", { store: sid, sel: local || null, cols: local ? "section" : null });
    };
    const selCi = sel ? cols.findIndex((c) => c.name === sel.name) : -1;
    const secRi = st.sec ? rows.findIndex((r) => r.key === `s:${st.sec}`) : -1;
    // both heatmaps share the label width so their candidate columns line up
    const taskNames = [...new Set(all.flatMap((c) => Object.keys(c.per_task || {})))];
    const labelW = Math.max(labelWidth(rows), labelWidth(taskNames.map((n) => ({ label: n, depth: 0 }))));

    if (!rows.length || !cols.length) {
      hm = null;
      mapCard.append(h("div", { class: "empty", text: st.q ? `No section matches “${st.q}”.` : "No digests in this run yet." }));
    } else if (st.view === "table") {
      hm = null;
      const recs = [];
      for (const r of rows.filter((x) => x.kind === "section" || x.collapsed)) for (const c of cols) {
        const cell = cellOf(r, c);
        if (cell) recs.push({ row: r, c, section: r.kind === "section" ? r.name : `${r.label} (${r.count})`, candidate: c.name, iteration: c.iteration, score: c.score, v: cell.v, value: cell[m], tokens: cell.tokens, status: cell.removed ? "removed" : cell.added ? "new" : "" });
      }
      mapCard.append(table([
        { key: "section", label: "Section", cls: "wrap", render: (x) => h("span", null, h("span", { class: "swatch", style: { background: ratioColor(x.v, R) || "transparent" } }), x.section) },
        { key: "candidate", label: "Candidate", cls: "wrap" },
        { key: "iteration", label: "Iteration", num: true },
        { key: "score", label: "Score", num: true, fmt: fmtScore },
        { key: "value", label: LABEL[m], num: true, fmt: fmt.x },
        cmp !== "abs" ? { key: "v", label: "Ratio", num: true, fmt: fmt.x } : null,
        { key: "tokens", label: "Tokens", num: true, fmt: fmt.int },
        cmp !== "abs" ? { key: "status", label: "Status" } : null,
      ].filter(Boolean), recs.filter((x) => !activeBins.size || activeBins.has(binOf(bins, x.v))), {
        sort: st.tsort, onSort: (v) => update({ tsort: v }), onRow: (x) => update({ cand: x.c.name, sec: x.row.kind === "section" ? x.row.name : null }),
      }));
    } else {
      hm = heatmap({
        rows, ncols: cols.length, label: "Section reading by candidate", q: st.q, texture: st.tex, marks: false, labelW,
        avail: content.clientWidth - 34 || 900,
        corner: h("div", { style: { display: "flex", flexDirection: "column", gap: "4px", width: "100%", textAlign: "right" } }, h("span", { text: "iteration" }), h("span", { text: "score ↑" })),
        meanTitle: st.sort === "mean" ? "mean ↓" : "mean",
        pinned: secRi >= 0 || selCi >= 0 ? { ri: secRi >= 0 ? secRi : null, ci: selCi >= 0 ? selCi : null } : null,
        strip: {
          height: 54,
          draw(cx, { cw, W, hl, ink, ink2, muted, grid }) {
            cx.font = "10.5px system-ui, sans-serif";
            cx.textAlign = "center";
            const every = Math.max(1, Math.ceil(22 / cw));
            cols.forEach((c, i) => {
              const x = i * cw;
              if (i === hl || i === selCi) { cx.fillStyle = grid; cx.fillRect(x, 0, cw, 54); }
              cx.fillStyle = i === hl ? ink : muted;
              if (i % every === 0 || i === hl) cx.fillText(c.baseline ? "B" : String(c.iteration ?? ""), x + cw / 2, 10);
              if (c.score != null) {
                const hgt = 6 + (smax > smin ? (c.score - smin) / (smax - smin) : 1) * 30;
                cx.fillStyle = c === best ? ink : ink2;
                cx.globalAlpha = i === hl || c === best ? 0.95 : 0.5;
                cx.fillRect(x + 1, 52 - hgt, Math.max(1, cw - 2), hgt);
                cx.globalAlpha = 1;
              } else {
                cx.strokeStyle = muted;
                cx.strokeRect(x + cw / 2 - 3.5, 44, 7, 7);
              }
            });
            cx.fillStyle = grid;
            cx.fillRect(0, 14, W, 1);
          },
        },
        cell(ri, ci) {
          const cell = cellOf(rows[ri], cols[ci]);
          if (!cell) return null;
          if (cell.removed) return { fill: null, mark: "x", faded: activeBins.size > 0 || !!hoverBin };
          if (cell.noRef) return null;
          const t = scalePos(cell.v, R), b = binOf(bins, cell.v);
          return { fill: divergingColor(t), t, faded: (activeBins.size && !activeBins.has(b)) || (hoverBin && b !== hoverBin) };
        },
        mean(ri) { const x = statOf(rows[ri]); return x ? { t: scalePos(x.mean, R) } : null; },
        onHover({ ri, ci, ev, strip, mean: isMean, label }) {
          if (strip) { tip(ev, candTip(cols[ci])); for (const ch of charts) ch.setHover(cols[ci].name); thm?.setHover({ ri: null, ci }); return; }
          const r = rows[ri];
          if (isMean || label) {
            const x = statOf(r);
            tip(ev, {
              title: r.kind === "section" ? r.name : `${r.label} · ${r.count} sections`,
              sub: r.kind === "section" ? groupLabel(r.group) : "union of the visible sections",
              rows: [[fmt.x(x?.mean), `geometric mean${cmp === "abs" ? "" : " ratio"} over candidates`], [String(x?.n ?? 0), `of ${cols.length} candidates have it`],
                r.kind === "section" ? [String(secBy.get(r.name)?.candidates ?? 0), "candidates in the run have it"] : null].filter(Boolean),
            });
            return;
          }
          tip(ev, cellTip(r, cols[ci], cellOf(r, cols[ci])));
          for (const ch of charts) ch.setHover(cols[ci].name);
          thm?.setHover({ ri: null, ci });
        },
        onLeave() { tip(null); for (const ch of charts) ch.setHover(null); thm?.setHover(null); },
        onClick({ ri, ci }) {
          const r = rows[ri];
          update({ cand: cols[ci].name, sec: r.kind === "section" ? r.name : st.sec });
          hm?.el.focus({ preventScroll: true });
        },
        onDblClick({ ri, ci }) { const r = rows[ri]; openStore(cols[ci], r.kind === "section" ? r.name : null); },
        onRowClick(ri) { const r = rows[ri]; if (r.kind === "section") update({ sec: st.sec === r.name ? null : r.name }); },
        onToggle(r) { const fold = new Set(st.fold); fold.has(r.key) ? fold.delete(r.key) : fold.add(r.key); update({ fold: fold.size ? fold : "-" }); },
        onBrush(a, b) {
          if (a == null) { update({ i0: null, i1: null }); return; }
          const its = cols.slice(a, b + 1).map((c) => c.iteration).filter((v) => v != null);
          if (its.length) update({ i0: Math.min(...its), i1: Math.max(...its) });
        },
        onColClick(ci) { pick(cols[ci].name); },
        onMeanClick() { update({ sort: st.sort === "mean" ? null : "mean" }); },
        onKey(d) {
          if (!d) { update({ cand: null }); return; }
          const secRows = rows.map((r, i) => (r.kind === "section" ? i : -1)).filter((i) => i >= 0);
          const cur = Math.max(0, secRows.indexOf(secRi));
          const ri = secRows[Math.max(0, Math.min(secRows.length - 1, cur + d[0]))];
          const ci = Math.max(0, Math.min(cols.length - 1, (selCi < 0 ? 0 : selCi) + d[1]));
          update({ sec: rows[ri].name, cand: cols[ci].name });
          hm?.el.focus({ preventScroll: true });
        },
      });
      mapCard.append(hm.el, legend({
        title: cmp === "abs" ? LABEL[m] : `${LABEL[m]} ratio`, range: R, active: activeBins,
        onToggle(id, add) { const next = new Set(add ? activeBins : []); if (activeBins.has(id) && (add || activeBins.size === 1)) next.delete(id); else next.add(id); update({ bins: next }); },
        onHover(id) { hoverBin = id; hm.redraw(); },
        extra: [
          cmp !== "abs" ? h("span", { class: "key" }, h("span", { class: "swatch", style: { border: "1px solid var(--muted)", background: "transparent" } }), "new") : null,
          cmp !== "abs" ? h("span", { class: "key" }, xIcon(), "removed") : null,
          h("span", { class: "key muted", text: "Click a cell to select; double-click to open the candidate's store." }),
          data.matrix.cells.some((row) => row[F.indexOf("instances")] > 1) ? h("span", { class: "key muted", text: "Numbered instances (demo:label#n) are merged per name." }) : null,
        ],
      }));
      if (secRi >= 0 || selCi >= 0) hm.setPin({ ri: secRi >= 0 ? secRi : null, ci: selCi >= 0 ? selCi : null });
    }

    // -- per-task scores --------------------------------------------------------------------------------------
    const taskCard = renderTasks();

    content.replaceChildren(head, controls1, controls2, chartRow, mapCard, ...(taskCard ? [taskCard] : []));
    renderPanel();

    function renderTasks() {
      const withTasks = all.filter((c) => c.per_task);
      if (!withTasks.length) { thm = null; return null; }
      const tasks = [...new Set(withTasks.flatMap((c) => Object.keys(c.per_task)))];
      const pass = (task, c) => c.per_task?.[task] ?? null;
      const tsecs = tasks.map((t) => ({ name: t, group: "", parent: "", conversation: false }));
      const tq = st.tq.toLowerCase();
      const meanOf = (t) => { const v = cols.map((c) => pass(t, c)).filter((x) => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
      const spreadOf = (t) => { const v = cols.map((c) => pass(t, c)).filter((x) => x != null); return v.length ? Math.max(...v) - Math.min(...v) : null; };
      const trows = buildRows(tsecs, { fold: new Set(), hide: new Set(), q: tq, sort: st.trows, grouped: false, stats: { mean: meanOf, spread: spreadOf } });
      const tbActive = new Set([...st.tbins].filter((b) => TASK_BINS.some((x) => x.id === b)));
      const tbinOf = (v) => (v == null ? null : TASK_BINS.find((b) => v >= b.lo && v < b.hi)?.id);
      const card = h("div", { class: "card" }, h("div", { class: "card-head" },
        h("h2", { text: "Score by task and candidate" }),
        h("span", { class: "muted", text: cmp === "abs" ? "Mean score of the candidate's trials on each task." : `Dot: higher than ${cmp === "parent" ? "the parent" : refName}; ring: lower.` }),
        h("span", { class: "spacer" }),
        (() => { const i = h("input", { type: "search", placeholder: "Find task", value: st.tq, "aria-label": "Find task" }); let tt; i.addEventListener("input", (e) => { clearTimeout(tt); tt = setTimeout(() => update({ tq: e.target.value }), 180); }); return i; })(),
        seg([["name", "Name"], ["mean", "Mean"], ["spread", "Spread"]], st.trows, (v) => update({ trows: v === "name" ? null : v }), "Sort tasks")));
      const tSelRi = st.task ? trows.findIndex((r) => r.name === st.task) : -1;
      const openTask = (c, task) => {
        const ss = data.stores[c.name] || [];
        if (ss.length) go("/episode", { store: ss[0][0], eq: task });
      };
      if (st.view === "table") {
        thm = null;
        const recs = [];
        for (const r of trows) for (const c of cols) {
          const v = pass(r.name, c);
          if (v == null) continue;
          const o = against(c), b = o ? pass(r.name, byName.get(o)) : null;
          recs.push({ c, task: r.name, candidate: c.name, iteration: c.iteration, v, ref: b, d: b != null ? v - b : null });
        }
        card.append(table([
          { key: "task", label: "Task", cls: "wrap", render: (x) => h("span", null, h("span", { class: "swatch", style: { background: sequentialColor(x.v) } }), x.task) },
          { key: "candidate", label: "Candidate", cls: "wrap" },
          { key: "iteration", label: "Iteration", num: true },
          { key: "v", label: "Score", num: true, fmt: (v) => fmt.num(v) },
          cmp !== "abs" ? { key: "d", label: "Change", num: true, fmt: (v) => (v == null ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(2)}`) } : null,
        ].filter(Boolean), recs.filter((x) => !tbActive.size || tbActive.has(tbinOf(x.v))), {
          sort: st.ttsort, onSort: (v) => update({ ttsort: v }), onRow: (x) => update({ cand: x.c.name, task: x.task }),
        }));
        return card;
      }
      if (!trows.length) { thm = null; card.append(h("div", { class: "empty", text: "No task matches." })); return card; }
      thm = heatmap({
        rows: trows, ncols: cols.length, label: "Score by task and candidate", q: st.tq, marks: true, labelW,
        avail: content.clientWidth - 34 || 900, corner: h("span", { text: "iteration" }), meanTitle: st.trows === "mean" ? "mean ↓" : "mean",
        strip: {
          height: 18,
          draw(cx, { cw, hl, ink, muted, grid }) {
            cx.font = "10.5px system-ui, sans-serif";
            cx.textAlign = "center";
            const every = Math.max(1, Math.ceil(22 / cw));
            cols.forEach((c, i) => {
              if (i === hl || i === selCi) { cx.fillStyle = grid; cx.fillRect(i * cw, 0, cw, 18); }
              cx.fillStyle = i === hl ? ink : muted;
              if (i % every === 0 || i === hl) cx.fillText(c.baseline ? "B" : String(c.iteration ?? ""), i * cw + cw / 2, 12);
            });
          },
        },
        onColClick(ci) { pick(cols[ci].name); },
        pinned: tSelRi >= 0 || selCi >= 0 ? { ri: tSelRi >= 0 ? tSelRi : null, ci: selCi >= 0 ? selCi : null } : null,
        cell(ri, ci) {
          const c = cols[ci], v = pass(trows[ri].name, c);
          if (v == null) return null;
          const o = against(c), b = o ? pass(trows[ri].name, byName.get(o)) : null;
          const bin = tbinOf(v);
          return { fill: sequentialColor(v), t: null, mark: b == null || cmp === "abs" ? null : v > b + 1e-9 ? "high" : v < b - 1e-9 ? "low" : null,
            faded: (tbActive.size && !tbActive.has(bin)) || (hoverTBin && bin !== hoverTBin) };
        },
        meanBars: true,
        mean(ri) { const v = meanOf(trows[ri].name); return v == null ? null : { bar: v, fill: sequentialColor(v) }; },
        onHover({ ri, ci, ev, mean: isMean, label }) {
          const task = trows[ri].name;
          if (isMean || label) { tip(ev, { title: task, rows: [[fmt.num(meanOf(task)), "mean over the shown candidates"], [fmt.num(spreadOf(task)), "spread"]] }); return; }
          const c = cols[ci], v = pass(task, c), o = against(c), b = o ? pass(task, byName.get(o)) : null;
          tip(ev, { title: task, sub: `${c.name}${c.iteration != null ? ` · iteration ${c.iteration}` : ""}`,
            rows: [[fmt.num(v), "score"], b != null && cmp !== "abs" ? [fmt.num(b), `in ${o}`] : null].filter(Boolean) });
          for (const ch of charts) ch.setHover(c.name);
          hm?.setHover({ ri: null, ci });
        },
        onLeave() { tip(null); for (const ch of charts) ch.setHover(null); hm?.setHover(null); },
        onClick({ ri, ci }) { update({ cand: cols[ci].name, task: trows[ri].name }); thm?.el.focus({ preventScroll: true }); },
        onDblClick({ ri, ci }) { openTask(cols[ci], trows[ri].name); },
        onRowClick(ri) { update({ task: st.task === trows[ri].name ? null : trows[ri].name }); },
        onMeanClick() { update({ trows: st.trows === "mean" ? null : "mean" }); },
      });
      card.append(thm.el, legend({
        title: "Score", range: 1, active: tbActive,
        bins: TASK_BINS.map((b, i) => ({ ...b, color: sequentialColor(i / (TASK_BINS.length - 1)) })),
        onToggle(id, add) { const next = new Set(add ? tbActive : []); if (tbActive.has(id) && (add || tbActive.size === 1)) next.delete(id); else next.add(id); update({ tbins: next }); },
        onHover(id) { hoverTBin = id; thm.redraw(); },
        extra: [
          cmp !== "abs" ? h("span", { class: "key" }, dotIcon(false), "higher") : null,
          cmp !== "abs" ? h("span", { class: "key" }, dotIcon(true), "lower") : null,
          h("span", { class: "key muted", text: "Double-click a cell to open that task's episodes." }),
        ],
      }));
      if (tSelRi >= 0 || selCi >= 0) thm.setPin({ ri: tSelRi >= 0 ? tSelRi : null, ci: selCi >= 0 ? selCi : null });
      return card;
    }

    // -- selected candidate -------------------------------------------------------------------------------------
    function renderPanel() {
      if (!sel) return;
      const c = sel;
      const other = against(c) || (cmp === "abs" ? (c.parent && byName.has(c.parent) ? c.parent : refName !== c.name ? refName : null) : null);
      const link = (name) => h("a", { href: href("/evolution", { ...route().params, cand: name }), text: name });
      const ss = data.stores[c.name] || [];
      const parentStores = other ? data.stores[other] || [] : [];
      const headEl = h("div", { class: "pane-head" },
        h("div", { class: "row" }, h("h3", { text: c.name }),
          h("button", { class: "iconbtn", title: "Clear the selection", "aria-label": "Close", onclick: () => update({ cand: null }) }, "×")),
        h("div", { class: "muted", text: c.baseline ? "baseline" : `iteration ${c.iteration}${c.axis ? ` · ${c.axis}` : ""}` }));
      const tiles = h("div", { class: "pane-sec" }, h("div", { class: "tiles" },
        tileEl("Score", fmtScore(c.score)),
        tileEl("Change", c.delta == null ? "–" : `${c.delta > 0 ? "+" : ""}${(scale === 1 ? c.delta * 100 : c.delta).toFixed(1)}`),
        c.ctx_chars ? tileEl("Context", fmt.int(c.ctx_chars)) : tileEl("Sections", String(digestOf.get(c.name)?.length ?? 0))));
      const lineage = h("div", { class: "pane-sec" }, h("h4", { text: "Lineage" }),
        h("div", { class: "kv" }, h("span", { class: "muted", text: "parent" }), c.parent ? link(c.parent) : h("span", { class: "muted", text: "not recorded" })),
        c.children.length ? h("div", { class: "kv" }, h("span", { class: "muted", text: "children" }), h("span", { class: "chips" }, c.children.map(link))) : null,
        c.components?.length ? h("div", { class: "kv" }, h("span", { class: "muted", text: "components" }), h("span", { class: "chips" }, c.components.map((x) => h("span", { class: "chip static", text: x })))) : null,
        h("div", { class: "kv" }, h("span", { class: "muted", text: "stores" }), ss.length
          ? h("span", { class: "chips" }, ss.map(([sid, ds]) => h("a", { class: "chip", href: href("/store", { store: sid }), text: ds || "store" })),
            parentStores.length ? ss.map(([sid, ds]) => {
              const [rid] = parentStores.find(([, d]) => d === ds) || parentStores[0];
              return h("a", { class: "chip", href: href("/monitor", { store: sid, ref: rid }), title: `Monitor this candidate's episodes against ${other}'s`, text: `monitor vs ${other}${ds ? ` (${ds})` : ""}` });
            }) : null)
          : h("span", { class: "muted", text: "none replayed" })));
      const text = c.description ? h("div", { class: "pane-sec" }, h("h4", { text: c.changes || c.evidence ? "Hypothesis" : "Description" }), h("div", { class: "prose", text: c.description }),
        c.changes ? h("details", null, h("summary", { text: "Changes" }), h("div", { class: "prose", text: c.changes })) : null,
        c.evidence ? h("details", null, h("summary", { text: "Evidence" }), h("div", { class: "prose", text: c.evidence })) : null) : null;

      // reading changes against the comparison candidate
      let changes = null;
      if (other && digestOf.has(c.name) && digestOf.has(other)) {
        const mine = new Map(digestOf.get(c.name).map((x) => [x.section, x]));
        const theirs = new Map(digestOf.get(other).map((x) => [x.section, x]));
        const both = [...mine.keys()].filter((n) => theirs.has(n) && mine.get(n)[m] > 0 && theirs.get(n)[m] > 0)
          .map((n) => ({ n, r: mine.get(n)[m] / theirs.get(n)[m], a: mine.get(n)[m], b: theirs.get(n)[m] }))
          .sort((x, y) => Math.abs(Math.log2(y.r)) - Math.abs(Math.log2(x.r)));
        const added = [...mine.keys()].filter((n) => !theirs.has(n)), removed = [...theirs.keys()].filter((n) => !mine.has(n));
        const rr = Math.max(1, ...both.slice(0, 12).map((x) => Math.abs(Math.log2(x.r))));
        changes = h("div", { class: "pane-sec" },
          h("h4", null, `${LABEL[m]} against ${other}`, h("span", { class: "spacer" }), h("span", { class: "muted", text: `${added.length} new · ${removed.length} removed` })),
          h("div", { class: "members" }, both.slice(0, 12).map((x) => h("div", {
            class: `member ${x.n === st.sec ? "on" : ""}`, title: `${x.n}: ${fmt.x(x.a)} vs ${fmt.x(x.b)}`, onclick: () => update({ sec: x.n }),
          }, h("span", { class: "nm", text: x.n }), h("span", { class: "num", style: { textAlign: "right" }, text: fmt.x(x.r) }), barEl(Math.log2(x.r) / rr)))),
          added.length ? h("div", { class: "muted small", text: `New: ${added.slice(0, 8).join(", ")}${added.length > 8 ? ` and ${added.length - 8} more` : ""}` }) : null,
          removed.length ? h("div", { class: "muted small", text: `Removed: ${removed.slice(0, 8).join(", ")}${removed.length > 8 ? ` and ${removed.length - 8} more` : ""}` }) : null);
      }

      // the selected section across candidates
      let across = null;
      if (st.sec) {
        const vs = cols.map((x) => ({ c: x, v: raw(st.sec, x.name)?.[m] ?? null }));
        across = h("div", { class: "pane-sec" }, h("h4", null, `${st.sec}`, h("span", { class: "spacer" }),
          h("button", { class: "iconbtn", title: "Clear the section", "aria-label": "Clear section", onclick: () => update({ sec: null }) }, "×")),
        h("div", { class: "muted small", text: `${LABEL[m]} in each candidate; click a bar to select it.` }),
        acrossChart(vs, c.name, (name) => update({ cand: name }), (name) => hoverCand(name)));
      }
      panel.replaceChildren(...[headEl, tiles, text, lineage, changes, across].filter(Boolean));
    }
  }

  async function load(keep) {
    if (!st.run) {
      const runs = ctx.sources?.runs || [];
      content.replaceChildren(h("div", { class: "empty" }, h("p", { text: runs.length ? "Choose a Meta-Harness run:" : "No Meta-Harness runs found under the roots." }),
        ...runs.map((r) => h("p", null, h("a", { href: href("/evolution", { run: r.id }), text: r.label })))));
      return;
    }
    if (keep) content.classList.add("loading");
    else content.replaceChildren(h("div", { class: "empty", text: "Reading the run…" }));
    try {
      data = await api(`/api/runs/${st.run}`);
    } catch (e) {
      if (!keep || !data) content.replaceChildren(h("div", { class: "empty", text: String(e) }));
      return;
    } finally {
      content.classList.remove("loading");
    }
    st = settings(route().params, data.matrix.sections, data.run.datasets);
    render();
  }

  let rt;
  const onResize = () => { clearTimeout(rt); rt = setTimeout(render, 150); };
  addEventListener("resize", onResize);
  load(false);

  return {
    refresh(changes) { if (changes.some((c) => c.store === `run:${st.run}` || c.store === "*")) load(true); },
    destroy() { removeEventListener("resize", onResize); tip(null); },
  };
}

function tileEl(k, v) {
  return h("div", { class: "tile" }, h("div", { class: "k", text: k }), h("div", { class: "v", text: v }));
}

function barEl(pos) {
  const p = Math.max(-1, Math.min(1, pos)), w = p * 42;
  return s("svg", { viewBox: "0 0 90 10", width: 90, height: 10, "aria-hidden": "true" },
    s("rect", { x: w >= 0 ? 45 : 45 + w, y: 1, width: Math.max(1.5, Math.abs(w)), height: 8, rx: 2, fill: divergingColor(p) }),
    s("rect", { x: 44.5, y: 0, width: 1, height: 10, fill: "var(--ink-2)" }));
}

function xIcon() {
  return s("svg", { width: 10, height: 10, viewBox: "0 0 10 10", "aria-hidden": "true" },
    s("path", { d: "M2 2L8 8M8 2L2 8", stroke: "var(--ink)", "stroke-width": 1.6, "stroke-linecap": "round" }));
}

function dotIcon(low) {
  return s("svg", { width: 10, height: 10, viewBox: "0 0 10 10", "aria-hidden": "true" },
    s("circle", { cx: 5, cy: 5, r: 3.5, fill: low ? "var(--surface)" : "var(--ink)", stroke: low ? "var(--ink)" : "var(--surface)", "stroke-width": 1.2 }));
}

// One section's value in each candidate: bars from 1x on a log2 scale.
function acrossChart(vs, selName, onPick, onHover) {
  const W = 372, H = 120, PAD = { l: 36, r: 6, t: 6, b: 14 };
  const ls = vs.map((x) => (x.v > 0 ? Math.log2(x.v) : null));
  const lim = Math.max(1, ...ls.filter((v) => v != null).map(Math.abs));
  const Y = (l) => PAD.t + (1 - (l + lim) / (2 * lim)) * (H - PAD.t - PAD.b);
  const bw = (W - PAD.l - PAD.r) / Math.max(1, vs.length);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "section across candidates", class: "across" });
  for (const e of [-lim, 0, lim]) {
    const v = Math.round(e);
    svg.append(s("line", { class: e === 0 ? "base" : "gridl", x1: PAD.l, x2: W - PAD.r, y1: Y(v), y2: Y(v) }),
      s("text", { class: "axt", x: PAD.l - 4, y: Y(v) + 3.5, "text-anchor": "end" }, v >= 0 ? `${2 ** v}×` : `1/${2 ** -v}×`));
  }
  vs.forEach((x, i) => {
    const l = ls[i];
    const bx = PAD.l + i * bw;
    if (l != null) {
      const y = Y(Math.max(-lim, Math.min(lim, l)));
      svg.append(s("rect", { x: bx + 1, y: Math.min(y, Y(0)), width: Math.max(1, bw - 2), height: Math.max(1, Math.abs(Y(0) - y)), rx: 1.5,
        fill: divergingColor(Math.max(-1, Math.min(1, l / lim))), stroke: x.c.name === selName ? "var(--ink)" : "none", "stroke-width": 1.5 }));
    } else {
      svg.append(s("text", { class: "axt", x: bx + bw / 2, y: Y(0) - 3, "text-anchor": "middle" }, "·"));
    }
    const hit = s("rect", { x: bx, y: 0, width: bw, height: H, fill: "transparent", style: "cursor: pointer" });
    hit.addEventListener("pointermove", (ev) => { tip(ev, { title: x.c.name, sub: x.c.baseline ? "baseline" : `iteration ${x.c.iteration}`, rows: [[fmt.x(x.v), "value"]] }); onHover(x.c.name); });
    hit.addEventListener("pointerleave", () => { tip(null); onHover(null); });
    hit.addEventListener("click", () => onPick(x.c.name));
    svg.append(hit);
  });
  return h("div", { class: "lc" }, svg);
}
