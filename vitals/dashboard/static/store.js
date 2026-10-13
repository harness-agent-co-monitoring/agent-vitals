// Store overview: sections x episodes, each cell the episode's mean reading of the section. Columns carry
// the episode's outcome; clicking a cell opens that episode's read map pinned on the section.

import { divergingColor, ratioColor, scalePos } from "./color.js";
import { binOf, binsFor, heatmap, legend, outcomeMark } from "./heatmap.js";
import { aggregate, buildRows, defaultFold, groupChips } from "./rows.js";
import { api, checkbox, fmt, go, h, href, list, route, seg, setParams, table, tip } from "./util.js";

const MEASURES = [
  ["lift", "Read lift", "Geometric mean over the episode's turns of the section's read lift."],
  ["consult_ratio", "Consult ratio", "Geometric mean over the episode's turns of the section's consult ratio."],
];
const LABEL = Object.fromEntries(MEASURES.map(([k, l]) => [k, l]));

function settings(p, sections) {
  return {
    store: p.store,
    color: LABEL[p.color] ? p.color : "lift",
    range: p.range || "auto",
    hide: new Set(list(p.hide)),
    q: p.q || "",
    fold: p.fold == null ? defaultFold(sections) : new Set(list(p.fold).filter((k) => k !== "-")),
    sort: p.sort || "order",
    grouped: p.grouped !== "0",
    bins: new Set(list(p.bins)),
    view: p.view === "table" ? "table" : "chart",
    tex: p.tex === "1",
    cols: p.cols || "created",
    sel: p.sel || null,
    tsort: p.tsort || "",
  };
}

export function storeView(root, ctx) {
  let data = null, hm = null, hoverBin = null, timer = null;
  let st = settings(route().params, []);
  const content = h("div", { class: "content" });
  root.replaceChildren(content);

  function update(patch, rerender = true) {
    setParams(Object.fromEntries(Object.entries(patch).map(([k, v]) => [k, v instanceof Set ? [...v].join(",") || null : v])));
    st = settings(route().params, data?.sections || []);
    if (rerender) render();
  }

  function render() {
    if (!data) return;
    const m = st.color;
    const cellsBy = new Map();
    for (const c of data.cells) {
      const o = Object.fromEntries(data.fields.map((k, i) => [k, c[i]]));
      cellsBy.set(`${o.section}\u0000${o.episode}`, o);
    }
    const memo = new Map();
    const cellAt = (row, ep) => {
      const k = `${row.key}\u0000${ep}`;
      if (!memo.has(k)) {
        const ms = row.members.map((n) => cellsBy.get(`${n}\u0000${ep}`)).filter(Boolean);
        memo.set(k, row.kind === "section" ? ms[0] || null : aggregate(ms));
      }
      return memo.get(k);
    };

    let eps = [...data.episodes];
    const selRow = st.sel ? { key: `s:${st.sel}`, kind: "section", members: [st.sel] } : null;
    const colKey = {
      created: null,
      reward: (e) => -(e.reward ?? -1e9),
      name: (e) => e.episode,
      section: selRow ? (e) => -(cellAt(selRow, e.episode)?.[m] ?? -1e9) : null,
    }[st.cols];
    if (colKey) eps.sort((a, b) => { const x = colKey(a), y = colKey(b); return x < y ? -1 : x > y ? 1 : 0; });
    const stat = new Map();
    const statOf = (row) => {
      if (!stat.has(row.key)) {
        const ls = eps.map((e) => cellAt(row, e.episode)?.[m]).filter((v) => v > 0).map(Math.log2);
        stat.set(row.key, ls.length ? { mean: 2 ** (ls.reduce((a, b) => a + b, 0) / ls.length), n: ls.length } : null);
      }
      return stat.get(row.key);
    };
    const rows = buildRows(data.sections, {
      fold: st.fold, hide: st.hide, q: st.q, sort: st.sort, grouped: st.grouped,
      stat: (n) => statOf({ key: `s:${n}`, kind: "section", members: [n] })?.mean ?? null,
      tokens: (n) => Math.max(0, ...eps.map((e) => cellsBy.get(`${n}\u0000${e.episode}`)?.tokens || 0)),
    });
    const vals = [];
    for (const r of rows) for (const e of eps) { const v = cellAt(r, e.episode)?.[m]; if (v > 0) vals.push(Math.abs(Math.log2(v))); }
    vals.sort((a, b) => a - b);
    const p90 = vals.length ? vals[Math.floor(vals.length * 0.9)] : 1;
    const R = st.range === "auto" ? Math.max(1, Math.min(3, Math.ceil(p90))) : Number(st.range);
    const bins = binsFor(R);
    const active = new Set([...st.bins].filter((b) => bins.some((x) => x.id === b)));

    const info = ctx.sources?.stores.find((s) => s.id === st.store);
    const rewards = eps.map((e) => e.reward).filter((r) => r != null);
    const head = h("div", { class: "head" },
      h("h1", { text: info?.label || "Store" }),
      h("div", { class: "facts" },
        h("span", null, h("b", { text: String(data.progress.done) }), ` of ${data.progress.total} replayed episodes`),
        rewards.length ? h("span", null, "pass rate ", h("b", { text: fmt.pct(rewards.filter((r) => r > 0).length / rewards.length) })) : null,
        h("span", { class: "muted mono", text: info?.path || "" })),
      h("a", { class: "iconbtn linkbtn", href: href("/monitor", { store: st.store }), title: "Check this store's episodes against a reference store" }, "Monitor →"),
      data.progress.done < data.progress.total
        ? h("span", { class: "progress" }, h("span", { class: "bar" }, h("i", { style: { width: `${(100 * data.progress.done) / data.progress.total}%` } })), "computing read maps…")
        : null);

    const search = h("input", { type: "search", placeholder: "Find section", value: st.q, "aria-label": "Find section" });
    let qt;
    search.addEventListener("input", (e) => { clearTimeout(qt); qt = setTimeout(() => update({ q: e.target.value }), 180); });
    const controls = h("div", { class: "controls" },
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Color" }), seg(MEASURES.map(([k, l, d]) => [k, l, d]), m, (v) => update({ color: v === "lift" ? null : v, bins: null }), "Color by")),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Scale" }), seg([["auto", `Auto (${2 ** R}×)`], ["1", "2×"], ["2", "4×"], ["3", "8×"]], st.range, (v) => update({ range: v === "auto" ? null : v, bins: null }), "Scale")),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Episodes" }), seg([["created", "Recorded"], ["reward", "Reward"], ["name", "Name"], ["section", selRow ? "By selected section" : "By section (click a row)"]], st.cols, (v) => update({ cols: v === "created" ? null : v }), "Order episodes")),
      h("div", { class: "ctl" }, checkbox("Texture", st.tex, (on) => update({ tex: on ? "1" : null }))),
      h("div", { style: { flex: 1 } }),
      h("div", { class: "ctl" }, seg([["chart", "Chart"], ["table", "Table"]], st.view, (v) => update({ view: v === "chart" ? null : v }), "View")));
    const controls2 = h("div", { class: "controls" },
      h("div", { class: "ctl" }, search),
      h("div", { class: "ctl" }, groupChips(data.sections, st.hide, (hide) => update({ hide }))),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Rows" }),
        seg([["order", "Frequency"], ["mean", "Mean"], ["tokens", "Size"], ["name", "Name"]], st.sort, (v) => update({ sort: v === "order" ? null : v }), "Sort rows"),
        checkbox("Grouped", st.grouped, (on) => update({ grouped: on ? null : "0" }))));

    const card = h("div", { class: "card" }, h("div", { class: "card-head" },
      h("h2", { text: `${LABEL[m]} by section and episode` }), h("span", { class: "muted", text: MEASURES.find((x) => x[0] === m)[2] })));
    const open = (ep, row) => {
      const { params } = route();
      const keep = Object.fromEntries(["store", "range", "hide", "tex"].map((k) => [k, params[k]]));
      go("/episode", { ...keep, ep, pin: row && row.kind === "section" ? `last~${row.key}` : null });
    };
    if (!rows.length || !eps.length) {
      card.append(h("div", { class: "empty", text: data.progress.total ? "Nothing to show yet." : "No replayed episodes in this store." }));
    } else if (st.view === "table") {
      const recs = [];
      for (const r of rows.filter((x) => x.kind === "section" || x.collapsed)) for (const e of eps) {
        const c = cellAt(r, e.episode);
        if (c) recs.push({ ...c, row: r, section: r.kind === "section" ? r.name : `${r.label} (${r.count})`, episode: e.episode, reward: e.reward });
      }
      card.append(table([
        { key: "section", label: "Section", cls: "wrap", render: (x) => h("span", null, h("span", { class: "swatch", style: { background: ratioColor(x[m], R) || "transparent" } }), x.section) },
        { key: "episode", label: "Episode", cls: "wrap" },
        { key: "reward", label: "Reward", num: true, fmt: (v) => fmt.num(v) },
        { key: "tokens", label: "Tokens", num: true, fmt: fmt.int },
        { key: "lift", label: "Read lift", num: true, fmt: fmt.x },
        { key: "consult_ratio", label: "Consult ratio", num: true, fmt: fmt.x },
      ], recs.filter((x) => !active.size || active.has(binOf(bins, x[m]))), {
        sort: st.tsort, onSort: (v) => update({ tsort: v }), onRow: (x) => open(x.episode, x.row),
      }));
    } else {
      const Tmax = Math.max(1, ...eps.map((e) => e.turns || 0));
      hm = heatmap({
        rows, ncols: eps.length, label: `${LABEL[m]} by section and episode`, q: st.q, texture: st.tex, marks: false,
        avail: content.clientWidth - 34 || 900,
        corner: h("div", { style: { display: "flex", flexDirection: "column", gap: "3px", width: "100%", textAlign: "right" } }, h("span", { text: "outcome" }), h("span", { text: "turns ↓" })),
        meanTitle: st.sort === "mean" ? "mean ↓" : "mean",
        pinned: selRow ? { ri: rows.findIndex((r) => r.key === selRow.key), ci: null } : null,
        strip: {
          height: 40,
          draw(c, { cw, W, hl, ink2, grid }) {
            eps.forEach((e, i) => {
              const x = i * cw;
              if (i === hl) { c.fillStyle = grid; c.fillRect(x, 0, cw, 40); }
              outcomeMark(c, x, cw, 4, e.reward);
              c.fillStyle = ink2;
              c.globalAlpha = i === hl ? 0.9 : 0.45;
              c.fillRect(x + 1, 20, Math.max(1, cw - 2), ((e.turns || 0) / Tmax) * 18);
              c.globalAlpha = 1;
            });
            c.fillStyle = grid;
            c.fillRect(0, 18, W, 1);
          },
        },
        cell(ri, ci) {
          const c = cellAt(rows[ri], eps[ci].episode);
          if (!c) return null;
          const v = c[m], t = scalePos(v, R), b = binOf(bins, v);
          return { fill: divergingColor(t), t, faded: (active.size && !active.has(b)) || (hoverBin && b !== hoverBin) };
        },
        mean(ri) { const s = statOf(rows[ri]); return s ? { t: scalePos(s.mean, R) } : null; },
        onHover({ ri, ci, ev, strip, mean: isMean, label }) {
          if (strip) {
            const e = eps[ci];
            tip(ev, { title: e.task && !e.episode.includes(e.task) ? `${e.task} · ${e.episode}` : e.episode, rows: [[e.reward == null ? "–" : e.reward > 0 ? `pass (${e.reward})` : `fail (${e.reward})`, "outcome"], [String(e.turns), "turns"]] });
            return;
          }
          const r = rows[ri];
          if (isMean || label) {
            const s = statOf(r);
            tip(ev, { title: r.kind === "section" ? r.name : `${r.label} · ${r.count} sections`, rows: [[fmt.x(s?.mean), `geometric mean over episodes`], [String(s?.n ?? 0), `of ${eps.length} episodes present`]] });
            return;
          }
          const e = eps[ci], c = cellAt(r, e.episode);
          if (!c) { tip(ev, { title: r.kind === "section" ? r.name : r.label, sub: `${e.episode} · not in this episode` }); return; }
          tip(ev, {
            title: r.kind === "section" ? r.name : `${r.label} · ${c.n || 1} sections`,
            sub: `${e.episode} · ${e.reward == null ? "no reward" : e.reward > 0 ? "pass" : "fail"}`,
            rows: [[fmt.x(c[m]), LABEL[m].toLowerCase()], [fmt.x(c[m === "lift" ? "consult_ratio" : "lift"]), (m === "lift" ? LABEL.consult_ratio : LABEL.lift).toLowerCase()], [fmt.int(c.tokens), "tokens (median)"], [String(c.turns ?? "–"), "turns present"]],
          });
        },
        onLeave() { tip(null); },
        onClick({ ri, ci }) { if (cellAt(rows[ri], eps[ci].episode)) open(eps[ci].episode, rows[ri]); },
        onRowClick(ri) {
          const r = rows[ri];
          update(r.kind === "section" && st.sel !== r.name ? { sel: r.name, cols: "section" } : { sel: null, cols: st.cols === "section" ? null : st.cols });
        },
        onToggle(r) { const fold = new Set(st.fold); fold.has(r.key) ? fold.delete(r.key) : fold.add(r.key); update({ fold: fold.size ? fold : "-" }); },
        onColClick(ci) { open(eps[ci].episode, null); },
        onMeanClick() { update({ sort: st.sort === "mean" ? null : "mean" }); },
      });
      card.append(hm.el, legend({
        title: LABEL[m], range: R, active,
        onToggle(id, add) { const next = new Set(add ? active : []); if (active.has(id) && (add || active.size === 1)) next.delete(id); else next.add(id); update({ bins: next }); },
        onHover(id) { hoverBin = id; hm.redraw(); },
        extra: [h("span", { class: "key muted", text: "Click a row to order episodes by it; click a cell to open that episode." })],
      }));
    }
    content.replaceChildren(head, controls, controls2, card);
  }

  async function load(keep) {
    if (!st.store) {
      const stores = ctx.sources?.stores || [];
      content.replaceChildren(h("div", { class: "empty" }, h("p", { text: "Choose a trace store:" }),
        ...stores.map((s) => h("p", null, h("a", { href: `#/store?store=${s.id}`, text: s.label })))));
      return;
    }
    if (keep) content.classList.add("loading");
    else content.replaceChildren(h("div", { class: "empty", text: "Computing read maps…" }));
    try {
      data = await api(`/api/stores/${st.store}/matrix`);
    } catch (e) {
      content.replaceChildren(h("div", { class: "empty", text: String(e) }));
      return;
    } finally {
      content.classList.remove("loading");
    }
    st = settings(route().params, data.sections);
    render();
    clearTimeout(timer);
    if (data.progress.done < data.progress.total) timer = setTimeout(() => load(true), 2500);
  }
  let rt;
  const onResize = () => { clearTimeout(rt); rt = setTimeout(render, 150); };
  addEventListener("resize", onResize);
  load(false);

  return {
    refresh(changes) { if (changes.some((c) => c.store === st.store)) load(true); },
    destroy() { clearTimeout(timer); removeEventListener("resize", onResize); tip(null); },
  };
}
