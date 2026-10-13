// Episode read map: sections x turns, colored by one measure, with the episode list on the left and the
// pinned cell's details (section text, trajectory, turn) on the right.

import { divergingColor, inkOn, ratioColor, scalePos } from "./color.js";
import { binOf, binsFor, heatmap, legend } from "./heatmap.js";
import { lineChart } from "./linechart.js";
import { aggregate, buildRows, defaultFold, groupChips, groupLabel } from "./rows.js";
import { api, checkbox, fmt, go, h, list, liveDot, reward, route, seg, setParams, table, tip } from "./util.js";
import { whySection } from "./why.js";

export const MEASURES = [
  ["lift", "Read lift", "Attention per token over the context mean: 1× reads like an average context token."],
  ["peak", "Peak lift", "Lift of the eight most-read content tokens; the hottest line is shown in the tooltip. Sections under 32 tokens have none."],
  ["consult_ratio", "Consult ratio", "Whether the section's attention covaries with next-token uncertainty: above 1× it is read more at uncertain steps."],
  ["consult_contrast", "Consult contrast", "Attention at the most uncertain 10% of steps over the least uncertain 10%."],
];
const LABEL = Object.fromEntries(MEASURES.map(([k, l]) => [k, l]));

function settings(p, sections) {
  return {
    store: p.store, ep: p.ep,
    color: LABEL[p.color] ? p.color : "lift",
    range: p.range || "auto",
    marks: p.marks !== "0",
    ct: Number(p.ct) > 1 ? Number(p.ct) : 1.5,
    hide: new Set(list(p.hide)),
    q: p.q || "",
    t0: p.t0 != null ? Number(p.t0) : null,
    t1: p.t1 != null ? Number(p.t1) : null,
    pin: p.pin ? { turn: Number(p.pin.split("~")[0]), key: p.pin.split("~").slice(1).join("~") } : null,
    fold: p.fold == null ? defaultFold(sections) : new Set(list(p.fold).filter((k) => k !== "-")),
    sort: p.sort || "order",
    grouped: p.grouped !== "0",
    bins: new Set(list(p.bins)),
    view: p.view === "table" ? "table" : "chart",
    tex: p.tex === "1",
    esort: p.esort || "name",
    eq: p.eq || "",
    lh: new Set(list(p.lh)),
    tsort: p.tsort || "",
    follow: p.follow === "1",
  };
}

export function episodeView(root, ctx) {
  let eps = [];
  let data = null;
  let expl = new Map();
  let st = settings(route().params, []);
  let hm = null, chartA = null, chartB = null, hoverBin = null;
  let cellsBy = new Map();
  let rows = [], vturns = [], R = 2;

  const list_ = h("div", { class: "list" });
  const content = h("div", { class: "content" });
  const panel = h("div", { class: "panel" });
  const split = h("div", { class: "split" }, list_, content);
  root.replaceChildren(split);

  function update(patch, rerender = true) {
    setParams(Object.fromEntries(Object.entries(patch).map(([k, v]) => [k, v instanceof Set ? [...v].join(",") || null : v])));
    st = settings(route().params, data?.sections || []);
    if (rerender) render();
  }

  // -- episode list ---------------------------------------------------------------------------------------
  function renderList() {
    const stores = ctx.sources?.stores || [];
    const picker = h("select", {
      "aria-label": "Trace store",
      onchange: (e) => go("/episode", { store: e.target.value }),
    }, stores.map((s) => {
      const o = h("option", { value: s.id, text: `${s.label} (${s.episodes})` });
      if (s.id === st.store) o.selected = true;
      return o;
    }));
    const search = h("input", { type: "search", placeholder: "Filter episodes", value: st.eq, "aria-label": "Filter episodes" });
    search.addEventListener("input", (e) => { setParams({ eq: e.target.value }); st.eq = e.target.value; renderItems(); });
    const sorter = seg([["name", "Name"], ["reward", "Reward"], ["turns", "Turns"], ["updated", "Recent"]], st.esort,
      (v) => { update({ esort: v }, false); renderList(); }, "Sort episodes");
    const follow = checkbox("Follow newest", st.follow, (on) => { update({ follow: on ? "1" : null }, false); if (on) followNewest(); },
      "Open the most recently updated episode whenever the store changes");
    const body = h("div", { class: "list-body", role: "listbox", "aria-label": "Episodes" });
    list_.replaceChildren(h("div", { class: "list-head" }, picker, search, h("div", { class: "ctl" }, sorter), follow), body);

    function renderItems() {
      const q = st.eq.toLowerCase();
      const key = {
        name: (e) => e.episode, reward: (e) => -(e.reward ?? -1e9), turns: (e) => -e.turns, updated: (e) => -(e.updated || 0),
      }[st.esort];
      const shown = eps.filter((e) => !q || `${e.episode} ${e.task || ""}`.toLowerCase().includes(q))
        .sort((a, b) => { const x = key(a), y = key(b); return x < y ? -1 : x > y ? 1 : a.episode.localeCompare(b.episode, undefined, { numeric: true }); });
      body.replaceChildren(...shown.map((e) => h("div", {
        class: `item ${e.episode === st.ep ? "on" : ""}`, role: "option", "aria-selected": String(e.episode === st.ep), tabindex: "0",
        onclick: () => openEpisode(e.episode),
        onkeydown: (ev) => { if (ev.key === "Enter") openEpisode(e.episode); },
      },
      e.live ? h("span", { class: "dot live", title: "live" }) : h("span"),
      h("span", { class: "name", title: e.episode, text: e.task && !e.episode.includes(e.task) ? `${e.task} · ${e.episode}` : e.episode }),
      reward(e.reward),
      h("span", { class: "sub" }, h("span", { text: `${e.turns} turns` }),
        e.replayed ? null : h("span", { text: "not replayed" }),
        e.fault ? h("span", { text: `fault: ${e.fault}` }) : null,
        h("span", { text: fmt.ago(e.updated) })))));
      if (!shown.length) body.append(h("div", { class: "empty", text: eps.length ? "No episodes match." : "This store has no episodes yet." }));
    }
    renderItems();
  }

  function openEpisode(ep) {
    const { params } = route();
    delete params.pin; delete params.t0; delete params.t1;
    go("/episode", { ...params, ep });
  }

  function followNewest() {
    const newest = [...eps].filter((e) => e.replayed).sort((a, b) => (b.updated || 0) - (a.updated || 0))[0];
    if (newest && newest.episode !== st.ep) openEpisode(newest.episode);
  }

  // -- read map -------------------------------------------------------------------------------------------
  function prepare() {
    const F = data.fields;
    cellsBy = new Map();
    for (const c of data.cells) {
      const o = Object.fromEntries(F.map((k, i) => [k, c[i]]));
      cellsBy.set(`${o.section}\u0000${o.turn}`, o);
    }
  }

  function cellAt(row, turn, memo) {
    const k = `${row.key}\u0000${turn}`;
    if (memo.has(k)) return memo.get(k);
    const members = [];
    for (const n of row.members) {
      const c = cellsBy.get(`${n}\u0000${turn}`);
      if (c) members.push(c);
    }
    const out = row.kind === "section" ? members[0] || null : aggregate(members);
    memo.set(k, out);
    return out;
  }

  function render() {
    if (!data) return;
    const memo = new Map();
    const turns = data.turns.map((t) => t.turn);
    vturns = turns.filter((t) => (st.t0 == null || t >= st.t0) && (st.t1 == null || t <= st.t1));
    if (!vturns.length) vturns = turns;
    const turnInfo = new Map(data.turns.map((t) => [t.turn, t]));
    const m = st.color;
    if (st.pin && !Number.isFinite(st.pin.turn)) resolvePin(turns);

    const secStat = new Map();
    const statOf = (row) => {
      if (secStat.has(row.key)) return secStat.get(row.key);
      const ls = vturns.map((t) => cellAt(row, t, memo)?.[m]).filter((v) => v > 0).map(Math.log2);
      const out = ls.length ? { mean: 2 ** (ls.reduce((a, b) => a + b, 0) / ls.length), n: ls.length } : null;
      secStat.set(row.key, out);
      return out;
    };
    const tokOf = (name) => Math.max(0, ...vturns.map((t) => cellsBy.get(`${name}\u0000${t}`)?.tokens || 0));
    rows = buildRows(data.sections, {
      fold: st.fold, hide: st.hide, q: st.q, sort: st.sort, grouped: st.grouped,
      stat: (name) => statOf({ key: `s:${name}`, kind: "section", members: [name] })?.mean ?? null,
      tokens: tokOf,
    });

    const vals = [];
    for (const r of rows) for (const t of vturns) { const v = cellAt(r, t, memo)?.[m]; if (v > 0) vals.push(Math.abs(Math.log2(v))); }
    vals.sort((a, b) => a - b);
    const p90 = vals.length ? vals[Math.floor(vals.length * 0.9)] : 1;
    R = st.range === "auto" ? Math.max(1, Math.min(3, Math.ceil(p90))) : Number(st.range);
    const bins = binsFor(R);
    const activeBins = new Set([...st.bins].filter((b) => bins.some((x) => x.id === b)));

    // header
    const ep = data.episode;
    const L = data.turns.map((t) => t.L).filter(Boolean);
    const head = h("div", { class: "head" },
      h("h1", { text: ep.task && !ep.episode.includes(ep.task) ? `${ep.task} · ${ep.episode}` : ep.episode }),
      h("div", { class: "facts" },
        reward(ep.reward), liveDot(ep.live),
        h("span", null, h("b", { text: String(data.turns.length) }), " turns replayed"),
        L.length ? h("span", null, "context ", h("b", { text: fmt.int(Math.min(...L)) }), "–", h("b", { text: fmt.int(Math.max(...L)) }), " tokens") : null,
        h("span", null, h("b", { text: String(data.sections.filter((s) => !s.conversation).length) }), " declared sections"),
        h("span", { class: "muted", text: `updated ${fmt.ago(ep.updated)}` })));

    // controls
    const measure = seg(MEASURES.map(([k, l, d]) => [k, l, d]), m, (v) => update({ color: v, bins: null }), "Color by");
    const scale = seg([["auto", `Auto (${2 ** R}×)`, "Fit the scale to the 90th percentile of the shown cells"], ["1", "2×"], ["2", "4×"], ["3", "8×"]],
      st.range, (v) => update({ range: v === "auto" ? null : v, bins: null }), "Color scale range");
    const ctIn = h("input", { type: "range", min: "1.1", max: "3", step: "0.1", value: String(st.ct), "aria-label": "Consult mark threshold" });
    const ctOut = h("span", { class: "num ink2", text: `${st.ct.toFixed(1)}×` });
    ctIn.addEventListener("input", (e) => { ctOut.textContent = `${Number(e.target.value).toFixed(1)}×`; });
    ctIn.addEventListener("change", (e) => update({ ct: e.target.value }));
    const search = h("input", { type: "search", placeholder: "Find section", value: st.q, "aria-label": "Find section" });
    let qTimer;
    search.addEventListener("input", (e) => { clearTimeout(qTimer); qTimer = setTimeout(() => update({ q: e.target.value }), 180); });

    const chips = groupChips(data.sections, st.hide, (hide) => update({ hide }));

    const foldKeys = rows.filter((r) => r.kind !== "section").map((r) => r.key);
    const controls1 = h("div", { class: "controls" },
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Color" }), measure),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Scale" }), scale),
      h("div", { class: "ctl" }, checkbox("Consult marks", st.marks, (on) => update({ marks: on ? null : "0" }),
        "Dot: consult ratio at or above the threshold. Ring: at or below its inverse."), ctIn, ctOut),
      h("div", { class: "ctl" }, checkbox("Texture", st.tex, (on) => update({ tex: on ? "1" : null }), "Hatch cells by direction and size, for print and color-vision deficiency")),
      h("div", { class: "spacer", style: { flex: 1 } }),
      h("div", { class: "ctl" }, seg([["chart", "Chart"], ["table", "Table"]], st.view, (v) => update({ view: v === "chart" ? null : v }), "View")));
    const controls2 = h("div", { class: "controls" },
      h("div", { class: "ctl" }, search),
      h("div", { class: "ctl" }, chips),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Rows" }),
        seg([["order", "Prompt order"], ["mean", "Mean"], ["tokens", "Size"], ["name", "Name"]], st.sort, (v) => update({ sort: v === "order" ? null : v }), "Sort rows"),
        checkbox("Grouped", st.grouped, (on) => update({ grouped: on ? null : "0" })),
        st.grouped ? h("button", { class: "iconbtn", title: "Collapse every group", onclick: () => update({ fold: new Set([...foldKeys, ...data.sections.map((s) => `g:${s.group}`)]) }) }, "Collapse") : null,
        st.grouped ? h("button", { class: "iconbtn", title: "Expand every group", onclick: () => update({ fold: "-" }) }, "Expand") : null),
      st.t0 != null || st.t1 != null
        ? h("button", { class: "chip x", title: "Show all turns", onclick: () => update({ t0: null, t1: null }) }, `turns ${vturns[0]}–${vturns.at(-1)}`)
        : h("span", { class: "muted", style: { fontSize: "12px" }, text: "Drag across the turn strip to zoom" }));

    // chart or table
    const card = h("div", { class: "card" });
    const cardHead = h("div", { class: "card-head" },
      h("h2", { text: `${LABEL[m]} by section and turn` }),
      h("span", { class: "muted", text: MEASURES.find((x) => x[0] === m)[2] }));
    card.append(cardHead);

    const pinRi = st.pin ? rows.findIndex((r) => r.key === st.pin.key) : -1;
    const pinCi = st.pin ? vturns.indexOf(st.pin.turn) : -1;
    showPanel(pinRi >= 0);

    if (!rows.length) {
      card.append(h("div", { class: "empty", text: st.q ? `No section matches “${st.q}”.` : "No sections to show." }));
    } else if (st.view === "table") {
      const recs = [];
      for (const r of rows.filter((x) => x.kind === "section" || x.collapsed)) for (const t of vturns) {
        const c = cellAt(r, t, memo);
        if (c) recs.push({ ...c, row: r, section: r.kind === "section" ? r.name : `${r.label} (${r.count})`, turn: t });
      }
      card.append(table([
        { key: "section", label: "Section", cls: "wrap", render: (x) => h("span", null, h("span", { class: "swatch", style: { background: ratioColor(x[m], R) || "transparent" } }), x.section) },
        { key: "turn", label: "Turn", num: true },
        { key: "tokens", label: "Tokens", num: true, fmt: fmt.int },
        { key: "lift", label: "Read lift", num: true, fmt: fmt.x },
        { key: "peak", label: "Peak lift", num: true, fmt: fmt.x },
        { key: "consult_ratio", label: "Consult ratio", num: true, fmt: fmt.x },
        { key: "consult_contrast", label: "Consult contrast", num: true, fmt: fmt.x },
        { key: "hot_line", label: "Hottest line", cls: "wrap mono" },
      ], recs.filter((x) => !activeBins.size || activeBins.has(binOf(bins, x[m]))), {
        sort: st.tsort, onSort: (v) => update({ tsort: v }),
        onRow: (x) => update({ pin: `${x.turn}~${x.row.key}`, view: null }),
      }));
    } else {
      const strip = {
        height: 52,
        draw(c, { cw, W, H, hl, ink2, muted, grid }) {
          const Lmax = Math.max(1, ...vturns.map((t) => turnInfo.get(t)?.L || 0));
          const Gmax = Math.max(1, ...vturns.map((t) => turnInfo.get(t)?.n_gen || 0));
          c.fillStyle = grid;
          c.fillRect(0, 33, W, 1);
          c.font = "10.5px system-ui, sans-serif";
          c.textAlign = "center";
          const every = Math.max(1, Math.ceil(26 / cw));
          vturns.forEach((t, i) => {
            const ti = turnInfo.get(t) || {};
            const x = i * cw;
            if (i === hl) { c.fillStyle = grid; c.fillRect(x, 0, cw, H); }
            c.fillStyle = i === hl ? ink2 : muted;
            if (i % every === 0 || i === hl) c.fillText(String(t), x + cw / 2, 10);
            const bw = Math.max(1, cw - 2);
            const lh = ((ti.L || 0) / Lmax) * 17;
            c.fillStyle = ink2;
            c.globalAlpha = i === hl ? 0.9 : 0.45;
            c.fillRect(x + 1, 32 - lh, bw, lh);
            const gh = ((ti.n_gen || 0) / Gmax) * 16;
            c.fillRect(x + 1, 35, bw, gh);
            c.globalAlpha = 1;
          });
        },
      };
      hm = heatmap({
        rows, ncols: vturns.length, label: `${LABEL[m]} heatmap`, q: st.q,
        avail: content.clientWidth - 34 || 900,
        corner: h("div", { style: { display: "flex", flexDirection: "column", gap: "1px", width: "100%", textAlign: "right" } },
          h("span", { text: "turn" }), h("span", { text: "context ↑" }), h("span", { text: "generated ↓" })),
        meanTitle: st.sort === "mean" ? "mean ↓" : "mean",
        texture: st.tex, marks: st.marks,
        pinned: pinRi >= 0 ? { ri: pinRi, ci: pinCi >= 0 ? pinCi : null } : null,
        strip,
        cell(ri, ci) {
          const c = cellAt(rows[ri], vturns[ci], memo);
          if (!c) return null;
          const v = c[m];
          const t = scalePos(v, R);
          const b = binOf(bins, v);
          const faded = (activeBins.size && !activeBins.has(b)) || (hoverBin && b !== hoverBin);
          const cr = c.consult_ratio;
          const mark = cr == null ? null : cr >= st.ct ? "high" : cr <= 1 / st.ct ? "low" : null;
          return { fill: divergingColor(t), t, faded, mark };
        },
        mean(ri) {
          const s = statOf(rows[ri]);
          return s ? { t: scalePos(s.mean, R), v: s.mean } : null;
        },
        onHover({ ri, ci, ev, strip: isStrip, mean: isMean, label }) {
          if (isStrip) {
            const ti = turnInfo.get(vturns[ci]) || {};
            tip(ev, {
              title: `Turn ${vturns[ci]}`, sub: ti.action ? String(ti.action) : null,
              rows: [[fmt.int(ti.L), "context tokens"], [fmt.int(ti.n_gen), "generated tokens"], [fmt.num(ti.entropy), "mean entropy (nats)"]],
            });
            chartA?.setCross(vturns[ci]); chartB?.setCross(vturns[ci]);
            return;
          }
          const r = rows[ri];
          if (isMean || label) {
            const s = statOf(r);
            tip(ev, {
              title: r.kind === "section" ? r.name : `${r.label} · ${r.count} sections`,
              sub: r.kind === "section" ? groupLabel(r.group) : "union of the visible sections",
              rows: [[fmt.x(s?.mean), `geometric mean ${LABEL[m].toLowerCase()}`], [String(s?.n ?? 0), `of ${vturns.length} turns present`]],
            });
            return;
          }
          const c = cellAt(r, vturns[ci], memo);
          if (!c) { tip(ev, { title: r.kind === "section" ? r.name : r.label, sub: `Turn ${vturns[ci]} · not in the context` }); return; }
          const order = [m, ...MEASURES.map((x) => x[0]).filter((k) => k !== m)];
          tip(ev, {
            title: r.kind === "section" ? r.name : `${r.label} · ${c.n || 1} sections`,
            sub: `Turn ${vturns[ci]} · ${fmt.int(c.tokens)} tokens`,
            rows: order.map((k) => [fmt.x(c[k]), LABEL[k].toLowerCase()]),
            quote: c.hot_line ? `hottest line: ${c.hot_line}` : null,
          });
          if (st.pin && r.key === st.pin.key) { chartA?.setCross(vturns[ci]); chartB?.setCross(vturns[ci]); }
        },
        onLeave() { tip(null); chartA?.setCross(null); chartB?.setCross(null); },
        onClick({ ri, ci }) {
          if (!cellAt(rows[ri], vturns[ci], memo)) return;
          pinTo(rows[ri].key, vturns[ci]);
        },
        onRowClick(ri) {
          const r = rows[ri];
          const present = vturns.filter((t) => cellAt(r, t, memo));
          if (!present.length) return;
          const turn = st.pin && present.includes(st.pin.turn) ? st.pin.turn : present.at(-1);
          pinTo(r.key, turn);
        },
        onToggle(r) {
          const fold = new Set(st.fold);
          fold.has(r.key) ? fold.delete(r.key) : fold.add(r.key);
          update({ fold: fold.size ? fold : "-" });
        },
        onBrush(a, b) {
          if (a == null) update({ t0: null, t1: null });
          else update({ t0: vturns[a], t1: vturns[b] });
        },
        onColClick(ci) { if (st.pin) pinTo(st.pin.key, vturns[ci]); },
        onKey(d) {
          if (!d) { pinTo(null); return; }
          if (pinRi < 0) { pinTo(rows[0].key, vturns[0]); return; }
          const ri = Math.max(0, Math.min(rows.length - 1, pinRi + d[0]));
          const ci = Math.max(0, Math.min(vturns.length - 1, (pinCi < 0 ? 0 : pinCi) + d[1]));
          pinTo(rows[ri].key, vturns[ci]);
        },
        onMeanClick() { update({ sort: st.sort === "mean" ? null : "mean" }); },
      });
      card.append(hm.el);
      card.append(legend({
        title: LABEL[m], range: R, active: activeBins,
        onToggle(id, add) {
          const next = new Set(add ? activeBins : []);
          if (activeBins.has(id) && (add || activeBins.size === 1)) next.delete(id); else next.add(id);
          update({ bins: next });
        },
        onHover(id) { hoverBin = id; hm.redraw(); },
        extra: [
          st.marks ? h("span", { class: "key" }, dotKey(false), `consult ≥ ${st.ct.toFixed(1)}×`) : null,
          st.marks ? h("span", { class: "key" }, dotKey(true), `consult ≤ ${(1 / st.ct).toFixed(2)}×`) : null,
          h("span", { class: "key" }, h("span", { class: "swatch", style: { border: "1px solid var(--muted)", background: "transparent" } }), "no value"),
          h("span", { class: "key muted", text: "Blank: not in the context at that turn" }),
          h("span", { class: "key muted" }, h("kbd", { text: "←↑↓→" }), " move the pin, ", h("kbd", { text: "Esc" }), " clear"),
        ],
      }));
      if (pinRi >= 0) hm.setPin({ ri: pinRi, ci: pinCi >= 0 ? pinCi : null });
    }

    content.replaceChildren(head, controls1, controls2, card);
    renderPanel(memo, pinRi);
  }

  // A pin from another view names a section but not a turn: use its last turn, and unfold it.
  function resolvePin(turns) {
    const name = st.pin.key.startsWith("s:") ? st.pin.key.slice(2) : null;
    const present = name ? turns.filter((t) => cellsBy.has(`${name}\u0000${t}`)) : [];
    if (!present.length) { st.pin = null; setParams({ pin: null }); return; }
    const sec = data.sections.find((x) => x.name === name);
    st.fold.delete(`g:${sec.group}`);
    st.fold.delete(`p:${sec.parent}`);
    st.hide.delete(sec.group);
    st.pin.turn = present.at(-1);
    setParams({ pin: `${st.pin.turn}~${st.pin.key}`, fold: [...st.fold].join(",") || "-", hide: [...st.hide].join(",") || null });
  }

  function pinTo(key, turn) {
    update({ pin: key == null ? null : `${turn}~${key}` });
    hm?.el.focus({ preventScroll: true });
  }

  // -- pinned details -------------------------------------------------------------------------------------
  function showPanel(on) {
    split.classList.toggle("with-panel", on);
    if (on && !panel.isConnected) split.append(panel);
    if (!on) panel.remove();
  }

  function renderPanel(memo, pinRi) {
    if (!st.pin || pinRi < 0) {
      showPanel(false);
      chartA = chartB = null;
      return;
    }
    showPanel(true);
    const row = rows[pinRi];
    const turn = st.pin.turn;
    const c = cellAt(row, turn, memo);
    const ti = data.turns.find((t) => t.turn === turn) || {};
    const head = h("div", { class: "pane-head" },
      h("div", { class: "row" },
        h("h3", { text: row.kind === "section" ? row.name : `${row.label} (${row.count} sections)` }),
        h("button", { class: "iconbtn", title: "Clear the pin (Esc)", "aria-label": "Close", onclick: () => pinTo(null) }, "×")),
      h("div", { class: "muted", text: `${groupLabel(row.group)} · turn ${turn}` }));

    const tiles = h("div", { class: "pane-sec" }, h("div", { class: "tiles" },
      tile("Tokens", fmt.int(c?.tokens)), tile("Read lift", fmt.x(c?.lift)), tile("Peak lift", fmt.x(c?.peak)),
      tile("Consult ratio", fmt.x(c?.consult_ratio)), tile("Consult contrast", fmt.x(c?.consult_contrast)),
      tile("Turn", String(turn))));

    const turnsAll = data.turns.map((t) => t.turn);
    const series = (keys) => keys.map(([k, l, color]) => ({ key: k, label: l, color, values: turnsAll.map((t) => cellAt(row, t, memo)?.[k] ?? null) }));
    const toggle = (k) => { const lh = new Set(st.lh); lh.has(k) ? lh.delete(k) : lh.add(k); update({ lh }); };
    chartA = lineChart({
      xs: turnsAll, series: series([["lift", "read lift", "var(--series-1)"], ["peak", "peak lift", "var(--series-2)"]]),
      hidden: st.lh, onToggle: toggle, pinX: turn, title: "Read and peak lift across turns",
      onHover: (t) => hm?.setHover(t == null ? null : { ri: pinRi, ci: vturns.indexOf(t) >= 0 ? vturns.indexOf(t) : null }),
      onClick: (t) => cellAt(row, t, memo) && pinTo(row.key, t),
    });
    chartB = lineChart({
      xs: turnsAll, series: series([["consult_ratio", "consult ratio", "var(--series-1)"], ["consult_contrast", "consult contrast", "var(--series-2)"]]),
      hidden: st.lh, onToggle: toggle, pinX: turn, height: 104, title: "Consult ratio and contrast across turns",
      onHover: (t) => hm?.setHover(t == null ? null : { ri: pinRi, ci: vturns.indexOf(t) >= 0 ? vturns.indexOf(t) : null }),
      onClick: (t) => cellAt(row, t, memo) && pinTo(row.key, t),
    });

    const textSec = h("div", { class: "pane-sec" });
    if (row.kind === "section") {
      textSec.append(h("h4", null, `Text at turn ${turn}`, h("span", { class: "spacer" }), h("span", { class: "muted", text: "colored by token read lift" })));
      if (c?.hot_line) textSec.append(h("div", { class: "muted", style: { marginBottom: "6px", fontSize: "12px" } }, "Hottest line: ", h("span", { class: "mono ink2", text: c.hot_line })));
      const box = h("div", { class: "text", text: "Loading…" });
      textSec.append(box);
      api(`/api/stores/${st.store}/episodes/${encodeURIComponent(st.ep)}/text?turn=${turn}&section=${encodeURIComponent(row.name)}`)
        .then((res) => {
          if (!res.tokens.length) { box.textContent = res.reason || "No text."; return; }
          box.replaceChildren(...res.tokens.map(([t, v]) => {
            const pos = scalePos(v, R);
            const on = pos != null && Math.abs(pos) > 0.12;
            const sp = h("span", { text: t, style: { background: on ? divergingColor(pos) : "transparent", color: on ? inkOn(pos) : null } });
            sp.addEventListener("pointermove", (ev) => tip(ev, { title: JSON.stringify(t), rows: [[fmt.x(v), "token read lift"]] }));
            sp.addEventListener("pointerleave", () => tip(null));
            return sp;
          }));
        })
        .catch((e) => { box.textContent = String(e); });
    } else {
      textSec.append(h("h4", null, `Sections at turn ${turn}`, h("span", { class: "spacer" }), h("span", { class: "muted", text: "read lift" })));
      const members = row.members.map((n) => ({ n, c: cellsBy.get(`${n}\u0000${turn}`) })).filter((x) => x.c)
        .sort((a, b) => (b.c.lift ?? 0) - (a.c.lift ?? 0));
      textSec.append(h("div", { class: "members" }, members.map(({ n, c: mc }) => {
        const pos = scalePos(mc.lift, R) ?? 0;
        return h("div", {
          class: "member", title: n,
          onclick: () => {
            const fold = new Set(st.fold);
            fold.delete(row.key);
            fold.delete(`p:${data.sections.find((s) => s.name === n)?.parent}`);
            update({ fold: fold.size ? fold : "-", pin: `${turn}~s:${n}` });
          },
        }, h("span", { class: "nm", text: n.startsWith(`${row.name}:`) || n.startsWith(`${row.name}/`) ? n.slice(row.name.length + 1) : n }),
        h("span", { class: "num", style: { textAlign: "right" }, text: fmt.x(mc.lift) }),
        barSvg(pos));
      })));
    }

    const turnSec = h("div", { class: "pane-sec" }, h("h4", { text: `Turn ${turn}` }), h("div", { class: "tiles" },
      tile("Context", fmt.int(ti.L)), tile("Generated", fmt.int(ti.n_gen)), tile("Entropy", fmt.num(ti.entropy))),
    ti.action ? h("div", { class: "muted", style: { marginTop: "6px" }, text: `Action: ${ti.action}` }) : null);

    const ex = expl.get(turn);
    const why = ex ? whySection(ex, { current: row.kind === "section" ? row.name : null, onSection: (n) => pinTo(`s:${n}`, turn) }) : null;
    panel.replaceChildren(head, ...(why ? [why] : []), tiles, textSec,
      h("div", { class: "pane-sec" }, h("h4", { text: "Across turns" }), chartA.el, h("div", { style: { height: "8px" } }), chartB.el),
      turnSec);
  }

  // -- loading --------------------------------------------------------------------------------------------
  async function loadList() {
    if (!st.store) return;
    eps = await api(`/api/stores/${st.store}/episodes`);
    renderList();
  }

  async function loadEpisode(keepFrame) {
    if (!st.store) {
      content.replaceChildren(h("div", { class: "empty", text: "Choose a trace store." }));
      return;
    }
    if (!st.ep) {
      const q = st.eq.toLowerCase(), match = (e) => !q || `${e.episode} ${e.task || ""}`.toLowerCase().includes(q);
      const first = eps.find((e) => e.replayed && match(e)) || eps.find((e) => e.replayed) || eps[0];
      if (first) { openEpisode(first.episode); return; }
      content.replaceChildren(h("div", { class: "empty", text: "This store has no episodes yet." }));
      return;
    }
    if (keepFrame) content.classList.add("loading");
    else content.replaceChildren(h("div", { class: "empty", text: "Computing the read map…" }));
    try {
      data = await api(`/api/stores/${st.store}/episodes/${encodeURIComponent(st.ep)}/readmap`);
    } catch (e) {
      // a live refresh that fails keeps the last frame; the next change retries
      if (!keepFrame || !data) content.replaceChildren(h("div", { class: "empty", text: String(e) }));
      return;
    } finally {
      content.classList.remove("loading");
    }
    try {
      const ex = await api(`/api/stores/${st.store}/episodes/${encodeURIComponent(st.ep)}/explanations`);
      expl = new Map(ex.map((e) => [e.turn, e]));
    } catch { expl = new Map(); }
    st = settings(route().params, data.sections);
    prepare();
    if (!data.cells.length) {
      const store = ctx.sources?.stores.find((s) => s.id === st.store);
      content.replaceChildren(h("div", { class: "empty" },
        h("p", { text: "This episode has not been replayed, so there is no attention to show yet." }),
        h("p", null, "Replay it with ", h("code", { text: `vitals replay --store ${store?.path || "<store>"} --gguf <model.gguf> --episodes ${st.ep}` }))));
      renderPanel(new Map(), -1);
      return;
    }
    render();
  }

  let resizeTimer;
  const onResize = () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 150); };
  addEventListener("resize", onResize);

  (async () => {
    await loadList();
    await loadEpisode(false);
  })();

  return {
    async refresh(changes) {
      if (!changes.some((c) => c.store === st.store || c.store === "*")) return;
      await loadList();
      if (st.follow) followNewest();
      if (changes.some((c) => c.store === st.store && c.episode === st.ep)) await loadEpisode(true);
    },
    destroy() { removeEventListener("resize", onResize); tip(null); },
  };
}

function tile(k, v) {
  return h("div", { class: "tile" }, h("div", { class: "k", text: k }), h("div", { class: "v", text: v }));
}

function dotKey(low) {
  const sv = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  sv.setAttribute("width", "10"); sv.setAttribute("height", "10"); sv.setAttribute("aria-hidden", "true");
  const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  c.setAttribute("cx", "5"); c.setAttribute("cy", "5"); c.setAttribute("r", "3.5");
  c.setAttribute("fill", low ? "var(--surface)" : "var(--ink)");
  c.setAttribute("stroke", low ? "var(--ink)" : "var(--surface)");
  c.setAttribute("stroke-width", "1.2");
  sv.append(c);
  return sv;
}

function barSvg(pos) {
  const sv = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  sv.setAttribute("viewBox", "0 0 90 10"); sv.setAttribute("width", "90"); sv.setAttribute("height", "10");
  const w = pos * 42;
  const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  r.setAttribute("x", String(w >= 0 ? 45 : 45 + w)); r.setAttribute("y", "1");
  r.setAttribute("width", String(Math.max(1.5, Math.abs(w)))); r.setAttribute("height", "8"); r.setAttribute("rx", "2");
  r.setAttribute("fill", divergingColor(pos));
  const base = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  base.setAttribute("x", "44.5"); base.setAttribute("y", "0"); base.setAttribute("width", "1"); base.setAttribute("height", "10");
  base.setAttribute("fill", "var(--ink-2)");
  sv.append(r, base);
  return sv;
}
