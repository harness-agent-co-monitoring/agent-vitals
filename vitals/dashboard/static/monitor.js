// Monitor: each target episode's section reading against a reference store's normal. An alarm is raised
// when an episode reads a section far outside the reference spread (robust z), when a section the reference
// always has is missing, or when a declared section is new. Alarms on the left; the selected section's
// control chart and the sections x episodes health map on the right.

import { divergingColor } from "./color.js";
import { controlChart, kindIcon } from "./control.js";
import { heatmap, legend, outcomeMark } from "./heatmap.js";
import { buildRows, defaultFold, groupChips, groupLabel } from "./rows.js";
import { api, checkbox, fmt, go, h, href, list, route, seg, setParams, statusIcon, table, tip } from "./util.js";

const KINDS = [
  ["over-read", "Over-read", "Read far more than the reference normal"],
  ["under-read", "Under-read", "Read far less than the reference normal"],
  ["missing", "Missing", "In at least 90% of reference episodes, absent here"],
  ["new", "New", "A declared section the reference never had"],
];
const STRUCTURAL = new Set(["missing", "new"]);

const fmtZ = (z) => (z == null ? "–" : `${z > 0 ? "+" : z < 0 ? "−" : ""}${Math.abs(z).toFixed(1)}`);

function settings(p, sections) {
  return {
    store: p.store, ref: p.ref || p.store,
    thr: Number(p.thr) > 0 ? Number(p.thr) : 3,
    conv: p.conv === "1",
    sel: p.sel || null, ep: p.ep || null,
    kinds: new Set(list(p.kinds)),
    aq: p.aq || "",
    asort: p.asort === "recent" ? "recent" : "severity",
    agroup: p.agroup === "episode" ? "episode" : "section",
    follow: p.follow === "1",
    hide: new Set(list(p.hide)),
    q: p.q || "",
    fold: p.fold == null ? defaultFold(sections) : new Set(list(p.fold).filter((k) => k !== "-")),
    sort: p.sort || "order",
    grouped: p.grouped !== "0",
    bins: new Set(list(p.bins)),
    view: p.view === "table" ? "table" : "chart",
    tex: p.tex === "1",
    e0: p.e0 || null, e1: p.e1 || null,
    lh: new Set(list(p.lh)),
    tsort: p.tsort || "",
  };
}

// z bins for the legend filter; t is the swatch position on the diverging scale.
function zBins(T) {
  const pos = (z) => Math.max(-1, Math.min(1, z / (1.5 * T)));
  const mid = (1 + T) / 2;
  return [
    { id: "under", lo: -Infinity, hi: -T, text: `≤ −${T}`, t: pos(-1.25 * T) },
    { id: "low", lo: -T, hi: -1, text: `−${T} to −1`, t: pos(-mid) },
    { id: "norm", lo: -1, hi: 1, text: "−1 to +1", t: 0 },
    { id: "high", lo: 1, hi: T, text: `+1 to +${T}`, t: pos(mid) },
    { id: "over", lo: T, hi: Infinity, text: `≥ +${T}`, t: pos(1.25 * T) },
  ];
}

export function monitorView(root, ctx) {
  let data = null, hm = null, cc = null, hoverBin = null, timer = null, baseline = null, raf = 0;
  let st = settings(route().params, []);
  const feed = h("div", { class: "list" });
  const content = h("div", { class: "content" });
  root.replaceChildren(h("div", { class: "split wide-list" }, feed, content));

  function update(patch, rerender = true) {
    const before = [st.store, st.ref, st.conv].join("|");
    setParams(Object.fromEntries(Object.entries(patch).map(([k, v]) => [k, v instanceof Set ? [...v].join(",") || null : v])));
    st = settings(route().params, data?.sections || []);
    if ([st.store, st.ref, st.conv].join("|") !== before) load(false);
    else if (rerender) render();
  }

  const storeLabel = (sid) => ctx.sources?.stores.find((s) => s.id === sid)?.label || sid;
  const openReadMap = (sid, ep, section) => go("/episode", { store: sid, ep, pin: section ? `last~s:${section}` : null });

  function render() {
    if (!data) return;
    const T = st.thr;
    const zPos = (z) => (z == null ? null : Math.max(-1, Math.min(1, z / (1.5 * T))));
    const bins = zBins(T);
    const binOfZ = (z) => (z == null ? null : bins.find((b) => z >= b.lo && z < b.hi)?.id);
    const activeBins = new Set([...st.bins].filter((b) => bins.some((x) => x.id === b)));

    const secBy = new Map(data.sections.map((s) => [s.name, s]));
    const all = data.episodes.map((e, i) => ({
      ...e, index: i, missSet: new Set(e.missing),
      label: e.task != null && !e.episode.includes(String(e.task)) ? `${e.task} · ${e.episode}` : e.episode,
    }));
    let i0 = st.e0 ? all.findIndex((e) => e.episode === st.e0) : 0;
    let i1 = st.e1 ? all.findIndex((e) => e.episode === st.e1) : all.length - 1;
    if (i0 < 0) i0 = 0;
    if (i1 < 0 || i1 < i0) i1 = all.length - 1;
    const eps = all.slice(i0, i1 + 1);
    const zoomed = i0 > 0 || i1 < all.length - 1;
    const fresh = (e) => baseline && (!baseline.has(e.episode) || baseline.get(e.episode) !== e.updated);

    // alarms, at the current threshold, for the shown episodes and sections
    const q = st.q.toLowerCase();
    const shownSec = (n) => { const s = secBy.get(n); return s && !st.hide.has(s.group) && (!q || n.toLowerCase().includes(q)); };
    const alarms = [];
    const epBy = new Map(eps.map((e) => [e.episode, e]));
    for (const e of eps) {
      for (const [n, z] of Object.entries(e.z)) {
        if (Math.abs(z) >= T) alarms.push({ e, section: n, kind: z < 0 ? "under-read" : "over-read", z, lift: e.lift[n], normal: secBy.get(n)?.normal });
      }
    }
    for (const a of data.alarms) {
      if (STRUCTURAL.has(a.kind) && epBy.has(a.episode)) alarms.push({ e: epBy.get(a.episode), section: a.section, kind: a.kind, z: null, lift: a.lift, normal: a.normal });
    }
    const shown = alarms.filter((a) => shownSec(a.section));
    const counts = Object.fromEntries(KINDS.map(([k]) => [k, shown.filter((a) => a.kind === k).length]));
    const alarmAt = new Map(shown.map((a) => [`${a.section}\u0000${a.e.episode}`, a]));
    const perSection = new Map(), perEpisode = new Map();
    for (const a of shown) {
      perSection.set(a.section, (perSection.get(a.section) || 0) + 1);
      perEpisode.set(a.e.episode, (perEpisode.get(a.e.episode) || 0) + 1);
    }
    const severity = (a) => (STRUCTURAL.has(a.kind) ? 1e6 : Math.abs(a.z));
    const bySeverity = (a, b) => severity(b) - severity(a) || b.e.index - a.e.index;
    const byRecent = (a, b) => b.e.index - a.e.index || severity(b) - severity(a);
    const ranked = [...shown].sort(bySeverity);

    // selection: the URL's, else the most severe alarm's section
    let sel = st.sel && secBy.has(st.sel) ? st.sel : ranked[0]?.section || data.sections.find((s) => shownSec(s.name))?.name || null;
    const selEp = st.ep && eps.some((e) => e.episode === st.ep) ? st.ep : null;

    // -- feed -----------------------------------------------------------------------------------------------
    const items = shown.filter((a) => !st.kinds.size || st.kinds.has(a.kind)).sort(st.asort === "recent" ? byRecent : bySeverity);
    const ktiles = h("div", { class: "ktiles", role: "group", "aria-label": "Alarm kinds" }, KINDS.map(([k, label, desc]) => h("button", {
      class: `ktile ${st.kinds.size && !st.kinds.has(k) ? "off" : ""} ${st.kinds.has(k) ? "solo" : ""}`,
      "aria-pressed": String(st.kinds.has(k)), title: `${desc}. Click to show only these, shift-click to add.`,
      onclick: (ev) => {
        const next = new Set(ev.shiftKey ? st.kinds : []);
        if (st.kinds.has(k) && (ev.shiftKey || st.kinds.size === 1)) next.delete(k); else next.add(k);
        update({ kinds: next });
      },
    }, kindIcon(k), h("span", { class: "kl", text: label }), h("span", { class: "kn", text: String(counts[k]) }))));
    const search = h("input", { type: "search", placeholder: "Filter alarms", value: st.aq, "aria-label": "Filter alarms" });
    search.addEventListener("input", (e) => { st.aq = e.target.value; setParams({ aq: e.target.value }); renderItems(); });
    const body = h("div", { class: "list-body", role: "listbox", "aria-label": "Alarms" });
    feed.replaceChildren(h("div", { class: "list-head" },
      h("div", { class: "lab muted", text: `Alarms in ${eps.length} episode${eps.length === 1 ? "" : "s"}` }),
      ktiles, search,
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Group" }), seg([["section", "Section"], ["episode", "Episode"]], st.agroup, (v) => update({ agroup: v === "section" ? null : v }), "Group alarms")),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Order" }), seg([["severity", "Severity"], ["recent", "Recent"]], st.asort, (v) => update({ asort: v === "severity" ? null : v }), "Order alarms")),
      checkbox("Follow new alarms", st.follow, (on) => update({ follow: on ? "1" : null }, false), "Select the newest alarm whenever an episode changes")), body);

    function renderItems() {
      const aq = st.aq.toLowerCase();
      const list_ = items.filter((a) => !aq || `${a.section} ${a.e.label}`.toLowerCase().includes(aq));
      const readMapLink = (a) => h("a", {
        href: href("/episode", { store: st.store, ep: a.e.episode, pin: a.kind === "missing" ? null : `last~s:${a.section}` }),
        title: "Open this episode's read map on the section", onclick: (ev) => ev.stopPropagation(), text: "read map",
      });
      const detail = (a) => (a.kind === "missing" ? `normal ${fmt.x(a.normal)}` : a.kind === "new" ? `read ${fmt.x(a.lift)}` : `${fmt.x(a.lift)} vs ${fmt.x(a.normal)}`);
      const justRaised = () => h("span", { class: "status fresh", title: "An episode changed while the page was open" }, h("span", { class: "dot live" }), "just raised");
      const hoverable = (el, section, episode) => {
        el.addEventListener("pointerenter", () => linkHover(section, episode));
        el.addEventListener("pointerleave", () => linkHover(null));
        return el;
      };
      const els = [];
      if (st.agroup === "episode") {
        for (const a of list_.slice(0, 500)) {
          const on = a.section === sel && a.e.episode === selEp;
          els.push(hoverable(h("div", {
            class: `item alarm ${on ? "on" : ""}`, role: "option", "aria-selected": String(on), tabindex: "0",
            onclick: () => update({ sel: a.section, ep: a.e.episode }),
            onkeydown: (ev) => { if (ev.key === "Enter") update({ sel: a.section, ep: a.e.episode }); },
          }, kindIcon(a.kind), h("span", { class: "name", title: a.section, text: a.section }),
          h("span", { class: "zv num", text: a.z != null ? fmtZ(a.z) : a.kind }),
          h("span", { class: "sub" }, fresh(a.e) ? justRaised() : null, h("span", { class: "ep", title: a.e.label, text: a.e.label }),
            h("span", { text: detail(a) }), readMapLink(a))), a.section, a.e.episode));
        }
      } else {
        const groups = new Map();
        for (const a of list_) {
          const k = `${a.section}\u0000${a.kind}`;
          if (!groups.has(k)) groups.set(k, []);
          groups.get(k).push(a);
        }
        for (const as of [...groups.values()].slice(0, 300)) {
          const worst = as.reduce((w, a) => (severity(a) > severity(w) ? a : w)), on = worst.section === sel;
          els.push(hoverable(h("div", {
            class: `item alarm ${on ? "on" : ""}`, role: "option", "aria-selected": String(on), "aria-expanded": String(on), tabindex: "0",
            onclick: () => update({ sel: worst.section, ep: on ? selEp : worst.e.episode }),
            onkeydown: (ev) => { if (ev.key === "Enter") update({ sel: worst.section, ep: worst.e.episode }); },
          }, kindIcon(worst.kind), h("span", { class: "name", title: worst.section, text: worst.section }),
          h("span", { class: "zv num", title: `${as.length} episode${as.length > 1 ? "s" : ""}`, text: `${as.length}×` }),
          h("span", { class: "sub" }, as.some((a) => fresh(a.e)) ? justRaised() : null,
            h("span", { text: `${worst.kind} in ${as.length} of ${eps.length} episodes` }),
            worst.z != null ? h("span", { text: `worst ${fmtZ(worst.z)}` }) : null)), worst.section, worst.e.episode));
          if (!on) continue;
          for (const a of as) {
            const aon = a.e.episode === selEp;
            els.push(hoverable(h("div", {
              class: `subitem ${aon ? "on" : ""}`, tabindex: "0", role: "option", "aria-selected": String(aon),
              onclick: () => update({ ep: aon ? null : a.e.episode }),
              onkeydown: (ev) => { if (ev.key === "Enter") update({ ep: a.e.episode }); },
            }, h("span", { class: "ep", title: a.e.label, text: a.e.label }), h("span", { class: "num zv", text: a.z != null ? fmtZ(a.z) : "" }),
            h("span", { class: "sub" }, h("span", { text: detail(a) }), readMapLink(a))), a.section, a.e.episode));
          }
        }
      }
      body.replaceChildren(...els);
      if (!list_.length) {
        body.append(h("div", { class: "empty" }, shown.length
          ? "No alarms match the filter."
          : h("span", { class: "status" }, statusIcon("good"), `No alarms at |z| ≥ ${T} in ${eps.length} episodes`)));
      }
      (body.querySelector(".subitem.on") || body.querySelector(".item.on"))?.scrollIntoView({ block: "nearest" });
    }
    renderItems();

    // -- header and controls ----------------------------------------------------------------------------
    const stores = ctx.sources?.stores || [];
    const picker = (value, onChange, label, self) => h("select", { "aria-label": label, onchange: (e) => onChange(e.target.value) },
      self ? h("option", { value: st.store, text: "itself (outliers within the store)", selected: value === st.store }) : null,
      stores.filter((s) => !self || s.id !== st.store).map((s) => h("option", { value: s.id, text: `${s.label} (${s.replayed})`, selected: s.id === value })));
    const alarmedEps = new Set(shown.map((a) => a.e.episode)).size;
    const prog = [data.target.progress, data.reference.progress];
    const done = prog.every((p) => p.done >= p.total);
    const head = h("div", { class: "head" },
      h("h1", { text: "Monitor" }),
      h("div", { class: "facts" },
        h("span", null, h("b", { text: storeLabel(st.store) }), st.ref === st.store ? " against itself" : [" against ", h("b", { text: storeLabel(st.ref) })]),
        h("span", null, h("b", { text: String(eps.length) }), zoomed ? ` of ${all.length}` : "", " episodes"),
        alarmedEps
          ? h("span", { class: "status" }, statusIcon("warning"), h("b", { text: String(alarmedEps) }), " with alarms")
          : h("span", { class: "status" }, statusIcon("good"), "no alarms"),
        h("span", null, h("b", { text: String(data.reference.episodes) }), " reference episodes")),
      done ? null : h("span", { class: "progress" }, h("span", { class: "bar" }, h("i", { style: { width: `${(100 * (prog[0].done + prog[1].done)) / Math.max(1, prog[0].total + prog[1].total)}%` } })), "computing read maps…"));

    const thrIn = h("input", { type: "range", min: "1.5", max: "6", step: "0.5", value: String(T), "aria-label": "Alarm threshold" });
    thrIn.addEventListener("input", (e) => {
      setParams({ thr: e.target.value === "3" ? null : e.target.value });
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => { st = settings(route().params, data.sections); render(); });
    });
    const controls1 = h("div", { class: "controls" },
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Target" }), picker(st.store, (v) => update({ store: v, ref: st.ref === st.store ? null : st.ref, sel: null, ep: null, e0: null, e1: null }), "Target store")),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "against" }), picker(st.ref, (v) => update({ ref: v === st.store ? null : v }), "Reference store", true)),
      h("div", { class: "ctl", title: "z: the episode's mean log2 read lift minus the reference median, over the reference spread (1.4826 × MAD)" },
        h("span", { class: "lab", text: "Alarm at |z| ≥" }), thrIn, h("span", { class: "num ink2", text: T.toFixed(1) })),
      h("div", { class: "ctl" }, checkbox("Conversation sections", st.conv, (on) => update({ conv: on ? "1" : null }), "Also monitor observations, user turns and the agent's own turns")),
      h("div", { class: "ctl" }, checkbox("Texture", st.tex, (on) => update({ tex: on ? "1" : null }), "Hatch cells by direction and size, for print and color-vision deficiency")),
      h("div", { style: { flex: 1 } }),
      h("div", { class: "ctl" }, seg([["chart", "Chart"], ["table", "Table"]], st.view, (v) => update({ view: v === "chart" ? null : v }), "View")));
    const fsearch = h("input", { type: "search", placeholder: "Find section", value: st.q, "aria-label": "Find section" });
    let qt;
    fsearch.addEventListener("input", (e) => { clearTimeout(qt); qt = setTimeout(() => update({ q: e.target.value }), 180); });
    const controls2 = h("div", { class: "controls" },
      h("div", { class: "ctl" }, fsearch),
      h("div", { class: "ctl" }, groupChips(data.sections, st.hide, (hide) => update({ hide }))),
      h("div", { class: "ctl" }, h("span", { class: "lab", text: "Rows" }),
        seg([["order", "Reference order"], ["alarms", "Alarms"], ["shift", "Shift"], ["name", "Name"]], st.sort, (v) => update({ sort: v === "order" ? null : v }), "Sort rows"),
        checkbox("Grouped", st.grouped, (on) => update({ grouped: on ? null : "0" }))),
      zoomed
        ? h("button", { class: "chip x", title: "Show all episodes", onclick: () => update({ e0: null, e1: null }) }, `episodes ${i0 + 1}–${i1 + 1}`)
        : h("span", { class: "muted", style: { fontSize: "12px" }, text: "Drag across the chart or the outcome strip to zoom" }));

    // -- per-section statistics over the shown episodes -----------------------------------------------------
    const medZ = new Map();
    for (const s of data.sections) {
      const zs = eps.map((e) => e.z[s.name]).filter((z) => z != null).sort((a, b) => a - b);
      medZ.set(s.name, zs.length ? zs[Math.floor((zs.length - 1) / 2)] / 2 + zs[Math.ceil((zs.length - 1) / 2)] / 2 : null);
    }
    const rows = buildRows(data.sections, {
      fold: st.fold, hide: st.hide, q: st.q, sort: st.sort, grouped: st.grouped,
      stats: { alarms: (n) => perSection.get(n) || 0, shift: (n) => (medZ.get(n) == null ? null : Math.abs(medZ.get(n))) },
    });
    for (const r of rows) if (r.kind === "section" && secBy.get(r.name)?.new && !secBy.get(r.name).varying) r.label += " · new";

    const memo = new Map();
    function cellOf(r, e) {
      const k = `${r.key}\u0000${e.episode}`;
      if (memo.has(k)) return memo.get(k);
      let out = null;
      if (r.kind === "section") {
        const n = r.name, s = secBy.get(n);
        if (s.new || s.sparse) out = e.lift[n] != null ? { kind: s.new ? "new" : "sparse", name: n, lift: e.lift[n] } : null;
        else if (e.z[n] != null) out = { name: n, z: e.z[n], lift: e.lift[n], alarm: Math.abs(e.z[n]) >= T };
        else if (e.missSet.has(n)) out = { name: n, missing: true };
      } else {
        let worst = null, alarm = 0, miss = 0, fresh_ = 0, present = 0;
        for (const n of r.members) {
          const c = cellOf({ key: `s:${n}`, kind: "section", name: n }, e);
          if (!c) continue;
          present++;
          if (c.missing) miss++;
          if (c.kind === "new" && !secBy.get(n).varying) fresh_++;
          if (c.alarm) alarm++;
          if (c.z != null && (!worst || Math.abs(c.z) > Math.abs(worst.z))) worst = c;
        }
        if (present) out = { name: worst?.name, z: worst?.z ?? null, lift: worst?.lift, alarm: alarm > 0, alarms: alarm, missing: !worst && miss > 0, miss, added: fresh_, n: present };
      }
      memo.set(k, out);
      return out;
    }
    const rowMedian = (r) => {
      if (r.kind === "section") return medZ.get(r.name);
      const zs = eps.map((e) => cellOf(r, e)?.z).filter((z) => z != null).sort((a, b) => a - b);
      return zs.length ? zs[Math.floor(zs.length / 2)] : null;
    };

    const selRi = rows.findIndex((r) => r.key === `s:${sel}`);
    const selCi = selEp ? eps.findIndex((e) => e.episode === selEp) : -1;

    // -- control chart ------------------------------------------------------------------------------------
    const s = sel ? secBy.get(sel) : null;
    const chartCard = h("div", { class: "card" });
    if (s) {
      const sectionPick = h("select", { "aria-label": "Section", onchange: (e) => update({ sel: e.target.value }) },
        [...data.sections].filter((x) => shownSec(x.name) || x.name === sel)
          .sort((a, b) => (perSection.get(b.name) || 0) - (perSection.get(a.name) || 0))
          .map((x) => h("option", { value: x.name, selected: x.name === sel, text: `${x.name}${perSection.get(x.name) ? ` (${perSection.get(x.name)} alarm${perSection.get(x.name) > 1 ? "s" : ""})` : ""}` })));
      const selAlarm = selEp ? alarmAt.get(`${sel}\u0000${selEp}`) : null;
      const selE = selEp ? eps.find((e) => e.episode === selEp) : null;
      chartCard.append(h("div", { class: "card-head" },
        h("h2", { text: s.new ? "New section, read per episode" : s.sparse ? "Read lift per episode" : "Read lift per episode against the reference" }),
        h("span", { class: "muted", text: s.new ? "The reference has no episodes with this section."
          : s.sparse ? `Only ${s.n} reference episode${s.n === 1 ? " has" : "s have"} it; 3 are needed for a normal.`
            : `${s.n} of ${data.reference.episodes} reference episodes have it; normal ${fmt.x(s.normal)}, spread ×${(2 ** s.spread).toFixed(2)}` }),
        h("span", { class: "spacer" }), sectionPick));
      if (selE) {
        chartCard.append(h("div", { class: "selbar" },
          selAlarm ? kindIcon(selAlarm.kind) : null,
          h("span", null, h("b", { text: selE.label }), selAlarm
            ? ` · ${selAlarm.kind}${selAlarm.z != null ? ` at z ${fmtZ(selAlarm.z)}` : ""}`
            : selE.z[sel] != null ? ` · z ${fmtZ(selE.z[sel])}, within the normal band` : selE.lift[sel] != null ? "" : " · section not in this episode"),
          h("span", { class: "muted", text: selE.lift[sel] != null ? `read ${fmt.x(selE.lift[sel])}` : "" }),
          h("span", { class: "spacer" }),
          h("button", { class: "iconbtn", onclick: () => openReadMap(st.store, selE.episode, selE.lift[sel] != null ? sel : null) }, "Open read map →"),
          h("button", { class: "iconbtn", title: "Clear the episode selection", "aria-label": "Clear", onclick: () => update({ ep: null }) }, "×")));
      }
      cc = controlChart({
        width: Math.max(360, content.clientWidth - 58), title: `${sel} control chart`,
        episodes: eps, values: eps.map((e) => e.lift[sel] ?? null), z: eps.map((e) => e.z[sel] ?? null),
        missing: eps.map((e) => e.missSet.has(sel)),
        band: s.new || s.sparse ? null : { median: Math.log2(s.normal), spread: s.spread, threshold: T },
        ref: s.ref || [], hidden: st.lh, selCi: selCi >= 0 ? selCi : null,
        onToggle: (k) => { const lh = new Set(st.lh); lh.has(k) ? lh.delete(k) : lh.add(k); update({ lh }); },
        onHover: (ci) => hm?.setHover(ci == null ? null : { ri: selRi >= 0 ? selRi : null, ci }),
        onClick: (ci) => update({ ep: eps[ci].episode === selEp ? null : eps[ci].episode }),
        onRefClick: (ep) => openReadMap(st.ref, ep, sel),
        onBrush: (a, b) => (a == null ? update({ e0: null, e1: null }) : update({ e0: eps[a].episode, e1: eps[b].episode, ep: null })),
      });
      chartCard.append(h("div", { class: "cc-body" }, cc.el));
    } else {
      cc = null;
      chartCard.append(h("div", { class: "empty", text: "No sections to monitor yet." }));
    }

    // -- health map -----------------------------------------------------------------------------------------
    const mapCard = h("div", { class: "card" }, h("div", { class: "card-head" },
      h("h2", { text: "Reading against the reference, by section and episode" }),
      h("span", { class: "muted", text: "z of each episode's mean read lift. Click a cell to chart it; double-click to open the read map." })));
    const tipFor = (r, e, c) => {
      const title = r.kind === "section" ? r.name : `${r.label} · ${r.count} sections`;
      const sub = `${e.label} · ${e.reward == null ? "no reward" : e.reward > 0 ? "pass" : "fail"}`;
      if (!c) return { title, sub: `${sub} · not in this episode` };
      if (r.kind !== "section") {
        return {
          title, sub,
          rows: [
            c.name ? [fmtZ(c.z), `largest shift: ${c.name}`] : null,
            [String(c.alarms), `alarm${c.alarms === 1 ? "" : "s"} at |z| ≥ ${T}`],
            c.miss ? [String(c.miss), "missing"] : null,
            c.added ? [String(c.added), "new"] : null,
          ].filter(Boolean),
        };
      }
      const sec = secBy.get(r.name);
      if (c.missing) return { title, sub: `${sub} · missing`, rows: [[fmt.pct(sec.present), "of reference episodes have it"], [fmt.x(sec.normal), "reference normal"]] };
      if (c.kind === "new") {
        return { title, sub: `${sub} · new`, rows: [[fmt.x(c.lift), "read lift"], ["–", sec.varying ? "the reference never had it; its group varies by episode, so no alarm" : "the reference never had this section"]] };
      }
      if (c.kind === "sparse") return { title, sub: `${sub} · no normal`, rows: [[fmt.x(c.lift), "read lift"], [String(sec.n), "reference episodes have it (3 needed for a normal)"]] };
      return {
        title, sub: `${sub}${c.alarm ? ` · ${c.z > 0 ? "over-read" : "under-read"} alarm` : ""}`,
        rows: [[fmtZ(c.z), "z against the reference"], [fmt.x(c.lift), "read lift"], [fmt.x(sec.normal), "reference normal"],
          [`${fmt.x(2 ** (Math.log2(sec.normal) - T * sec.spread))}–${fmt.x(2 ** (Math.log2(sec.normal) + T * sec.spread))}`, "normal band"]],
      };
    };
    const select = (r, e) => {
      const c = cellOf(r, e);
      if (!c) return;
      if (r.kind === "section") update({ sel: r.name, ep: e.episode });
      else if (c.name) {
        const fold = new Set(st.fold);
        fold.delete(r.key);
        fold.delete(`g:${secBy.get(c.name).group}`);
        fold.delete(`p:${secBy.get(c.name).parent}`);
        update({ sel: c.name, ep: e.episode, fold: fold.size ? fold : "-" });
      }
      hm?.el.focus({ preventScroll: true });
    };

    if (!rows.length || !eps.length) {
      hm = null;
      mapCard.append(h("div", { class: "empty", text: !eps.length ? "No replayed episodes in the target store yet." : st.q ? `No section matches “${st.q}”.` : "No sections to show." }));
    } else if (st.view === "table") {
      hm = null;
      const recs = [];
      for (const r of rows.filter((x) => x.kind === "section" || x.collapsed)) for (const e of eps) {
        const c = cellOf(r, e);
        if (!c) continue;
        const a = r.kind === "section" ? alarmAt.get(`${r.name}\u0000${e.episode}`) : null;
        recs.push({
          row: r, e, section: r.kind === "section" ? r.name : `${r.label} (${r.count})`, episode: e.label, reward: e.reward, z: c.z, lift: c.lift,
          normal: r.kind === "section" ? secBy.get(r.name).normal : null,
          status: r.kind !== "section" ? `${c.alarms} alarm${c.alarms === 1 ? "" : "s"}` : a ? a.kind : c.missing ? "missing" : c.kind === "new" ? "new" : "within normal",
        });
      }
      mapCard.append(table([
        { key: "section", label: "Section", cls: "wrap", render: (x) => h("span", null, h("span", { class: "swatch", style: { background: divergingColor(zPos(x.z)) || "transparent" } }), x.section) },
        { key: "episode", label: "Episode", cls: "wrap" },
        { key: "reward", label: "Reward", num: true, fmt: (v) => fmt.num(v) },
        { key: "z", label: "z", num: true, fmt: fmtZ },
        { key: "lift", label: "Read lift", num: true, fmt: fmt.x },
        { key: "normal", label: "Normal", num: true, fmt: fmt.x },
        { key: "status", label: "Status", render: (x) => h("span", { class: "status" }, KINDS.some(([k]) => k === x.status) ? kindIcon(x.status) : null, x.status) },
      ], recs.filter((x) => !activeBins.size || activeBins.has(binOfZ(x.z))), {
        sort: st.tsort, onSort: (v) => update({ tsort: v }), onRow: (x) => { update({ view: null }, false); select(x.row, x.e); },
      }));
    } else {
      const maxA = Math.max(1, ...eps.map((e) => perEpisode.get(e.episode) || 0));
      hm = heatmap({
        rows, ncols: eps.length, label: "z by section and episode", q: st.q, texture: st.tex, marks: true,
        avail: content.clientWidth - 34 || 900,
        corner: h("div", { style: { display: "flex", flexDirection: "column", gap: "3px", width: "100%", textAlign: "right" } }, h("span", { text: "outcome" }), h("span", { text: "alarms ↓" })),
        meanTitle: st.sort === "shift" ? "median z ↓" : "median z",
        pinned: selRi >= 0 ? { ri: selRi, ci: selCi >= 0 ? selCi : null } : null,
        strip: {
          height: 44,
          draw(c, { cw, W, hl, ink2, grid }) {
            eps.forEach((e, i) => {
              const x = i * cw;
              if (i === hl || i === selCi) { c.fillStyle = grid; c.fillRect(x, 0, cw, 44); }
              outcomeMark(c, x, cw, 4, e.reward);
              const n = perEpisode.get(e.episode) || 0;
              c.fillStyle = ink2;
              c.globalAlpha = i === hl ? 0.9 : 0.5;
              c.fillRect(x + 1, 20, Math.max(1, cw - 2), (n / maxA) * 22);
              c.globalAlpha = 1;
            });
            c.fillStyle = grid;
            c.fillRect(0, 18, W, 1);
          },
        },
        cell(ri, ci) {
          const c = cellOf(rows[ri], eps[ci]);
          if (!c) return null;
          if (c.z == null) return { fill: null, t: null, mark: c.missing ? "x" : null, faded: activeBins.size > 0 || !!hoverBin };
          const t = zPos(c.z), b = binOfZ(c.z);
          return { fill: divergingColor(t), t, mark: c.alarm ? "high" : null, faded: (activeBins.size && !activeBins.has(b)) || (hoverBin && b !== hoverBin) };
        },
        mean(ri) { const m = rowMedian(rows[ri]); return m == null ? null : { t: zPos(m) }; },
        onHover({ ri, ci, ev, strip, mean: isMean, label }) {
          if (strip) {
            const e = eps[ci], top = ranked.find((a) => a.e.episode === e.episode);
            tip(ev, {
              title: e.label, sub: `episode ${e.index + 1}${fresh(e) ? " · changed while open" : ""}`,
              rows: [[e.reward == null ? "–" : e.reward > 0 ? `pass (${e.reward})` : `fail (${e.reward})`, "outcome"],
                [String(perEpisode.get(e.episode) || 0), "alarms"], top ? [top.z != null ? fmtZ(top.z) : top.kind, `top: ${top.section}`] : null].filter(Boolean),
            });
            cc?.setCross(ci);
            return;
          }
          const r = rows[ri];
          if (isMean || label) {
            const sec = r.kind === "section" ? secBy.get(r.name) : null;
            tip(ev, {
              title: r.kind === "section" ? r.name : `${r.label} · ${r.count} sections`,
              sub: r.kind === "section" ? (sec.new ? "new in the target" : groupLabel(r.group)) : "the largest shift among the visible sections",
              rows: [[fmtZ(rowMedian(r)), "median z over the shown episodes"],
                [String(r.members.reduce((acc, n) => acc + (perSection.get(n) || 0), 0)), "alarms"],
                sec && !sec.new ? [fmt.x(sec.normal), "reference normal"] : null,
                sec && !sec.new ? [fmt.pct(sec.present), "of reference episodes have it"] : null].filter(Boolean),
            });
            return;
          }
          tip(ev, tipFor(r, eps[ci], cellOf(r, eps[ci])));
          cc?.setCross(ci);
        },
        onLeave() { tip(null); cc?.setCross(null); },
        onClick({ ri, ci }) { select(rows[ri], eps[ci]); },
        onDblClick({ ri, ci }) {
          const c = cellOf(rows[ri], eps[ci]);
          if (c?.name && !c.missing) openReadMap(st.store, eps[ci].episode, c.name);
        },
        onRowClick(ri) { const r = rows[ri]; if (r.kind === "section") { update({ sel: r.name }); hm?.el.focus({ preventScroll: true }); } },
        onToggle(r) { const fold = new Set(st.fold); fold.has(r.key) ? fold.delete(r.key) : fold.add(r.key); update({ fold: fold.size ? fold : "-" }); },
        onBrush(a, b) { if (a == null) update({ e0: null, e1: null }); else update({ e0: eps[a].episode, e1: eps[b].episode }); },
        onColClick(ci) { update({ ep: eps[ci].episode === selEp ? null : eps[ci].episode }); },
        onMeanClick() { update({ sort: st.sort === "shift" ? null : "shift" }); },
        onKey(d) {
          if (!d) { update({ ep: null }); return; }
          const secRows = rows.map((r, i) => (r.kind === "section" ? i : -1)).filter((i) => i >= 0);
          const cur = Math.max(0, secRows.indexOf(selRi));
          const ri = secRows[Math.max(0, Math.min(secRows.length - 1, cur + d[0]))];
          const ci = Math.max(0, Math.min(eps.length - 1, (selCi < 0 ? eps.length - 1 : selCi) + d[1]));
          update({ sel: rows[ri].name, ep: eps[ci].episode });
          hm?.el.focus({ preventScroll: true });
        },
      });
      mapCard.append(hm.el, legend({
        title: "z against the reference", range: 1, bins, active: activeBins,
        onToggle(id, add) { const next = new Set(add ? activeBins : []); if (activeBins.has(id) && (add || activeBins.size === 1)) next.delete(id); else next.add(id); update({ bins: next }); },
        onHover(id) { hoverBin = id; hm.redraw(); },
        extra: [
          h("span", { class: "key" }, dot(), `alarm (|z| ≥ ${T})`),
          h("span", { class: "key" }, kindIcon("missing", 10), "missing"),
          h("span", { class: "key" }, h("span", { class: "swatch", style: { border: "1px solid var(--muted)", background: "transparent" } }), "new: no reference"),
          h("span", { class: "key muted" }, h("kbd", { text: "←↑↓→" }), " move the selection"),
        ],
      }));
      if (selRi >= 0) hm.setPin({ ri: selRi, ci: selCi >= 0 ? selCi : null });
    }

    content.replaceChildren(head, controls1, controls2, chartCard, mapCard);

    function linkHover(section, episode) {
      if (section == null) { hm?.setHover(null); cc?.setCross(null); return; }
      const ri = rows.findIndex((r) => r.key === `s:${section}`);
      const ci = eps.findIndex((e) => e.episode === episode);
      hm?.setHover({ ri: ri >= 0 ? ri : null, ci: ci >= 0 ? ci : null });
      cc?.setCross(section === sel && ci >= 0 ? ci : null);
    }
  }

  function followNewest() {
    if (!data) return;
    const T = st.thr;
    let best = null;
    for (const e of data.episodes) {
      if (baseline?.get(e.episode) === e.updated) continue;
      const z = Object.entries(e.z).filter(([, v]) => Math.abs(v) >= T).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0];
      const sec = e.missing[0] || z?.[0];
      if (sec && (!best || (e.updated || 0) > (best.e.updated || 0))) best = { e, sec };
    }
    if (best) update({ sel: best.sec, ep: best.e.episode, e0: null, e1: null }, false);
  }

  async function load(keep) {
    clearTimeout(timer);
    if (!st.store) {
      const stores = ctx.sources?.stores || [];
      feed.replaceChildren();
      content.replaceChildren(h("div", { class: "empty" }, h("p", { text: "Choose a store to monitor:" }),
        ...stores.map((s) => h("p", null, h("a", { href: href("/monitor", { store: s.id }), text: s.label })))));
      return;
    }
    if (keep) content.classList.add("loading");
    else content.replaceChildren(h("div", { class: "empty", text: "Computing read maps…" }));
    try {
      const qs = new URLSearchParams({ target: st.store, ref: st.ref, threshold: String(st.thr), harness_only: String(!st.conv) });
      data = await api(`/api/monitor?${qs}`);
    } catch (e) {
      if (!keep || !data) content.replaceChildren(h("div", { class: "empty", text: String(e) }));
      return;
    } finally {
      content.classList.remove("loading");
    }
    st = settings(route().params, data.sections);
    const done = [data.target.progress, data.reference.progress].every((p) => p.done >= p.total);
    const rp = data.reference.progress;
    if (rp.done < rp.total) {
      // alarms against a partly computed reference would be wrong (every section would look new)
      feed.replaceChildren();
      content.replaceChildren(h("div", { class: "empty" },
        h("p", { text: `Computing the reference store's read maps: ${rp.done} of ${rp.total} episodes.` }),
        h("span", { class: "progress" }, h("span", { class: "bar" }, h("i", { style: { width: `${(100 * rp.done) / Math.max(1, rp.total)}%` } }))),
        h("p", { class: "muted", text: "Alarms appear once the reference is complete; each read map is cached, so this happens once per episode." })));
      timer = setTimeout(() => load(true), 2500);
      return;
    }
    if (!baseline && done) baseline = new Map(data.episodes.map((e) => [e.episode, e.updated]));
    else if (keep && st.follow) followNewest();
    render();
    if (!done) timer = setTimeout(() => load(true), 2500);
  }

  let rt;
  const onResize = () => { clearTimeout(rt); rt = setTimeout(render, 150); };
  addEventListener("resize", onResize);
  load(false);

  return {
    refresh(changes) { if (changes.some((c) => c.store === st.store || c.store === st.ref || c.store === "*")) load(true); },
    destroy() { clearTimeout(timer); cancelAnimationFrame(raf); removeEventListener("resize", onResize); tip(null); },
  };
}

function dot() {
  const sv = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  sv.setAttribute("width", "10"); sv.setAttribute("height", "10"); sv.setAttribute("aria-hidden", "true");
  const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  c.setAttribute("cx", "5"); c.setAttribute("cy", "5"); c.setAttribute("r", "3.5");
  c.setAttribute("fill", "var(--ink)"); c.setAttribute("stroke", "var(--surface)"); c.setAttribute("stroke-width", "1.2");
  sv.append(c);
  return sv;
}
