// Section rows for the heatmaps: grouped by name prefix, with collapsible groups and headings,
// search, hidden groups, and ordering. Header rows stand for the union of their visible members.

import { h } from "./util.js";

const GROUP_LABELS = {
  "": "sections", obs: "observations", user: "user turns", self: "agent turns", fb: "harness feedback",
  tool_result: "tool results", system: "system prompt",
  gen: "generation", tool: "tools", policy: "policy", doc: "documents", demo: "demonstrations", sys: "system",
};

export const groupLabel = (g) => GROUP_LABELS[g] ?? g;

// Folded at first: the conversation, tool-schema lists over 20, any group over 50 (per-task documents), and
// headings over 20 (demonstrations per label).
export function defaultFold(sections) {
  const size = new Map(), under = new Map();
  for (const s of sections) {
    size.set(s.group, (size.get(s.group) || 0) + 1);
    if (s.parent && s.parent !== s.group) under.set(s.parent, (under.get(s.parent) || 0) + 1);
  }
  const big = (g) => g && size.get(g) > (g === "tool" ? 20 : 50);
  const out = new Set(sections.filter((s) => s.conversation || big(s.group)).map((s) => `g:${s.group}`));
  for (const [p, n] of under) if (n > 20) out.add(`p:${p}`);
  return out;
}

function shortName(sec, under) {
  if (under && sec.name.startsWith(under) && sec.name.length > under.length + 1) return sec.name.slice(under.length + 1);
  return sec.name;
}

// sections: [{name, group, parent, conversation}] in display order.
// opts: {fold: Set, hide: Set of groups, q, sort, grouped, stat: name -> number, tokens: name -> number,
//        stats: {sortName: name -> number}}. Sorts other than order and name are descending by their stat.
export function buildRows(sections, opts) {
  const q = (opts.q || "").toLowerCase();
  const visible = sections.filter((s) => !opts.hide.has(s.group) && (!q || s.name.toLowerCase().includes(q)));
  const index = new Map(sections.map((s, i) => [s.name, i]));
  const inOrder = (a, b) => index.get(a.name) - index.get(b.name);
  const stat = { mean: opts.stat, tokens: opts.tokens, ...opts.stats }[opts.sort];
  const by = opts.sort === "name" ? (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true })
    : stat ? (a, b) => (stat(b.name) ?? -Infinity) - (stat(a.name) ?? -Infinity) || inOrder(a, b)
      : inOrder;

  const row = (sec, depth, under) => ({
    key: `s:${sec.name}`, kind: "section", label: shortName(sec, under), name: sec.name, depth,
    members: [sec.name], group: sec.group, conversation: sec.conversation,
  });
  if (!opts.grouped) return [...visible].sort(by).map((s) => row(s, 0, ""));

  const groups = new Map();
  for (const s of visible) {
    if (!groups.has(s.group)) groups.set(s.group, []);
    groups.get(s.group).push(s);
  }
  const order = [...groups.keys()].sort((a, b) => {
    const ca = groups.get(a)[0].conversation, cb = groups.get(b)[0].conversation;
    return ca - cb || index.get(groups.get(a)[0].name) - index.get(groups.get(b)[0].name);
  });
  const rows = [];
  for (const g of order) {
    const secs = groups.get(g);
    const gkey = `g:${g}`;
    const gfold = opts.fold.has(gkey) && !q;
    rows.push({
      key: gkey, kind: "group", label: groupLabel(g), name: g, depth: 0, members: secs.map((s) => s.name),
      collapsed: gfold, group: g, conversation: secs[0].conversation, count: secs.length,
    });
    if (gfold) continue;
    const prefix = g ? `${g}:` : "";
    const parents = new Map();
    for (const s of secs) {
      const p = s.parent && s.parent !== g ? s.parent : "";
      if (!parents.has(p)) parents.set(p, []);
      parents.get(p).push(s);
    }
    const items = [];
    for (const [p, ps] of parents) {
      if (!p || ps.length < 2) {
        for (const s of ps) items.push({ sec: s });
      } else {
        items.push({ parent: p, secs: ps.sort(by) });
      }
    }
    const first = (it) => (it.sec ? it.sec : it.secs[0]);
    items.sort((a, b) => by(first(a), first(b)));
    for (const it of items) {
      if (it.sec) {
        rows.push(row(it.sec, 1, g ? g : ""));
        continue;
      }
      const pkey = `p:${it.parent}`;
      const pfold = opts.fold.has(pkey) && !q;
      rows.push({
        key: pkey, kind: "parent", label: it.parent.startsWith(prefix) ? it.parent.slice(prefix.length) : it.parent,
        name: it.parent, depth: 1, members: it.secs.map((s) => s.name), collapsed: pfold, group: g,
        conversation: it.secs[0].conversation, count: it.secs.length,
      });
      if (!pfold) for (const s of it.secs) rows.push(row(s, 2, it.parent));
    }
  }
  return rows;
}

// Union of member cells: lift is exact for the union (token-weighted mean); peak is the largest member
// peak; consult ratio and contrast are attention-mass-weighted means of the members'.
export function aggregate(cells) {
  if (!cells.length) return null;
  if (cells.length === 1) return cells[0];
  let tok = 0, mass = 0, crw = 0, crm = 0, ccw = 0, ccm = 0, peak = null, hot = null;
  for (const c of cells) {
    const t = c.tokens || 0, m = (c.lift ?? 0) * t;
    tok += t;
    mass += m;
    if (c.consult_ratio != null) { crw += c.consult_ratio * m; crm += m; }
    if (c.consult_contrast != null) { ccw += c.consult_contrast * m; ccm += m; }
    if (c.peak != null && (peak == null || c.peak > peak)) { peak = c.peak; hot = c.hot_line ?? hot; }
  }
  return {
    tokens: tok, lift: tok ? mass / tok : null, peak, hot_line: hot,
    consult_ratio: crm ? crw / crm : null, consult_contrast: ccm ? ccw / ccm : null, n: cells.length,
  };
}

// Group chips: click hides or shows a group, double-click shows only that group.
export function groupChips(sections, hide, onChange) {
  const groups = [];
  for (const s of sections) {
    let g = groups.find((x) => x.g === s.group);
    if (!g) groups.push(g = { g: s.group, conv: s.conversation, n: 0 });
    g.n++;
  }
  groups.sort((a, b) => a.conv - b.conv);
  return h("div", { class: "chips", role: "group", "aria-label": "Section groups" }, groups.map((g) => h("button", {
    class: `chip ${hide.has(g.g) ? "off" : ""}`, "aria-pressed": String(!hide.has(g.g)),
    title: "Click to hide or show; double-click to show only this group",
    onclick: (ev) => {
      if (ev.detail > 1) return;
      const next = new Set(hide);
      next.has(g.g) ? next.delete(g.g) : next.add(g.g);
      onChange(next);
    },
    ondblclick: () => onChange(new Set(groups.filter((x) => x.g !== g.g).map((x) => x.g))),
  }, groupLabel(g.g), h("span", { class: "n", text: String(g.n) }))));
}
