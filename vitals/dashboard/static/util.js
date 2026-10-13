export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const SVGNS = "http://www.w3.org/2000/svg";
export function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  for (const c of children.flat()) if (c != null) el.append(c);
  return el;
}

export const fmt = {
  x(v) {
    if (v == null || !Number.isFinite(v)) return "–";
    return (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + "×";
  },
  int(v) { return v == null ? "–" : Math.round(v).toLocaleString(); },
  num(v, d = 2) { return v == null || !Number.isFinite(v) ? "–" : v.toFixed(d); },
  pct(v) { return v == null ? "–" : `${Math.round(v * 100)}%`; },
  ago(ts) {
    if (!ts) return "–";
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 60) return `${Math.round(s)}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    if (s < 86400) return `${Math.round(s / 3600)}h ago`;
    return `${Math.round(s / 86400)}d ago`;
  },
};

export async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

// -- URL state: #/<view>?k=v&... ------------------------------------------------------------------------
export function route() {
  const hash = location.hash.slice(1) || "/sources";
  const [path, qs] = hash.split("?");
  return { path, params: Object.fromEntries(new URLSearchParams(qs || "")) };
}

export function href(path, params = {}) {
  const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v != null && v !== "")).toString();
  return `#${path}${qs ? `?${qs}` : ""}`;
}

// Update view settings in place (no re-render, no history entry).
export function setParams(patch) {
  const { path, params } = route();
  for (const [k, v] of Object.entries(patch)) {
    if (v == null || v === "") delete params[k];
    else params[k] = String(v);
  }
  history.replaceState(null, "", href(path, params));
}

export function go(path, params) {
  location.hash = href(path, params).slice(1);
}

export const list = (v) => (v ? v.split(",").filter(Boolean) : []);

// -- tooltip ----------------------------------------------------------------------------------------------
let tipEl;
export function tip(ev, { title, sub, rows = [], quote } = {}) {
  if (!tipEl) {
    tipEl = h("div", { class: "tip", role: "tooltip" });
    document.body.append(tipEl);
  }
  if (!ev) {
    tipEl.style.display = "none";
    return;
  }
  tipEl.replaceChildren(...[
    title ? h("div", { class: "tt", text: title }) : null,
    sub ? h("div", { class: "tsub", text: sub }) : null,
    rows.length
      ? h("table", null, rows.map(([v, k, color]) => h("tr", null,
        h("td", { class: "v", text: v }),
        h("td", { class: "k" }, color ? h("span", { class: "line", style: { background: color } }) : null, k))))
      : null,
    quote ? h("div", { class: "quote", text: quote }) : null,
  ].filter(Boolean));
  tipEl.style.display = "block";
  const pad = 14, w = tipEl.offsetWidth, ht = tipEl.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + ht > innerHeight - 8) y = ev.clientY - ht - pad;
  tipEl.style.left = `${Math.max(8, x)}px`;
  tipEl.style.top = `${Math.max(8, y)}px`;
}

// -- small widgets ----------------------------------------------------------------------------------------
export function seg(options, value, onChange, label) {
  const wrap = h("div", { class: "seg", role: "group", "aria-label": label });
  for (const [v, text, title] of options) {
    wrap.append(h("button", {
      class: String(v) === String(value) ? "on" : "", "aria-pressed": String(String(v) === String(value)), title,
      onclick: () => onChange(v),
    }, text));
  }
  return wrap;
}

export function checkbox(text, checked, onChange, title) {
  const input = h("input", { type: "checkbox", onchange: (e) => onChange(e.target.checked) });
  input.checked = !!checked;
  return h("label", { class: "check", title }, input, text);
}

export function statusIcon(kind) {
  const paths = {
    good: "M3 8.5l3 3 7-7",
    critical: "M4 4l8 8M12 4l-8 8",
    warning: "M8 3v6M8 12v.5",
    serious: "M8 3v6M8 12v.5",
  };
  const color = { good: "var(--good)", critical: "var(--critical)", warning: "var(--warning)", serious: "var(--serious)" }[kind];
  return s("svg", { width: 12, height: 12, viewBox: "0 0 16 16", "aria-hidden": "true" },
    s("path", { d: paths[kind], stroke: color, "stroke-width": 2.4, fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round" }));
}

export function reward(r) {
  if (r == null) return h("span", { class: "status muted", text: "no reward" });
  const pass = r > 0;
  return h("span", { class: "status", title: `reward ${r}` }, statusIcon(pass ? "good" : "critical"),
    h("span", { style: { color: pass ? "var(--good-text)" : "var(--critical)" }, text: pass ? `pass ${fmt.num(r, r % 1 ? 2 : 0)}` : `fail ${fmt.num(r, r % 1 ? 2 : 0)}` }));
}

export function liveDot(on) {
  return on ? h("span", { class: "status", title: "updated in the last 2 minutes" }, h("span", { class: "dot live" }), "live") : null;
}

// Sortable table: columns [{key, label, num, fmt, cls, title}], rows of objects.
export function table(columns, rows, { sort, onSort, onRow, limit = 2000 } = {}) {
  const [skey, sdir] = (sort || "").split(":");
  const col = columns.find((c) => c.key === skey);
  let data = rows;
  if (col) {
    const dir = sdir === "asc" ? 1 : -1;
    const val = col.value || ((r) => r[col.key]);
    data = [...rows].sort((a, b) => {
      const x = val(a), y = val(b);
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * dir;
    });
  }
  const thead = h("thead", null, h("tr", null, columns.map((c) => h("th", {
    class: [c.num ? "r" : "", c.key === skey ? "sorted" : "", c.key === skey && sdir === "asc" ? "asc" : ""].join(" "),
    title: c.title || "Sort", scope: "col",
    onclick: () => onSort && onSort(`${c.key}:${c.key === skey && sdir !== "asc" ? "asc" : "desc"}`),
  }, c.label))));
  const tbody = h("tbody", null, data.slice(0, limit).map((r) => h("tr", {
    class: onRow ? "click" : "", onclick: onRow ? () => onRow(r) : null,
  }, columns.map((c) => {
    const v = c.value ? c.value(r) : r[c.key];
    const cell = c.render ? c.render(r) : c.fmt ? c.fmt(v) : v ?? "–";
    return h("td", { class: [c.num ? "r" : "", c.cls || ""].join(" ") }, cell);
  }))));
  const out = h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" }, thead, tbody));
  if (data.length > limit) out.append(h("div", { class: "empty", text: `Showing ${limit} of ${data.length} rows.` }));
  return out;
}
