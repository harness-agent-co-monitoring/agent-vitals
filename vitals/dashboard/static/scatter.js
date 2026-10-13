// Candidate scatter: score against iteration (with the best-so-far line and parent links) or against context
// size (with the Pareto frontier). Hover snaps to the nearest point, click selects, dragging selects an x
// range, double-click clears it.

import { fmt, h, s, tip } from "./util.js";

const PAD = { l: 48, r: 18, t: 12, b: 34 };

// points: [{key, x, y, label, tip: {title, sub, rows, quote}}]; points without y are not drawn.
export function scatterChart({ width, height = 250, points, xLog = false, xFmt = String, yFmt = fmt.num, xLabel, frontier = null,
  links = [], hidden = new Set(), onToggle, onHover, onClick, onBrush, sel = null, range = null, title, integerX = false }) {
  const W = Math.max(320, width), H = height;
  const pts = points.filter((p) => p.y != null && p.x != null && (!xLog || p.x > 0));
  const tx = (x) => (xLog ? Math.log10(x) : x);
  const xs = pts.map((p) => tx(p.x)), ys = pts.map((p) => p.y);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (!pts.length) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
  const xpad = integerX ? 0.5 : (x1 - x0 || 1) * 0.06, ypad = (y1 - y0 || Math.abs(y1) || 1) * 0.12;
  x0 -= xpad; x1 += xpad; y0 -= ypad; y1 += ypad;
  const X = (x) => PAD.l + ((tx(x) - x0) / (x1 - x0)) * (W - PAD.l - PAD.r);
  const Y = (y) => PAD.t + (1 - (y - y0) / (y1 - y0)) * (H - PAD.t - PAD.b);
  const unX = (px) => {
    const v = x0 + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (x1 - x0);
    return xLog ? 10 ** v : v;
  };

  // points sharing an x spread sideways so they stay separate
  const byX = new Map();
  for (const p of pts) { const k = String(p.x); if (!byX.has(k)) byX.set(k, []); byX.get(k).push(p); }
  const pos = new Map();
  for (const group of byX.values()) {
    group.forEach((p, i) => pos.set(p.key, [X(p.x) + (i - (group.length - 1) / 2) * 7, Y(p.y)]));
  }

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": title || "scatter", class: "sc" });
  const grid = s("g", { class: "grid" }), axis = s("g", { class: "axis" });
  const ystep = niceStep((y1 - y0) / 4);
  for (let v = Math.ceil(y0 / ystep) * ystep; v <= y1; v += ystep) {
    grid.append(s("line", { x1: PAD.l, x2: W - PAD.r, y1: Y(v), y2: Y(v) }));
    axis.append(s("text", { x: PAD.l - 6, y: Y(v) + 3.5, "text-anchor": "end" }, yFmt(v)));
  }
  const xticks = [];
  if (xLog) {
    for (let e = Math.floor(x0); e <= Math.ceil(x1); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (tx(v) >= x0 && tx(v) <= x1) xticks.push(v); }
  } else {
    const step = integerX ? Math.max(1, Math.ceil((x1 - x0) / 10)) : niceStep((x1 - x0) / 6);
    for (let v = Math.ceil(x0 / step) * step; v <= x1; v += step) xticks.push(v);
  }
  for (const v of xticks) axis.append(s("text", { x: X(v), y: H - PAD.b + 14, "text-anchor": "middle" }, xFmt(v)));
  if (xLabel) axis.append(s("text", { class: "xlabel", x: (PAD.l + W - PAD.r) / 2, y: H - 4, "text-anchor": "middle" }, xLabel));
  svg.append(grid, s("line", { class: "xaxis", x1: PAD.l, x2: W - PAD.r, y1: H - PAD.b, y2: H - PAD.b }), axis);

  if (range) {
    const a = Math.max(PAD.l, X(range[0])), b = Math.min(W - PAD.r, X(range[1]));
    svg.append(s("rect", { class: "outside", x: PAD.l, y: PAD.t, width: Math.max(0, a - PAD.l), height: H - PAD.t - PAD.b }),
      s("rect", { class: "outside", x: b, y: PAD.t, width: Math.max(0, W - PAD.r - b), height: H - PAD.t - PAD.b }));
  }

  if (!hidden.has("links")) {
    for (const [from, to] of links) {
      const a = pos.get(from), b = pos.get(to);
      if (a && b) svg.append(s("line", { class: "link", x1: a[0], y1: a[1], x2: b[0], y2: b[1] }));
    }
  }

  const front = frontierPoints(pts, frontier);
  if (front.length && !hidden.has("frontier")) {
    let d = "";
    front.forEach((p, i) => {
      const [px, py] = pos.get(p.key);
      if (i === 0) d += `M${px},${py}`;
      else d += frontier === "best" ? `H${px}V${py}` : `L${px},${py}`;
    });
    if (frontier === "best") d += `H${W - PAD.r}`;
    svg.append(s("path", { class: "frontier", d }));
  }
  const onFront = new Set(front.map((p) => p.key));
  for (const p of pts) {
    const [px, py] = pos.get(p.key);
    svg.append(s("circle", { class: `pt ${onFront.has(p.key) ? "front" : ""}`, cx: px, cy: py, r: 5 }));
  }
  const best = pts.reduce((b, p) => (!b || p.y > b.y ? p : b), null);
  for (const p of [best, pts.find((q) => q.key === sel)]) {
    if (!p) continue;
    const [px, py] = pos.get(p.key);
    const right = px < W - 160;
    svg.append(s("text", { class: "plabel", x: px + (right ? 9 : -9), y: py - 7, "text-anchor": right ? "start" : "end" }, p.label));
  }
  if (sel && pos.get(sel)) { const [px, py] = pos.get(sel); svg.append(s("circle", { class: "selring", cx: px, cy: py, r: 8.5 })); }

  const ring = s("circle", { class: "hring", r: 8.5, visibility: "hidden" });
  const brush = s("rect", { class: "brush", y: PAD.t, height: H - PAD.t - PAD.b, visibility: "hidden" });
  const hit = s("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent", style: "cursor: crosshair" });
  svg.append(ring, brush, hit);

  const at = (ev) => { const b = svg.getBoundingClientRect(); return [((ev.clientX - b.left) / b.width) * W, ((ev.clientY - b.top) / b.height) * H]; };
  const nearest = (x, y) => {
    let bestP = null, bd = 18 * 18;
    for (const p of pts) { const [px, py] = pos.get(p.key); const d = (px - x) ** 2 + (py - y) ** 2; if (d < bd) { bd = d; bestP = p; } }
    return bestP;
  };
  function setHover(key) {
    const q = key && pos.get(key);
    if (!q) { ring.setAttribute("visibility", "hidden"); return; }
    ring.setAttribute("cx", q[0]); ring.setAttribute("cy", q[1]); ring.setAttribute("visibility", "visible");
  }
  let down = null, hoverKey = null;
  hit.addEventListener("pointerdown", (ev) => { const [x] = at(ev); down = { x, moved: false }; hit.setPointerCapture(ev.pointerId); });
  hit.addEventListener("pointermove", (ev) => {
    const [x, y] = at(ev);
    if (down && onBrush) {
      if (Math.abs(x - down.x) > 4) down.moved = true;
      if (down.moved) {
        brush.setAttribute("x", Math.min(x, down.x)); brush.setAttribute("width", Math.abs(x - down.x));
        brush.setAttribute("visibility", "visible");
      }
    }
    const p = nearest(x, y);
    if ((p?.key || null) !== hoverKey) { hoverKey = p?.key || null; setHover(hoverKey); onHover?.(hoverKey); }
    if (p) tip(ev, p.tip); else tip(null);
  });
  hit.addEventListener("pointerup", (ev) => {
    if (!down) return;
    const [x, y] = at(ev);
    const d = down;
    down = null;
    brush.setAttribute("visibility", "hidden");
    if (d.moved && onBrush) onBrush(unX(Math.min(x, d.x)), unX(Math.max(x, d.x)));
    else onClick?.(nearest(x, y)?.key || null);
  });
  hit.addEventListener("dblclick", () => onBrush?.(null));
  hit.addEventListener("pointerleave", () => { hoverKey = null; setHover(null); tip(null); onHover?.(null); });

  const toggles = [
    frontier ? ["frontier", frontier === "best" ? "best so far" : "Pareto frontier", h("span", { class: "line", style: { background: "var(--series-1)" } })] : null,
    links.length ? ["links", "parent → child", h("span", { class: "line", style: { background: "var(--axis)" } })] : null,
  ].filter(Boolean);
  const legend = h("div", { class: "lc-legend" },
    h("span", { class: "key" }, h("span", { class: "dotkey", style: { background: "var(--ink-2)" } }), "candidate"),
    frontier ? h("span", { class: "key" }, h("span", { class: "dotkey", style: { background: "var(--series-1)" } }), frontier === "best" ? "new best" : "on the frontier") : null,
    toggles.map(([k, label, icon]) => h("button", { class: hidden.has(k) ? "off" : "", "aria-pressed": String(!hidden.has(k)), title: "Show or hide", onclick: () => onToggle?.(k) }, icon, label)));
  return { el: h("div", { class: "lc sc-wrap" }, legend, svg), setHover };
}

function frontierPoints(pts, kind) {
  if (kind === "best") {
    const out = [];
    for (const p of [...pts].sort((a, b) => a.x - b.x || a.order - b.order)) if (!out.length || p.y > out.at(-1).y) out.push(p);
    return out;
  }
  if (kind === "pareto") {
    const out = [];
    for (const p of [...pts].sort((a, b) => a.x - b.x || b.y - a.y)) if (!out.length || p.y > out.at(-1).y) out.push(p);
    return out;
  }
  return [];
}

function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw || 1));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}
