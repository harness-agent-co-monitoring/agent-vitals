// Control chart: one section's read lift per target episode against the reference normal (median, and
// median ± threshold × spread in log2 lift), with the reference episodes as a dot strip on the left. Hover
// snaps to an episode, click selects it, drag zooms, double-click resets.

import { divergingColor } from "./color.js";
import { fmt, h, s, tip } from "./util.js";

const PAD = { l: 46, r: 52, t: 10, b: 44 }, REF_W = 64, GAP = 14;

export function kindIcon(kind, size = 12) {
  const c = size / 2, r = size * 0.36;
  const sv = s("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}`, "aria-hidden": "true", class: "kind-icon" });
  if (kind === "over-read") sv.append(s("path", { d: `M${c} ${c - r}L${c + r} ${c + r * 0.8}L${c - r} ${c + r * 0.8}Z`, fill: divergingColor(0.85), stroke: "var(--surface)", "stroke-width": 1 }));
  else if (kind === "under-read") sv.append(s("path", { d: `M${c} ${c + r}L${c + r} ${c - r * 0.8}L${c - r} ${c - r * 0.8}Z`, fill: divergingColor(-0.85), stroke: "var(--surface)", "stroke-width": 1 }));
  else if (kind === "missing") sv.append(s("path", { d: `M${c - r} ${c - r}L${c + r} ${c + r}M${c + r} ${c - r}L${c - r} ${c + r}`, stroke: "var(--ink)", "stroke-width": 1.8, "stroke-linecap": "round" }));
  else if (kind === "new") sv.append(s("path", { d: `M${c} ${c - r}V${c + r}M${c - r} ${c}H${c + r}`, stroke: "var(--ink)", "stroke-width": 1.8, "stroke-linecap": "round" }));
  else sv.append(s("circle", { cx: c, cy: c, r: r * 0.8, fill: "var(--ink-2)", stroke: "var(--surface)", "stroke-width": 1.5 }));
  return sv;
}

// episodes: [{episode, reward, label}], values: lift per episode (null when absent), z: per episode,
// missing: bool per episode, band: {median, spread, threshold} in log2 units or null for a new section,
// ref: [[episode, lift]].
export function controlChart({ width, height = 230, episodes, values, z, missing, band, ref = [], hidden = new Set(),
  onToggle, onHover, onClick, onRefClick, onBrush, selCi = null, title }) {
  const W = Math.max(360, width), H = height;
  const showRef = !hidden.has("ref") && ref.length > 0 && band;
  const ex0 = PAD.l + (showRef ? REF_W + GAP : 0), ex1 = W - PAD.r;
  const n = episodes.length, step = (ex1 - ex0) / Math.max(1, n);
  const X = (i) => ex0 + (i + 0.5) * step;

  const logs = values.filter((v) => v > 0).map(Math.log2);
  if (showRef) logs.push(...ref.map(([, v]) => v).filter((v) => v > 0).map(Math.log2));
  if (band) logs.push(band.median - band.threshold * band.spread, band.median + band.threshold * band.spread);
  let lo = Math.min(...logs), hi = Math.max(...logs);
  if (!logs.length) { lo = -1; hi = 1; }
  if (hi - lo < 1) { const m = (hi + lo) / 2; lo = m - 0.5; hi = m + 0.5; }
  const pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;
  const plotB = H - PAD.b;
  const Y = (l) => PAD.t + (1 - (l - lo) / (hi - lo)) * (plotB - PAD.t);

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": title || "control chart", class: "cc" });
  const g = s("g", { class: "grid" }), axis = s("g", { class: "axis" });
  for (const e of ratioTicks(lo, hi, Y)) {
    g.append(s("line", { x1: PAD.l, x2: ex1, y1: Y(e), y2: Y(e) }));
    axis.append(s("text", { x: PAD.l - 6, y: Y(e) + 3.5, "text-anchor": "end" }, fmt.x(2 ** e)));
  }
  svg.append(g);

  if (band && !hidden.has("band")) {
    const top = Y(band.median + band.threshold * band.spread), bot = Y(band.median - band.threshold * band.spread);
    const x0 = showRef ? PAD.l : ex0;
    svg.append(
      s("rect", { class: "band", x: x0, y: top, width: ex1 - x0, height: bot - top }),
      s("line", { class: "band-edge", x1: x0, x2: ex1, y1: top, y2: top }),
      s("line", { class: "band-edge", x1: x0, x2: ex1, y1: bot, y2: bot }),
      s("line", { class: "median", x1: x0, x2: ex1, y1: Y(band.median), y2: Y(band.median) }),
      s("text", { class: "band-label", x: ex1 + 6, y: top + 3.5 }, `z +${band.threshold}`),
      s("text", { class: "band-label", x: ex1 + 6, y: Y(band.median) + 3.5 }, "normal"),
      s("text", { class: "band-label", x: ex1 + 6, y: bot + 3.5 }, `z −${band.threshold}`));
  }
  if (lo < 0 && hi > 0) svg.append(s("line", { class: "base", x1: PAD.l, x2: ex1, y1: Y(0), y2: Y(0) }));
  svg.append(axis);

  // reference strip
  const refDots = [];
  if (showRef) {
    const cx0 = PAD.l + REF_W / 2;
    svg.append(s("line", { class: "sep", x1: PAD.l + REF_W + GAP / 2, x2: PAD.l + REF_W + GAP / 2, y1: PAD.t, y2: plotB }));
    svg.append(s("text", { class: "lane-label", x: cx0, y: plotB + 14, "text-anchor": "middle" }, "reference"));
    ref.forEach(([ep, v], i) => {
      if (!(v > 0)) return;
      const jitter = ((hash(ep) % 1000) / 1000 - 0.5) * (REF_W - 18);
      const dot = s("circle", { class: "refdot", cx: cx0 + jitter, cy: Y(Math.log2(v)), r: 3.2 });
      refDots.push({ ep, v, x: cx0 + jitter, y: Y(Math.log2(v)), el: dot, i });
      svg.append(dot);
    });
  }

  // target episodes
  if (!hidden.has("target")) {
    let d = "", pen = false;
    values.forEach((v, i) => {
      if (!(v > 0)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${X(i).toFixed(1)},${Y(Math.log2(v)).toFixed(1)}`;
      pen = true;
    });
    svg.append(s("path", { class: "trace", d }));
    values.forEach((v, i) => {
      if (!(v > 0)) return;
      const kind = band && z[i] != null && Math.abs(z[i]) >= band.threshold ? (z[i] > 0 ? "over-read" : "under-read") : null;
      svg.append(marker(kind, X(i), Y(Math.log2(v)), step));
    });
  }
  missing.forEach((m, i) => {
    if (!m) return;
    const r = Math.min(4, step / 2 - 1), x = X(i), y = plotB - 8;
    svg.append(s("path", { class: "miss", d: `M${x - r} ${y - r}L${x + r} ${y + r}M${x + r} ${y - r}L${x - r} ${y + r}` }));
  });

  // x axis: episode numbers and outcome squares
  svg.append(s("line", { class: "xaxis", x1: ex0, x2: ex1, y1: plotB, y2: plotB }));
  const every = Math.max(1, Math.ceil(28 / step));
  episodes.forEach((e, i) => {
    if (i % every === 0 || i === n - 1) svg.append(s("text", { class: "tick", x: X(i), y: plotB + 13, "text-anchor": "middle" }, String(e.index + 1)));
    if (e.reward != null) {
      const sz = Math.max(4, Math.min(8, step - 3));
      svg.append(s("rect", { x: X(i) - sz / 2, y: plotB + 20, width: sz, height: sz, rx: 1.5, fill: e.reward > 0 ? "var(--good)" : "var(--critical)" }));
    }
  });
  svg.append(s("text", { class: "lane-label", x: 2, y: plotB + 27 }, "outcome"));
  svg.append(s("text", { class: "lane-label", x: 2, y: plotB + 13 }, "episode"));

  const pin = s("line", { class: "pinline", y1: PAD.t, y2: plotB, visibility: selCi == null ? "hidden" : "visible", x1: X(selCi ?? 0), x2: X(selCi ?? 0) });
  const cross = s("line", { class: "cross", y1: PAD.t, y2: plotB, visibility: "hidden" });
  const ring = s("circle", { class: "hring", r: 7, visibility: "hidden" });
  const brushRect = s("rect", { class: "brush", y: PAD.t, height: plotB - PAD.t, visibility: "hidden" });
  const hit = s("rect", { x: showRef ? PAD.l : ex0, y: 0, width: ex1 - (showRef ? PAD.l : ex0), height: H, fill: "transparent", style: "cursor: crosshair" });
  svg.append(pin, cross, ring, brushRect, hit);

  const px = (ev) => {
    const b = svg.getBoundingClientRect();
    return [((ev.clientX - b.left) / b.width) * W, ((ev.clientY - b.top) / b.height) * H];
  };
  const nearest = (x) => Math.max(0, Math.min(n - 1, Math.floor((x - ex0) / step)));
  function setCross(i) {
    if (i == null || i < 0 || i >= n) {
      cross.setAttribute("visibility", "hidden");
      ring.setAttribute("visibility", "hidden");
      return;
    }
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i));
    cross.setAttribute("visibility", "visible");
    if (values[i] > 0) {
      ring.setAttribute("cx", X(i)); ring.setAttribute("cy", Y(Math.log2(values[i])));
      ring.setAttribute("visibility", "visible");
    } else ring.setAttribute("visibility", "hidden");
  }

  let down = null, refHover = null;
  hit.addEventListener("pointerdown", (ev) => {
    const [x] = px(ev);
    if (x < ex0) return;
    down = { x, i: nearest(x), moved: false };
    hit.setPointerCapture(ev.pointerId);
  });
  hit.addEventListener("pointermove", (ev) => {
    const [x, y] = px(ev);
    if (down) {
      if (Math.abs(x - down.x) > 4) down.moved = true;
      if (down.moved) {
        const a = Math.min(down.i, nearest(x)), b = Math.max(down.i, nearest(x));
        brushRect.setAttribute("x", X(a) - step / 2);
        brushRect.setAttribute("width", (b - a + 1) * step);
        brushRect.setAttribute("visibility", "visible");
      }
    }
    if (x < ex0 && showRef) {
      setCross(null);
      onHover?.(null);
      let best = null, bd = 144;
      for (const d of refDots) { const dd = (d.x - x) ** 2 + (d.y - y) ** 2; if (dd < bd) { bd = dd; best = d; } }
      refHover?.el.classList.remove("on");
      refHover = best;
      if (best) {
        best.el.classList.add("on");
        tip(ev, { title: best.ep, sub: "reference episode", rows: [[fmt.x(best.v), "read lift"]] });
      } else tip(null);
      return;
    }
    refHover?.el.classList.remove("on");
    refHover = null;
    const i = nearest(x);
    setCross(i);
    const e = episodes[i];
    const rows = [[fmt.x(values[i]), "read lift"]];
    if (band) {
      rows.push([z[i] == null ? "–" : `${z[i] > 0 ? "+" : ""}${z[i].toFixed(1)}`, "z against the reference"]);
      rows.push([fmt.x(2 ** band.median), "reference normal"]);
      rows.push([`${fmt.x(2 ** (band.median - band.threshold * band.spread))}–${fmt.x(2 ** (band.median + band.threshold * band.spread))}`, "normal band"]);
    }
    tip(ev, {
      title: e.label,
      sub: `episode ${e.index + 1} · ${e.reward == null ? "no reward" : e.reward > 0 ? "pass" : "fail"}${missing[i] ? " · section missing" : values[i] == null ? " · section absent" : ""}`,
      rows,
    });
    onHover?.(i);
  });
  hit.addEventListener("pointerup", (ev) => {
    if (!down) return;
    const [x] = px(ev);
    const d = down;
    down = null;
    brushRect.setAttribute("visibility", "hidden");
    if (d.moved) {
      const a = Math.min(d.i, nearest(x)), b = Math.max(d.i, nearest(x));
      if (b > a) onBrush?.(a, b);
    } else onClick?.(d.i);
  });
  hit.addEventListener("click", () => { if (refHover) onRefClick?.(refHover.ep); });
  hit.addEventListener("dblclick", () => onBrush?.(null));
  hit.addEventListener("pointerleave", () => {
    setCross(null);
    refHover?.el.classList.remove("on");
    refHover = null;
    tip(null);
    onHover?.(null);
  });

  const toggles = [
    ["target", "target episodes", h("span", { class: "line", style: { background: "var(--ink-2)" } })],
    band && ref.length ? ["ref", "reference episodes", h("span", { class: "dotkey" })] : null,
    band ? ["band", `normal band (|z| < ${band.threshold})`, h("span", { class: "bandkey" })] : null,
  ].filter(Boolean);
  const legendEl = h("div", { class: "lc-legend" },
    toggles.map(([k, label, icon]) => h("button", {
      class: hidden.has(k) ? "off" : "", "aria-pressed": String(!hidden.has(k)), title: "Show or hide", onclick: () => onToggle?.(k),
    }, icon, label)),
    band ? h("span", { class: "key" }, kindIcon("over-read"), "over-read alarm") : null,
    band ? h("span", { class: "key" }, kindIcon("under-read"), "under-read alarm") : null,
    missing.some(Boolean) ? h("span", { class: "key" }, kindIcon("missing"), "missing") : null);
  return { el: h("div", { class: "lc cc-wrap" }, legendEl, svg), setCross };
}

function marker(kind, x, y, step) {
  if (!kind) return s("circle", { class: "pt", cx: x, cy: y, r: step < 10 ? 3 : 4 });
  const r = step < 10 ? 4.5 : 6;
  const d = kind === "over-read"
    ? `M${x} ${y - r}L${x + r} ${y + r * 0.75}L${x - r} ${y + r * 0.75}Z`
    : `M${x} ${y + r}L${x + r} ${y - r * 0.75}L${x - r} ${y - r * 0.75}Z`;
  return s("path", { class: "alarm-pt", d, fill: divergingColor(kind === "over-read" ? 0.85 : -0.85) });
}

// Round ratios (1, 1.5, 2, 3, 5 and 7 per decade) inside [lo, hi] in log2 units, at least 26px apart.
function ratioTicks(lo, hi, Y) {
  const out = [];
  for (let m = -4; m <= 4; m++) {
    for (const k of [1, 1.5, 2, 3, 5, 7]) {
      const l = Math.log2(k * 10 ** m);
      if (l >= lo && l <= hi && (!out.length || Math.abs(Y(out.at(-1)) - Y(l)) >= 26)) out.push(l);
    }
  }
  return out;
}

function hash(str) {
  let x = 2166136261;
  for (let i = 0; i < str.length; i++) x = Math.imul(x ^ str.charCodeAt(i), 16777619);
  return x >>> 0;
}
