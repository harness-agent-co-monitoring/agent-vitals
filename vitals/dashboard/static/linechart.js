// SVG line chart for ratios over turns (log2 y axis, 1x baseline), with a snapping crosshair, a tooltip
// listing every series, a pinned-x marker, and a legend that toggles series.

import { fmt, h, s, tip } from "./util.js";

const W = 372, PAD = { l: 40, r: 10, t: 8, b: 20 };

export function lineChart({ xs, series, hidden = new Set(), onToggle, height = 120, pinX = null, onHover, onClick,
  title, xLabel = "turn", format = fmt.x, log = true }) {
  const H = height;
  const shown = series.filter((sr) => !hidden.has(sr.key));
  const tf = (v) => (v == null || !(log ? v > 0 : Number.isFinite(v)) ? null : log ? Math.log2(v) : v);
  const vals = shown.flatMap((sr) => sr.values.map(tf)).filter((v) => v != null);
  let lo = Math.min(log ? 0 : Infinity, ...vals), hi = Math.max(log ? 0 : -Infinity, ...vals);
  if (!vals.length) { lo = -1; hi = 1; }
  if (hi - lo < (log ? 1 : 1e-9)) { const m = (hi + lo) / 2; lo = m - (log ? 0.5 : 0.5); hi = m + (log ? 0.5 : 0.5); }
  const padY = (hi - lo) * 0.08;
  lo -= padY; hi += padY;
  const x0 = xs[0] ?? 0, x1 = xs.at(-1) ?? 1;
  const X = (x) => PAD.l + (x1 === x0 ? (W - PAD.l - PAD.r) / 2 : ((x - x0) / (x1 - x0)) * (W - PAD.l - PAD.r));
  const Y = (v) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title || "line chart" });
  const gridG = s("g", { class: "grid" }), axisG = s("g", { class: "axis" });
  const ticks = [];
  if (log) {
    const step = hi - lo > 6 ? 2 : 1;
    for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) if (e % step === 0) ticks.push(e);
  } else {
    const step = niceStep((hi - lo) / 4);
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  }
  for (const e of ticks) {
    gridG.append(s("line", { x1: PAD.l, x2: W - PAD.r, y1: Y(e), y2: Y(e) }));
    axisG.append(s("text", { x: PAD.l - 6, y: Y(e) + 3.5, "text-anchor": "end" },
      log ? (e >= 0 ? `${2 ** e}×` : `1/${2 ** -e}×`) : format(e)));
  }
  const xstep = Math.max(1, Math.ceil(xs.length / 8));
  xs.forEach((x, i) => {
    if (i % xstep === 0 || i === xs.length - 1) {
      axisG.append(s("text", { x: X(x), y: H - 5, "text-anchor": "middle" }, String(x)));
    }
  });
  svg.append(gridG, axisG);
  if (log && lo < 0 && hi > 0) svg.append(s("line", { class: "base", x1: PAD.l, x2: W - PAD.r, y1: Y(0), y2: Y(0) }));
  if (pinX != null && xs.includes(pinX)) svg.append(s("line", { class: "pinline", x1: X(pinX), x2: X(pinX), y1: PAD.t, y2: H - PAD.b }));

  const seriesG = s("g", { class: "series" });
  for (const sr of shown) {
    let d = "", pen = false;
    sr.values.forEach((v, i) => {
      const t = tf(v);
      if (t == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(t).toFixed(1)}`;
      pen = true;
    });
    seriesG.append(s("path", { d, stroke: sr.color }));
    if (xs.length <= 40) {
      sr.values.forEach((v, i) => {
        const t = tf(v);
        if (t != null) seriesG.append(s("circle", { cx: X(xs[i]), cy: Y(t), r: 4, fill: sr.color, stroke: "var(--surface)", "stroke-width": 2 }));
      });
    }
  }
  svg.append(seriesG);

  const cross = s("line", { class: "cross", y1: PAD.t, y2: H - PAD.b, visibility: "hidden" });
  svg.append(cross);
  const hit = s("rect", { x: PAD.l - 8, y: 0, width: W - PAD.l - PAD.r + 16, height: H, fill: "transparent" });
  svg.append(hit);

  const nearest = (ev) => {
    const b = svg.getBoundingClientRect();
    const px = ((ev.clientX - b.left) / b.width) * W;
    let best = 0;
    xs.forEach((x, i) => { if (Math.abs(X(x) - px) < Math.abs(X(xs[best]) - px)) best = i; });
    return best;
  };
  function setCross(i) {
    if (i == null || i < 0) { cross.setAttribute("visibility", "hidden"); return; }
    cross.setAttribute("x1", X(xs[i]));
    cross.setAttribute("x2", X(xs[i]));
    cross.setAttribute("visibility", "visible");
  }
  hit.addEventListener("pointermove", (ev) => {
    const i = nearest(ev);
    setCross(i);
    tip(ev, {
      title: `${xLabel} ${xs[i]}`,
      rows: shown.map((sr) => [format(sr.values[i]), sr.label, sr.color]),
    });
    onHover?.(xs[i]);
  });
  hit.addEventListener("pointerleave", () => { setCross(null); tip(null); onHover?.(null); });
  hit.addEventListener("click", (ev) => onClick?.(xs[nearest(ev)]));

  const legendEl = series.length > 1
    ? h("div", { class: "lc-legend" }, series.map((sr) => h("button", {
      class: hidden.has(sr.key) ? "off" : "", "aria-pressed": String(!hidden.has(sr.key)), title: "Show or hide this series",
      onclick: () => onToggle?.(sr.key),
    }, h("span", { class: "line", style: { background: sr.color } }), sr.label)))
    : null;
  return { el: h("div", { class: "lc" }, legendEl, svg), setCross: (x) => setCross(xs.indexOf(x)) };
}

function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw || 1));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}
