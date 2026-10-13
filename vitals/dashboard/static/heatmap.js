// Canvas heatmap: section rows x columns (turns or episodes), with a header strip aligned to the columns,
// a per-row mean bar, hover crosshair, click-to-pin, keyboard moves, collapsible header rows, and a brush
// on the strip.

import { cssVar, divergingColor, theme } from "./color.js";
import { h } from "./util.js";

const MEAN_W = 92;

export function heatmap(opts) {
  const {
    rows, ncols, cell, strip, mean, meanTitle = "mean", corner = "",
    onHover, onClick, onDblClick, onToggle, onRowClick, onBrush, onColClick, onKey, onMeanClick, onLeave,
    texture = false, marks = true, pinned = null, avail = 900,
  } = opts;
  const dpr = window.devicePixelRatio || 1;
  const ch = rows.length > 150 ? 12 : rows.length > 60 ? 15 : 18;
  const labelW = opts.labelW || labelWidth(rows);
  const cw = Math.max(6, Math.min(40, Math.floor((avail - labelW - MEAN_W - 4) / Math.max(1, ncols))));
  const W = cw * ncols, H = ch * rows.length, SH = strip?.height || 0;
  const gap = cw >= 12 && ch >= 15 ? 2 : 1;

  const canvas = (w, hgt, cls) => {
    const c = h("canvas", { class: cls || null, width: Math.max(1, Math.round(w * dpr)), height: Math.max(1, Math.round(hgt * dpr)) });
    c.style.width = `${w}px`;
    c.style.height = `${hgt}px`;
    const ctx = c.getContext("2d");
    ctx.scale(dpr, dpr);
    return [c, ctx];
  };
  const [cells, cx] = canvas(W, H);
  const [over, ox] = canvas(W, H, "overlay");
  const [stripC, sx] = canvas(W, SH);
  const [meanC, mx] = canvas(MEAN_W, H);
  let brushing = null;

  const th = theme();
  const ink = cssVar("--ink"), ink2 = cssVar("--ink-2"), grid = cssVar("--grid"), surface = cssVar("--surface"), muted = cssVar("--muted");

  // -- cells ----------------------------------------------------------------------------------------------
  function drawCells() {
    cx.clearRect(0, 0, W, H);
    for (let ri = 0; ri < rows.length; ri++) {
      const r = rows[ri];
      const y = ri * ch;
      if (r.kind === "group" && ri > 0) {
        cx.fillStyle = grid;
        cx.fillRect(0, y, W, 1);
      }
      for (let ci = 0; ci < ncols; ci++) {
        const c = cell(ri, ci);
        if (!c) continue;
        const x = ci * cw;
        cx.globalAlpha = c.faded ? 0.16 : 1;
        if (c.fill) {
          cx.fillStyle = c.fill;
          cx.fillRect(x + gap / 2, y + gap / 2, cw - gap, ch - gap);
          if (texture && c.t != null && Math.abs(c.t) > 0.15) hatch(cx, x + gap / 2, y + gap / 2, cw - gap, ch - gap, c.t, th);
        } else if (c.mark !== "x") {
          cx.strokeStyle = muted;
          cx.lineWidth = 1;
          cx.strokeRect(x + gap / 2 + 1.5, y + gap / 2 + 1.5, cw - gap - 3, ch - gap - 3);
        }
        if (c.mark === "x") {
          const r = Math.max(2, Math.min(4, Math.min(cw, ch) / 2 - 3)), mx0 = x + cw / 2, my0 = y + ch / 2;
          cx.strokeStyle = ink;
          cx.lineWidth = 1.6;
          cx.beginPath();
          cx.moveTo(mx0 - r, my0 - r); cx.lineTo(mx0 + r, my0 + r);
          cx.moveTo(mx0 + r, my0 - r); cx.lineTo(mx0 - r, my0 + r);
          cx.stroke();
        } else if (marks && c.mark && cw >= 7) {
          const rad = Math.max(2, Math.min(3.2, ch / 5));
          const mx0 = x + cw - gap / 2 - rad - 1.5, my0 = y + ch / 2;
          cx.beginPath();
          cx.arc(mx0, my0, rad, 0, Math.PI * 2);
          cx.fillStyle = c.mark === "low" ? surface : ink;
          cx.fill();
          cx.lineWidth = 1.2;
          cx.strokeStyle = c.mark === "low" ? ink : surface;
          cx.stroke();
        }
      }
    }
    cx.globalAlpha = 1;
  }

  // -- header strip ---------------------------------------------------------------------------------------
  function drawStrip(hl) {
    if (!strip) return;
    sx.clearRect(0, 0, W, SH);
    strip.draw(sx, { cw, W, H: SH, hl, ink, ink2, muted, grid, surface });
    if (brushing) {
      const [a, b] = [Math.min(brushing.a, brushing.b), Math.max(brushing.a, brushing.b)];
      sx.fillStyle = ink;
      sx.globalAlpha = 0.12;
      sx.fillRect(a * cw, 0, (b - a + 1) * cw, SH);
      sx.globalAlpha = 1;
    }
  }

  // -- mean bars --------------------------------------------------------------------------------------------
  function drawMean(hlRow) {
    mx.clearRect(0, 0, MEAN_W, H);
    const mid = MEAN_W / 2, half = MEAN_W / 2 - 8;
    for (let ri = 0; ri < rows.length; ri++) {
      const m = mean?.(ri);
      const y = ri * ch;
      if (ri === hlRow) {
        mx.fillStyle = grid;
        mx.fillRect(0, y, MEAN_W, ch);
      }
      if (m && m.bar != null) {
        // magnitude in [0, 1], drawn from the left edge
        mx.fillStyle = m.fill || ink2;
        const bh = Math.max(4, ch - 6);
        roundRect(mx, 8, y + (ch - bh) / 2, Math.max(1.5, m.bar * (MEAN_W - 16)), bh, 2);
        mx.fill();
        continue;
      }
      if (!m || m.t == null) continue;
      const w = m.t * half;
      mx.fillStyle = divergingColor(m.t, th) || muted;
      const bh = Math.max(4, ch - 6);
      const x0 = w >= 0 ? mid : mid + w;
      roundRect(mx, x0, y + (ch - bh) / 2, Math.max(1.5, Math.abs(w)), bh, 2);
      mx.fill();
    }
    mx.fillStyle = ink2;
    mx.fillRect(opts.meanBars ? 7.5 : mid - 0.5, 0, 1, H);
  }

  // -- overlay: hover crosshair and pin -------------------------------------------------------------------
  let hover = null;
  let pin = pinned;
  function drawOverlay() {
    ox.clearRect(0, 0, W, H);
    if (hover) {
      ox.fillStyle = ink;
      ox.globalAlpha = 0.07;
      if (hover.ri != null) ox.fillRect(0, hover.ri * ch, W, ch);
      if (hover.ci != null) ox.fillRect(hover.ci * cw, 0, cw, H);
      ox.globalAlpha = 1;
      if (hover.ri != null && hover.ci != null) outline(ox, hover.ci * cw, hover.ri * ch, cw, ch, ink, 1.5);
    }
    if (pin && pin.ri != null) {
      if (pin.ci != null) {
        outline(ox, pin.ci * cw - 1, pin.ri * ch - 1, cw + 2, ch + 2, surface, 2);
        outline(ox, pin.ci * cw, pin.ri * ch, cw, ch, ink, 2);
      } else {
        outline(ox, 0, pin.ri * ch, W, ch, ink, 1.5);
      }
    }
  }

  // -- DOM ------------------------------------------------------------------------------------------------------
  const labels = h("div", { class: "hm-labels", style: { width: `${labelW}px` } });
  rows.forEach((r, ri) => {
    const caret = r.kind === "group" || r.kind === "parent" ? (r.collapsed ? "▸" : "▾") : "";
    const el = h("div", {
      class: `hm-row ${r.kind === "section" ? "" : "group"}`,
      style: { height: `${ch}px`, lineHeight: `${ch}px`, fontSize: ch < 15 ? "10.5px" : null, paddingLeft: `${4 + r.depth * 12}px` },
      title: r.name || r.label, "data-ri": ri,
    },
    h("span", { class: "caret", text: caret }),
    h("span", { class: "lbl" }, highlight(r.label, opts.q)),
    r.count ? h("span", { class: "cnt", text: String(r.count) }) : null);
    el.addEventListener("pointerenter", (ev) => setHover({ ri, ci: null }, ev, true));
    el.addEventListener("pointermove", (ev) => onHover?.({ ri, ci: null, ev, label: true }));
    el.addEventListener("click", () => (r.kind === "section" ? onRowClick?.(ri) : onToggle?.(r)));
    labels.append(el);
  });
  labels.addEventListener("pointerleave", () => clearHover());

  const stripWrap = h("div", { class: "hm-strip" }, stripC);
  const mhead = h("div", { class: "hm-mhead", title: "Sort rows by this column", onclick: () => onMeanClick?.() }, meanTitle);
  const cellsWrap = h("div", { class: "hm-cells" }, cells, over);
  const meanWrap = h("div", { class: "hm-mean" }, meanC);
  const grid_ = h("div", {
    class: "hm-grid",
    style: { gridTemplateColumns: `${labelW}px ${W}px ${MEAN_W}px`, gridTemplateRows: `${SH}px ${H}px` },
  }, h("div", { class: "hm-corner" }, corner), stripWrap, mhead, labels, cellsWrap, meanWrap);
  const root = h("div", { class: "hm", tabindex: "0", role: "grid", "aria-label": opts.label || "heatmap" }, grid_);

  const at = (ev, el) => {
    const b = el.getBoundingClientRect();
    return [ev.clientX - b.left, ev.clientY - b.top];
  };
  function setHover(hv, ev, fromLabel) {
    hover = hv;
    drawOverlay();
    drawStrip(hv?.ci);
    drawMean(hv?.ri);
    labels.querySelectorAll(".hm-row.hl").forEach((e) => e.classList.remove("hl"));
    if (hv?.ri != null) labels.children[hv.ri]?.classList.add("hl");
    if (!fromLabel) onHover?.({ ...hv, ev });
  }
  function clearHover() {
    hover = null;
    drawOverlay();
    drawStrip(null);
    drawMean(null);
    labels.querySelectorAll(".hm-row.hl").forEach((e) => e.classList.remove("hl"));
    onLeave?.();
  }

  cells.addEventListener("pointermove", (ev) => {
    const [x, y] = at(ev, cells);
    const ri = Math.min(rows.length - 1, Math.floor(y / ch)), ci = Math.min(ncols - 1, Math.floor(x / cw));
    if (!hover || hover.ri !== ri || hover.ci !== ci) setHover({ ri, ci }, ev);
    else onHover?.({ ri, ci, ev });
  });
  cells.addEventListener("pointerleave", clearHover);
  const cellAt = (ev) => {
    const [x, y] = at(ev, cells);
    const ri = Math.floor(y / ch), ci = Math.floor(x / cw);
    return cells.isConnected && ri >= 0 && ri < rows.length && ci >= 0 && ci < ncols ? { ri, ci } : null;
  };
  // With a double-click action, a single click waits until it cannot be the start of one.
  let clickTimer = null;
  cells.addEventListener("click", (ev) => {
    const hit = cellAt(ev);
    if (!hit) return;
    if (!onDblClick) { onClick?.(hit); return; }
    clearTimeout(clickTimer);
    if (ev.detail > 1) return;
    clickTimer = setTimeout(() => onClick?.(hit), 230);
  });
  cells.addEventListener("dblclick", (ev) => {
    clearTimeout(clickTimer);
    const hit = cellAt(ev);
    if (hit) onDblClick?.(hit);
  });

  meanC.addEventListener("pointermove", (ev) => {
    const [, y] = at(ev, meanC);
    const ri = Math.min(rows.length - 1, Math.floor(y / ch));
    if (!hover || hover.ri !== ri || hover.ci != null) setHover({ ri, ci: null }, ev, true);
    onHover?.({ ri, ci: null, ev, mean: true });
  });
  meanC.addEventListener("pointerleave", clearHover);
  meanC.addEventListener("click", (ev) => onRowClick?.(Math.floor(at(ev, meanC)[1] / ch)));

  stripC.addEventListener("pointerdown", (ev) => {
    const ci = Math.floor(at(ev, stripC)[0] / cw);
    brushing = onBrush ? { a: ci, b: ci } : null;
    stripC.setPointerCapture(ev.pointerId);
  });
  stripC.addEventListener("pointermove", (ev) => {
    const ci = Math.max(0, Math.min(ncols - 1, Math.floor(at(ev, stripC)[0] / cw)));
    if (brushing) brushing.b = ci;
    if (!hover || hover.ci !== ci || hover.ri != null) setHover({ ri: null, ci }, ev, true);
    else drawStrip(ci);
    onHover?.({ ri: null, ci, ev, strip: true });
  });
  stripC.addEventListener("pointerup", () => {
    if (!brushing) return;
    const { a, b } = brushing;
    brushing = null;
    drawStrip(hover?.ci);
    if (a !== b) onBrush?.(Math.min(a, b), Math.max(a, b));
    else onColClick?.(a);
  });
  stripC.addEventListener("pointerleave", () => { if (!brushing) clearHover(); });
  stripC.addEventListener("dblclick", () => onBrush?.(null));

  root.addEventListener("keydown", (ev) => {
    const dirs = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    if (dirs[ev.key]) {
      ev.preventDefault();
      onKey?.(dirs[ev.key]);
    } else if (ev.key === "Escape") onKey?.(null);
  });

  drawCells();
  drawStrip(null);
  drawMean(null);
  drawOverlay();

  return {
    el: root, cw, ch, labelW,
    setPin(p) {
      pin = p;
      drawOverlay();
      labels.querySelectorAll(".hm-row.pinned").forEach((e) => e.classList.remove("pinned"));
      if (p?.ri != null) labels.children[p.ri]?.classList.add("pinned");
      if (p?.ri != null) {
        const y = p.ri * ch, top = root.scrollTop + SH, bottom = root.scrollTop + root.clientHeight;
        if (y < top - SH || y + ch > bottom - SH) root.scrollTop = Math.max(0, y - root.clientHeight / 2);
      }
    },
    setHover(hv) { hover = hv; drawOverlay(); drawStrip(hv?.ci); },
    redraw() { drawCells(); drawOverlay(); },
  };
}

// Pass/fail square with a check or cross, for strips whose columns are episodes.
export function outcomeMark(c, x, cw, y, reward) {
  if (reward == null) return;
  const pass = reward > 0;
  const sz = Math.min(10, cw - 2);
  c.fillStyle = pass ? "#0ca30c" : "#d03b3b";
  c.fillRect(x + (cw - sz) / 2, y, sz, sz);
  if (sz < 8) return;
  const cx0 = x + cw / 2, cy0 = y + sz / 2;
  c.strokeStyle = "#ffffff";
  c.lineWidth = 1.5;
  c.beginPath();
  if (pass) { c.moveTo(cx0 - 2.5, cy0); c.lineTo(cx0 - 0.5, cy0 + 2); c.lineTo(cx0 + 2.8, cy0 - 2); } else { c.moveTo(cx0 - 2, cy0 - 2); c.lineTo(cx0 + 2, cy0 + 2); c.moveTo(cx0 + 2, cy0 - 2); c.lineTo(cx0 - 2, cy0 + 2); }
  c.stroke();
}

// Width of the label column for these rows; pass the largest to heatmaps whose columns should line up.
export function labelWidth(rows) {
  return Math.min(300, Math.max(170, Math.max(0, ...rows.map((r) => r.label.length + r.depth * 2)) * 6.4 + 34));
}

function outline(ctx, x, y, w, hgt, color, lw) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.strokeRect(x + lw / 2, y + lw / 2, w - lw, hgt - lw);
}

function roundRect(ctx, x, y, w, hgt, r) {
  const rr = Math.min(r, w / 2, hgt / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + hgt, rr);
  ctx.arcTo(x + w, y + hgt, x, y + hgt, rr);
  ctx.arcTo(x, y + hgt, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

// Texture channel: 45 degree lines above 1x, 135 degree below, denser with magnitude.
function hatch(ctx, x, y, w, hgt, t, th) {
  const n = Math.abs(t) > 0.66 ? 3 : Math.abs(t) > 0.33 ? 2 : 1;
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, hgt);
  ctx.clip();
  ctx.strokeStyle = th === "dark" ? "rgba(255,255,255,0.55)" : "rgba(0,0,0,0.45)";
  ctx.lineWidth = 1;
  const step = Math.max(3, (w + hgt) / (n * 2));
  for (let k = -hgt; k < w + hgt; k += step) {
    ctx.beginPath();
    if (t > 0) { ctx.moveTo(x + k, y + hgt); ctx.lineTo(x + k + hgt, y); } else { ctx.moveTo(x + k, y); ctx.lineTo(x + k + hgt, y + hgt); }
    ctx.stroke();
  }
  ctx.restore();
}

function highlight(text, q) {
  if (!q) return text;
  const i = text.toLowerCase().indexOf(q.toLowerCase());
  if (i < 0) return text;
  return [text.slice(0, i), h("mark", { text: text.slice(i, i + q.length) }), text.slice(i + q.length)];
}

// Diverging legend whose bins filter the cells: click a bin to show only cells in it, shift-click to add.
export function binsFor(range) {
  const R = Math.max(1, Math.ceil(range - 1e-9));
  const edges = [];
  for (let e = -(R - 1); e <= R - 1; e++) edges.push(e);
  const bins = [];
  const label = (e) => (e >= 0 ? `${2 ** e}` : `1/${2 ** -e}`);
  bins.push({ id: `<${edges[0]}`, lo: -Infinity, hi: edges[0], text: `<${label(edges[0])}×` });
  for (let i = 0; i < edges.length - 1; i++) {
    bins.push({ id: `${edges[i]}`, lo: edges[i], hi: edges[i + 1], text: `${label(edges[i])}–${label(edges[i + 1])}×` });
  }
  bins.push({ id: `>${edges.at(-1)}`, lo: edges.at(-1), hi: Infinity, text: `>${label(edges.at(-1))}×` });
  if (R === 1) {
    return [{ id: "<0", lo: -Infinity, hi: 0, text: "<1×" }, { id: ">0", lo: 0, hi: Infinity, text: "≥1×" }];
  }
  return bins;
}

export function binOf(bins, v) {
  if (v == null || !(v > 0)) return null;
  const l = Math.log2(v);
  return bins.find((b) => l >= b.lo && l < b.hi)?.id ?? null;
}

// bins default to the ratio bins for the range; explicit bins carry their swatch position as t.
export function legend({ title, range, bins = binsFor(range), active, onToggle, onHover, extra = [] }) {
  const wrap = h("div", { class: "bins", role: "group", "aria-label": `${title} filter` });
  for (const b of bins) {
    const mid = b.t != null ? b.t * range : Number.isFinite(b.lo) && Number.isFinite(b.hi) ? (b.lo + b.hi) / 2 : Number.isFinite(b.lo) ? b.lo + 0.5 : b.hi - 0.5;
    const on = !active.size || active.has(b.id);
    const el = h("button", {
      class: `bin ${on ? "" : "off"} ${active.has(b.id) ? "solo" : ""}`,
      title: "Show only cells in this range (shift-click to add ranges, click again to clear)",
      "aria-pressed": String(active.has(b.id)),
      onclick: (ev) => onToggle(b.id, ev.shiftKey),
    }, h("span", { class: "sw", style: { background: b.color || divergingColor(Math.max(-1, Math.min(1, mid / range))) } }), h("span", { class: "t", text: b.text }));
    el.addEventListener("pointerenter", () => onHover?.(b.id));
    el.addEventListener("pointerleave", () => onHover?.(null));
    wrap.append(el);
  }
  return h("div", { class: "legend" }, h("span", { class: "title", text: title }), wrap, ...extra);
}
