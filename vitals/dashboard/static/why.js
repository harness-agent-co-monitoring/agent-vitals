// "Why this action": the explanation of a turn's action onset (vitals explain). One row per ablated section:
// the exact effect on the decision (signed bar, toward or away from the chosen token) beside its reading.

import { divergingColor } from "./color.js";
import { fmt, h, tip } from "./util.js";

const SVG = "http://www.w3.org/2000/svg";

export function bar(pos, color, centered) {
  const sv = document.createElementNS(SVG, "svg");
  sv.setAttribute("viewBox", "0 0 90 10"); sv.setAttribute("width", "90"); sv.setAttribute("height", "10");
  sv.setAttribute("aria-hidden", "true");
  const x0 = centered ? 45 : 0, span = centered ? 43 : 88;
  const w = pos * span;
  const r = document.createElementNS(SVG, "rect");
  r.setAttribute("x", String(w >= 0 ? x0 : x0 + w)); r.setAttribute("y", "1");
  r.setAttribute("width", String(Math.max(1.5, Math.abs(w)))); r.setAttribute("height", "8"); r.setAttribute("rx", "2");
  r.setAttribute("fill", color);
  sv.append(r);
  if (centered) {
    const base = document.createElementNS(SVG, "rect");
    base.setAttribute("x", "44.5"); base.setAttribute("y", "0"); base.setAttribute("width", "1"); base.setAttribute("height", "10");
    base.setAttribute("fill", "var(--ink-2)");
    sv.append(base);
  }
  return sv;
}

export function whySection(ex, { onSection, current } = {}) {
  const d = ex.decision;
  const ab = ex.sections.filter((s) => s.ablated).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  const maxD = Math.max(1e-9, ...ab.map((s) => Math.abs(s.delta)));
  const maxR = Math.max(1e-9, ...ex.sections.map((s) => s.reading_reasoning || 0));
  const sec = h("div", { class: "pane-sec why" },
    h("h4", null, "Why this action", h("span", { class: "spacer" }),
      h("span", { class: "muted", text: ex.method.calibrated ? "calibrated ranking · exact ablation" : "ranked by reading · exact ablation" })),
    h("div", { class: "why-decision" },
      h("span", { class: "mono", text: d.a_text }), h("span", { class: "muted", text: " over " }), h("span", { class: "mono", text: d.b_text }),
      h("span", { class: "muted", text: ` · P ${fmt.num(d.p_a, 3)} · margin ${d.margin >= 0 ? "+" : ""}${d.margin.toFixed(2)}` })));
  const head = h("div", { class: "why-row why-head" }, h("span", { text: "section" }), h("span", { text: "effect" }),
    h("span", { text: `← away · toward →` }), h("span", { text: "read" }));
  const rows = ab.map((s) => {
    const row = h("div", {
      class: `why-row ${s.section === current ? "on" : ""}`, tabindex: "0", role: "button",
      onclick: () => onSection?.(s.section),
      onkeydown: (ev) => { if (ev.key === "Enter") onSection?.(s.section); },
    },
    h("span", { class: "nm", title: s.section, text: s.section }),
    h("span", { class: "num", text: `${s.delta >= 0 ? "+" : ""}${s.delta.toFixed(2)}${s.flips ? " ⇄" : ""}` }),
    bar(s.delta / maxD, divergingColor(s.delta / maxD), true),
    bar((s.reading_reasoning || 0) / maxR, "var(--muted)", false));
    row.addEventListener("pointermove", (ev) => tip(ev, {
      title: s.section,
      rows: [
        [`${s.delta >= 0 ? "+" : ""}${s.delta.toFixed(2)}`, "effect on the logit margin"],
        [`${fmt.num(d.p_a, 3)} → ${fmt.num(s.p_without, 3)}`, `P(${d.a_text.trim()}) without it${s.flips ? " (flips)" : ""}`],
        ...(s.behavior_change != null ? [[`${s.behavior_change >= 0 ? "+" : ""}${s.behavior_change.toFixed(2)}`, "expected behavior change"]] : []),
        [fmt.pct(s.reading_reasoning), "read over the reasoning"],
        [fmt.pct(s.reading_at_decision), "read at the decision"],
        [s.excerpt.slice(0, 90), "text"],
      ],
    }));
    row.addEventListener("pointerleave", () => tip(null));
    return row;
  });
  sec.append(head, h("div", { class: "why-rows" }, rows));
  if (ex.claims.length) sec.append(h("ul", { class: "why-claims" }, ex.claims.map((c) => h("li", { text: c }))));
  const ck = ex.checks;
  const parts = [];
  if (ck.noop_delta != null) parts.push(`no-op check ${ck.noop_delta === 0 ? "exact (0)" : ck.noop_delta.toExponential(1)}`);
  if (ex.joint.delta != null) parts.push(`together ${ex.joint.delta >= 0 ? "+" : ""}${ex.joint.delta.toFixed(2)} vs ${ex.joint.sum_of_parts >= 0 ? "+" : ""}${ex.joint.sum_of_parts.toFixed(2)} summed${Math.abs(ck.additivity_gap) > 0.25 * Math.abs(ex.joint.sum_of_parts) ? " (the sections overlap)" : ""}`);
  parts.push(`${ex.method.ablations} ablations, ${ex.method.seconds}s`);
  sec.append(h("div", { class: "muted why-checks", text: parts.join(" · ") }));
  return sec;
}
