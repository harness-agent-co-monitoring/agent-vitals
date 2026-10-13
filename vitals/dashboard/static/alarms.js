// Alarms: the flags written by `vitals pipeline` for a store (check alarms and probe flags). A flag opens its
// explanation with both views side by side: the read map before the action (where the agent looked at the decision
// and over the reasoning before it, whatever the behavior, with z against this store's decisions) and, per probe,
// the probe split (which sections push toward the flagged behavior, as probe lift and z against each section's
// normal, with the read share, the read lift over the probe's layers and read z for that behavior beside it).

import { divergingColor } from "./color.js";
import { api, fmt, h, href, route, setParams, statusIcon, table, tip } from "./util.js";
import { bar } from "./why.js";

const SEV = { alert: ["critical", "alert"], warn: ["warning", "warn"] };
const sgn = (v, d = 1) => (v == null ? "–" : `${v >= 0 ? "+" : ""}${v.toFixed(d)}`);
const num = (v, d = 2) => (v == null ? "–" : v.toFixed(d));
const opt = (v, row) => (v == null ? [] : [row]);

function sevCell(sev) {
  const [kind, label] = SEV[sev] || ["good", sev || "info"];
  return h("span", { class: "status" }, statusIcon(kind), h("span", { text: label }));
}

// explanations before version 3 named read z z_decision / z_reasoning
const readZ = (r, foot) => r[`read_z_${foot}`] ?? r[`z_${foot}`];

function readRows(rm) {
  const liftKey = rm.lift_by || "read_lift";
  const zFoot = (rm.z_by || "read_z_decision").endsWith("reasoning") ? "reasoning" : "decision";
  const maxL = Math.max(1e-9, ...rm.rows.map((r) => r[liftKey] || 0));
  return rm.rows.map((r) => {
    const turnPeak = r.turn_peak ?? r.peak, turnHot = r.turn_hot_line ?? r.hot_line;
    const zR = readZ(r, "reasoning"), zD = readZ(r, "decision");
    const row = h("div", { class: "xp-row" },
      h("span", { class: "nm", title: r.section, text: r.section }),
      h("span", { class: "num", text: sgn(readZ(r, zFoot)) }),
      bar((r[liftKey] || 0) / maxL, "var(--muted)", false),
      h("span", { class: "num muted", text: num(r[liftKey], 1) }));
    row.addEventListener("pointermove", (ev) => tip(ev, {
      title: r.section,
      rows: [
        ...opt(r.reasoning_lift, [num(r.reasoning_lift), "read lift over the reasoning"]),
        [num(r.read_lift), "read lift at the decision"],
        ...opt(zR, [sgn(zR, 2), "read z over the reasoning, against this store's decisions"]),
        ...opt(zD, [sgn(zD, 2), "read z at the decision, against this store's decisions"]),
        ...opt(r.read_reasoning, [fmt.pct(r.read_reasoning), "share of attention over the reasoning"]),
        [fmt.pct(r.read), "share of attention at the decision"],
        ...opt(r.reason_peak, [num(r.reason_peak, 1), "peak lift over the reasoning"]),
        ...opt(r.turn_lift, [num(r.turn_lift), "read lift over the turn"]),
        ...opt(turnPeak, [num(turnPeak, 1), "peak lift over the turn"]),
        ...opt(r.consult_contrast, [num(r.consult_contrast), "consult contrast over the turn"]),
        [String(r.tokens), "tokens"],
        ...opt(r.picked_by, [(r.picked_by || []).join(", "), "in the top by"]),
      ],
      quote: r.reason_hot_line || turnHot || undefined,
    }));
    row.addEventListener("pointerleave", () => tip(null));
    return row;
  });
}

function probeRows(p) {
  const maxL = Math.max(1e-9, ...p.rows.map((r) => Math.abs(r.lift || 0)));
  return p.rows.map((r) => {
    const t = (r.lift || 0) / maxL;
    const row = h("div", { class: "xp-row" },
      h("span", { class: "nm", title: r.section, text: r.section }),
      h("span", { class: "num", text: sgn(r.z) }),
      bar(t, divergingColor(t), true),
      h("span", { class: "num muted", text: fmt.pct(r.read) }));
    row.addEventListener("pointermove", (ev) => tip(ev, {
      title: r.section,
      rows: [
        [sgn(r.lift, 2), "probe lift (signed, length-normalized)"],
        [sgn(r.z, 2), "z against the section's normal"],
        [sgn(r.contribution, 3), "contribution to the score"],
        ...opt(r.contribution_linear, [sgn(r.contribution_linear, 3), "of which DeltaNet layers"]),
        [fmt.pct(r.read), "share of attention (read map, all attention layers)"],
        ...opt(r.read_layers, [fmt.pct(r.read_layers), `share of attention, layers ≤ ${p.layer} (what the split sees)`]),
        ...opt(r.read_lift_layers, [num(r.read_lift_layers), `read lift, layers ≤ ${p.layer}`]),
        ...opt(r.read_z_decision, [sgn(r.read_z_decision, 2), "read z at the decision, for this behavior"]),
        ...opt(r.read_z_reasoning, [sgn(r.read_z_reasoning, 2), "read z over the reasoning, for this behavior"]),
        [String(r.tokens), "tokens"],
      ],
    }));
    row.addEventListener("pointerleave", () => tip(null));
    return row;
  });
}

function explanation(ex, sid) {
  const d = ex.decision || {};
  const head = h("div", { class: "card-head" },
    h("h2", { text: `${ex.episode} · turn ${ex.turn}` }),
    d.a_text != null ? h("span", { class: "muted" }, "decision ", h("span", { class: "mono", text: d.a_text }), " over ",
      h("span", { class: "mono", text: d.b_text })) : null,
    h("span", { class: "spacer" }),
    h("a", { href: href("/episode", { store: sid, episode: ex.episode, t: ex.turn }), text: "Open the read map" }));
  const rm = ex.read_map;
  const overReasoning = rm.lift_by === "reasoning_lift";
  const shown = [`read lift${overReasoning ? " over the reasoning" : " at the decision"}`,
    ...(rm.normal ? ["z against this store's decisions"] : []), `peak${rm.peak_by === "reason_peak" ? " over the reasoning" : " over the turn"}`];
  const views = [h("div", { class: "pane-sec" },
    h("h4", null, "Read map before the action", h("span", { class: "spacer" }), h("span", { class: "muted", text: "where the agent looked" })),
    h("div", { class: "xp-row xp-head" }, h("span", { text: "section" }), h("span", { text: "z" }),
      h("span", { text: overReasoning ? "read lift over the reasoning" : "read lift at the decision" }), h("span", { text: "lift" })),
    h("div", { class: "xp-rows" }, readRows(rm)),
    h("div", { class: "muted xp-note", text: `Rows: the top sections by ${shown.join(", by ")}. ${fmt.pct(rm.reasoning_share)} of the attention at the decision goes to the turn's own reasoning.` }))];
  for (const p of ex.probes) {
    const conf = ex.confirmed?.[p.probe];
    views.push(h("div", { class: "pane-sec" },
      h("h4", null, `Probe split · ${p.probe}`, h("span", { class: "spacer" }),
        p.severity ? sevCell(p.severity) : h("span", { class: "muted", text: "below threshold" }),
        h("span", { class: "muted", text: `score ${p.score >= 0 ? "+" : ""}${p.score.toFixed(2)} / ${p.threshold == null ? "–" : (p.threshold >= 0 ? "+" : "") + p.threshold.toFixed(2)}` })),
      h("div", { class: "xp-row xp-head" }, h("span", { text: "section" }), h("span", { text: "z" }),
        h("span", { text: "← against · toward →" }), h("span", { text: "read" })),
      h("div", { class: "xp-rows" }, probeRows(p)),
      h("div", { class: "muted xp-note", text: `ranked by ${p.ranked_by}; ${fmt.pct(p.remainder_share)} of the score comes from outside the section split (embedding, MLPs${p.rows.some((r) => r.contribution_linear != null) ? "" : ", DeltaNet layers"}).` }),
      conf && conf.confirmed && Object.keys(conf.confirmed).length
        ? h("div", { class: "xp-note", text: "Confirmed by ablation: " + Object.entries(conf.confirmed).map(([s, v]) => `without ${s} ${conf.score.toFixed(2)} → ${v.toFixed(2)}`).join("; ") })
        : null));
  }
  return h("div", { class: "card" }, head,
    ex.claims.length ? h("ul", { class: "why-claims xp-claims" }, ex.claims.map((c) => h("li", { text: c }))) : null,
    h("div", { class: "xp-grid" }, views));
}

export function alarmsView(root, ctx) {
  let data = null, exps = {}, alive = true;
  const st = () => {
    const p = route().params;
    return { store: p.store || ctx.sources?.stores[0]?.id, flag: p.flag || null, sev: p.sev || "", kind: p.kind || "", sort: p.asort || "" };
  };

  async function load() {
    const { store } = st();
    if (!store) { data = null; render(); return; }
    data = await api(`/api/stores/${store}/alarms`);
    exps = {};
    if (alive) render();
  }

  async function explanationFor(f) {
    const { store } = st();
    if (!exps[f.episode]) exps[f.episode] = await api(`/api/stores/${store}/episodes/${encodeURIComponent(f.episode)}/flags`);
    return exps[f.episode].find((x) => x.turn === f.turn) || null;
  }

  function render() {
    const s = st();
    const stores = ctx.sources?.stores || [];
    const picker = h("select", { "aria-label": "Trace store", onchange: (e) => { setParams({ store: e.target.value, flag: null }); load(); } },
      stores.map((x) => { const o = h("option", { value: x.id, text: x.label }); if (x.id === s.store) o.selected = true; return o; }));
    const filt = (key, opts, label) => h("select", { "aria-label": label, onchange: (e) => { setParams({ [key]: e.target.value || null }); render(); } },
      opts.map(([v, t]) => { const o = h("option", { value: v, text: t }); if (v === s[key]) o.selected = true; return o; }));
    const head = h("div", { class: "head" }, h("h1", { text: "Alarms" }),
      h("div", { class: "ctl" }, picker, filt("sev", [["", "All severities"], ["alert", "Alert"], ["warn", "Warn"]], "Severity"),
        filt("kind", [["", "Checks and probes"], ["check", "Checks"], ["probe", "Probes"]], "Kind")),
      h("div", { class: "facts muted", text: data?.last_run ? `pipeline last ran ${data.last_run}` : "" }));
    if (!data || !data.pipeline) {
      root.replaceChildren(h("div", { class: "content" }, head, h("div", { class: "card" }, h("div", { class: "empty" },
        "No pipeline output for this store yet. Run ", h("span", { class: "mono", text: "vitals pipeline --store <store> --gguf <model> [--probes p.json]" }), "."))));
      return;
    }
    const flags = data.flags.filter((f) => (!s.sev || f.severity === s.sev) && (!s.kind || f.kind === s.kind));
    const counts = { alert: flags.filter((f) => f.severity === "alert").length, warn: flags.filter((f) => f.severity === "warn").length };
    const list = h("div", { class: "card" },
      h("div", { class: "card-head" }, h("h2", { text: `Flags (${flags.length})` }),
        h("span", { class: "status" }, statusIcon("critical"), `${counts.alert} alert`),
        h("span", { class: "status" }, statusIcon("warning"), `${counts.warn} warn`),
        h("span", { class: "muted", text: "Click a flag for its explanation." })),
      flags.length ? table([
        { key: "severity", label: "Severity", value: (f) => (f.severity === "alert" ? 2 : 1), render: (f) => sevCell(f.severity) },
        { key: "episode", label: "Episode", cls: "mono" },
        { key: "turn", label: "Turn", num: true },
        { key: "check", label: "Check", cls: "mono" },
        { key: "message", label: "Message", cls: "wrap" },
        { key: "explained", label: "Explained", render: (f) => h("span", { class: "muted", text: f.explained ? "yes" : "–" }) },
      ], flags, { sort: s.sort, onSort: (v) => { setParams({ asort: v }); render(); }, onRow: (f) => { setParams({ flag: f.id }); render(); } })
        : h("div", { class: "empty", text: "No flags match." }));
    const detail = h("div", null);
    root.replaceChildren(h("div", { class: "content" }, head, list, detail));
    const f = data.flags.find((x) => x.id === s.flag);
    if (f) {
      if (!f.explained) { detail.replaceChildren(h("div", { class: "card" }, h("div", { class: "empty", text: f.turn == null ? "Store-level flag (section drift): no single decision to explain." : "No decision was captured at this turn (no action onset)." }))); return; }
      explanationFor(f).then((ex) => { if (alive && ex) detail.replaceChildren(explanation(ex, s.store)); });
    }
  }

  load();
  return { refresh(changed) { if (!changed || changed.length) load(); }, destroy() { alive = false; } };
}
