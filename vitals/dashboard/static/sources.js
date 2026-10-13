// Sources: every trace store and Meta-Harness run found under the dashboard's roots.

import { fmt, go, h, liveDot, route, setParams, table } from "./util.js";

export function sourcesView(root, ctx) {
  function render() {
    const src = ctx.sources || { stores: [], runs: [] };
    const p = route().params;
    const byId = Object.fromEntries(src.stores.map((s) => [s.id, s]));
    const storesCard = h("div", { class: "card" },
      h("div", { class: "card-head" }, h("h2", { text: `Trace stores (${src.stores.length})` }),
        h("span", { class: "muted", text: "Click a store to open its episodes' read maps." })),
      src.stores.length
        ? table([
          { key: "label", label: "Store", cls: "wrap", render: (s) => h("span", null, s.live ? h("span", { class: "dot live", style: { marginRight: "6px" } }) : null, s.label) },
          { key: "episodes", label: "Episodes", num: true },
          { key: "replayed", label: "Replayed", num: true },
          { key: "pass_rate", label: "Pass rate", num: true, fmt: fmt.pct },
          { key: "updated", label: "Updated", num: true, fmt: fmt.ago },
          { key: "path", label: "Path", cls: "wrap mono muted" },
        ], src.stores, { sort: p.sort, onSort: (v) => { setParams({ sort: v }); render(); }, onRow: (s) => go("/episode", { store: s.id }) })
        : h("div", { class: "empty", text: "No trace stores found under the roots." }));
    const runsCard = src.runs.length
      ? h("div", { class: "card" },
        h("div", { class: "card-head" }, h("h2", { text: `Meta-Harness runs (${src.runs.length})` }),
          h("span", { class: "muted", text: "Click a run to open its evolution; each candidate's replayed store is listed with it." })),
        table([
          { key: "label", label: "Run", cls: "wrap" },
          { key: "stores", label: "Candidate stores", cls: "wrap", value: (r) => r.stores.length,
            render: (r) => h("div", { class: "chips" }, r.stores.map((sid) => byId[sid]).filter(Boolean).map((s) => h("a", {
              class: "chip", href: `#/store?store=${s.id}`, onclick: (ev) => ev.stopPropagation(),
            }, liveDot(s.live), s.label.split("/").at(-1) || s.label, h("span", { class: "n", text: String(s.episodes) })))) },
          { key: "path", label: "Path", cls: "wrap mono muted" },
        ], src.runs, { onRow: (r) => go("/evolution", { run: r.id }) }))
      : null;
    root.replaceChildren(h("div", { class: "content" },
      h("div", { class: "head" }, h("h1", { text: "Sources" }),
        h("div", { class: "facts muted", text: "Stores and runs are rediscovered every 30 seconds; episodes are checked every 2 seconds." })),
      storesCard, runsCard));
  }
  render();
  return { refresh() { render(); }, destroy() {} };
}
