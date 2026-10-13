import { alarmsView } from "./alarms.js";
import { episodeView } from "./episode.js";
import { evolutionView } from "./evolution.js";
import { monitorView } from "./monitor.js";
import { sourcesView } from "./sources.js";
import { storeView } from "./store.js";
import { api, fmt, h, href, route, setParams } from "./util.js";

const VIEWS = {
  "/episode": { label: "Read map", make: episodeView },
  "/store": { label: "Store", make: storeView },
  "/monitor": { label: "Monitor", make: monitorView },
  "/alarms": { label: "Alarms", make: alarmsView },
  "/evolution": { label: "Evolution", make: evolutionView },
  "/sources": { label: "Sources", make: sourcesView },
};
const ctx = { sources: null };
let view = null, version = 0, lastChange = Date.now() / 1000, pollOk = true;
const main = document.getElementById("main");

function tabs() {
  const { path, params } = route();
  const store = params.store;
  const run = params.run || ctx.sources?.stores.find((s) => s.id === store)?.run || ctx.sources?.runs[0]?.id;
  const target = (p) => (p === "/sources" ? {} : p === "/evolution" ? { run } : { store });
  document.getElementById("tabs").replaceChildren(...Object.entries(VIEWS).map(([p, v]) => h("a", {
    href: href(p, target(p)), class: p === path ? "on" : "", "aria-current": p === path ? "page" : null,
  }, v.label)));
}

function mount() {
  view?.destroy();
  const { path } = route();
  const v = VIEWS[path] || VIEWS["/sources"];
  tabs();
  view = v.make(main, ctx);
  document.title = `${v.label} · agent-vitals`;
}

function themeButton() {
  const btn = document.getElementById("theme");
  const cur = document.documentElement.dataset.theme || "auto";
  btn.textContent = { auto: "Theme: auto", light: "Theme: light", dark: "Theme: dark" }[cur];
  btn.onclick = () => {
    const next = { auto: "light", light: "dark", dark: "auto" }[cur];
    if (next === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = next;
    try { next === "auto" ? localStorage.removeItem("vitals-theme") : localStorage.setItem("vitals-theme", next); } catch (e) { /* storage unavailable */ }
    setParams({ theme: next === "auto" ? null : next });
    themeButton();
    mount();
  };
}

function liveBar() {
  const n = ctx.sources?.stores.filter((s) => s.live).length || 0;
  document.getElementById("live").replaceChildren(...[
    pollOk ? null : h("span", { class: "status", style: { color: "var(--critical)" }, text: "server unreachable" }),
    n ? h("span", { class: "status" }, h("span", { class: "dot live" }), `${n} live`) : null,
    h("span", { class: "more", text: `${ctx.sources?.stores.length ?? 0} stores · last change ${fmt.ago(lastChange)}` }),
  ].filter(Boolean));
}

async function poll() {
  try {
    const ch = await api(`/api/changes?since=${version}`);
    pollOk = true;
    if (ch.version !== version) {
      const first = version === 0;
      version = ch.version;
      if (!first && ch.changed.length) {
        lastChange = Date.now() / 1000;
        ctx.sources = await api("/api/sources");
        await view?.refresh(ch.changed);
      }
    }
  } catch (e) {
    pollOk = false;
  }
  liveBar();
}

(async () => {
  ctx.sources = await api("/api/sources");
  version = ctx.sources.version;
  themeButton();
  addEventListener("hashchange", mount);
  mount();
  liveBar();
  setInterval(poll, 3000);
})();
