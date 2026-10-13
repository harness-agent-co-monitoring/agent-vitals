// Diverging color for ratios against 1x (read lift, peak lift, consult ratio): blue below, gray at 1x,
// red above, interpolated in OKLab. Each arm is a single-hue ramp stepped for the theme's surface.

const ARMS = {
  light: { mid: "#f0efec", under: ["#b7d3f6", "#6da7ec", "#256abf", "#0d366b"], over: ["#f6c4bc", "#ee8678", "#cf3e37", "#7f1a17"] },
  dark: { mid: "#383835", under: ["#22436b", "#2a69b8", "#5598e7", "#b7d3f6"], over: ["#5e2b28", "#a53a34", "#e5625b", "#f7b9b1"] },
};
const N = 128;

const toLin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toSrgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

function oklab(hex) {
  const n = parseInt(hex.slice(1), 16);
  const r = toLin(((n >> 16) & 255) / 255), g = toLin(((n >> 8) & 255) / 255), b = toLin((n & 255) / 255);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}

function rgb([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const c = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
  return c.map((v) => Math.round(255 * Math.min(1, Math.max(0, toSrgb(v)))));
}

function ramp(stops) {
  const labs = stops.map(oklab);
  return (t) => {
    const x = Math.min(1, Math.max(0, t)) * (labs.length - 1);
    const i = Math.min(labs.length - 2, Math.floor(x)), f = x - i;
    return rgb(labs[i].map((v, k) => v + (labs[i + 1][k] - v) * f));
  };
}

const luts = {};
function lut(theme) {
  if (!luts[theme]) {
    const a = ARMS[theme];
    const under = ramp([a.mid, ...a.under]), over = ramp([a.mid, ...a.over]);
    luts[theme] = { under: [], over: [] };
    for (let i = 0; i <= N; i++) {
      luts[theme].under.push(`rgb(${under(i / N).join(",")})`);
      luts[theme].over.push(`rgb(${over(i / N).join(",")})`);
    }
  }
  return luts[theme];
}

export function theme() {
  const t = document.documentElement.dataset.theme;
  if (t === "dark" || t === "light") return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

// Position of a ratio on the scale: log2(v) / range, clamped to [-1, 1].
export function scalePos(v, range) {
  if (v == null || !(v > 0)) return null;
  return Math.max(-1, Math.min(1, Math.log2(v) / range));
}

export function divergingColor(t, th = theme()) {
  if (t == null) return null;
  const L = lut(th);
  const i = Math.round(Math.abs(t) * N);
  return t < 0 ? L.under[i] : L.over[i];
}

// Text color that stays legible on a scale color.
export function inkOn(t, th = theme()) {
  if (t == null) return null;
  if (th === "dark") return Math.abs(t) > 0.75 ? "#0b0b0b" : null;
  return Math.abs(t) > 0.55 ? "#ffffff" : null;
}

export function ratioColor(v, range, th = theme()) {
  return divergingColor(scalePos(v, range), th);
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Sequential color for magnitudes in [0, 1] (pass rates): the blue arm alone, from the surface gray.
export function sequentialColor(t, th = theme()) {
  if (t == null || !Number.isFinite(t)) return null;
  return lut(th).under[Math.round(Math.max(0, Math.min(1, t)) * N)];
}
