// The app's themes: each the few things that make it look as it does, its fonts, its corners, how
// much room it leaves, and its colors, light and dark. A theme of one scheme alone is always that;
// whatever a theme leaves out is Leat's own, the first. Anyone may make one: a file of JSON, as
// these would be written, opened in Settings.

// a theme's fonts, sizes and corners, in pixels, and what each must be
const SHAPE = {
  font: "font", // of everything
  displayFont: "font", // of the greeting and the pages' titles
  displayWeight: [100, 900],
  codeFont: "font",
  size: [12, 22], // of the text
  radius: [0, 40], // of the controls, the code and the cards
  radiusLarge: [0, 40], // of the messages, the composer and the round buttons
  density: [0.75, 1.5], // the room around things, as a share of Leat's
};
// a scheme's colors: the window's, the sidebar's, the composer's and inputs', a message's and a
// hover's, the lines', the text's, the text said less loudly's, the accent's, the text on it's,
// an error's, a code's keywords', strings' and numbers'; and the composer's shadow
const COLORS = ["page", "side", "card", "bubble", "line", "text", "muted", "accent", "onAccent",
  "error", "keyword", "string", "number", "shadow"];
const SCHEMES = ["light", "dark"];
// the custom properties style.css takes them by
const PROPERTIES = { font: "--font", displayFont: "--font-display", displayWeight: "--display-weight",
  codeFont: "--font-mono", size: "--size", radius: "--round", radiusLarge: "--round-large",
  density: "--density", onAccent: "--on-accent" };
const PIXELS = new Set(["size", "radius", "radiusLarge"]);

const SANS = "system-ui, sans-serif";
const MONO = "ui-monospace, monospace";

const LEAT = {
  name: "Leat",
  font: SANS, displayFont: "ui-serif, Georgia, serif", displayWeight: 400, codeFont: MONO,
  size: 16, radius: 10, radiusLarge: 24, density: 1,
  light: {
    page: "#faf9f5", side: "#f3f1ea", card: "#fff", bubble: "#ece9e0", line: "#e3dfd4",
    text: "#1f1e1d", muted: "#75736c", accent: "#c96442", onAccent: "#fff", error: "#b3261e",
    keyword: "#a626a4", string: "#50a14f", number: "#986801",
    shadow: "0 4px 20px rgb(0 0 0 / 0.06)",
  },
  dark: {
    page: "#262624", side: "#1f1e1d", card: "#30302e", bubble: "#3a3936", line: "#3e3d39",
    text: "#ece9e1", muted: "#a3a097", accent: "#d97757", onAccent: "#fff", error: "#f2847b",
    keyword: "#c678dd", string: "#98c379", number: "#d19a66",
    shadow: "0 4px 20px rgb(0 0 0 / 0.06)",
  },
};

export const THEMES = [LEAT, ...[
  { // neutral grays, black on white or white on black, and pills
    name: "Graphite",
    font: `"Söhne", "Inter", ${SANS}`, displayFont: `"Söhne", "Inter", ${SANS}`, displayWeight: 500,
    radius: 12, radiusLarge: 28,
    light: {
      page: "#fff", side: "#f9f9f9", card: "#fff", bubble: "#f1f1f1", line: "#e6e6e6",
      text: "#0d0d0d", muted: "#6b6b6b", accent: "#0d0d0d", onAccent: "#fff", error: "#d93025",
      keyword: "#8b3fd9", string: "#18794e", number: "#b35900",
      shadow: "0 2px 12px rgb(0 0 0 / 0.08)",
    },
    dark: {
      page: "#212121", side: "#181818", card: "#2f2f2f", bubble: "#3a3a3a", line: "#3a3a3a",
      text: "#ececec", muted: "#a4a4a4", accent: "#ececec", onAccent: "#0d0d0d", error: "#f28b82",
      keyword: "#c49cff", string: "#7ee2a8", number: "#ffb86b",
      shadow: "0 2px 12px rgb(0 0 0 / 0.3)",
    },
  },
  { // dark alone: black, white and light type, wide open
    name: "Void",
    font: `"Geist", "Inter", ${SANS}`, displayFont: `"Geist", "Inter", ${SANS}`, displayWeight: 300,
    codeFont: `"Geist Mono", "JetBrains Mono", ${MONO}`, radius: 14, radiusLarge: 32, density: 1.1,
    dark: {
      page: "#000", side: "#0a0a0a", card: "#131313", bubble: "#1c1c1c", line: "#262626",
      text: "#f5f5f5", muted: "#8a8a8a", accent: "#f5f5f5", onAccent: "#000", error: "#ff6b6b",
      keyword: "#b4a1ff", string: "#9fe0b8", number: "#ffcf8a",
      shadow: "0 0 0 1px #262626, 0 8px 40px rgb(255 255 255 / 0.04)",
    },
  },
  { // a terminal's: monospace, square, gold on black, after Hermes Agent's
    name: "Hermes",
    font: `"JetBrains Mono", "IBM Plex Mono", "Iosevka", ${MONO}`,
    displayFont: `"JetBrains Mono", "IBM Plex Mono", "Iosevka", ${MONO}`, displayWeight: 700,
    codeFont: `"JetBrains Mono", "IBM Plex Mono", "Iosevka", ${MONO}`,
    size: 15, radius: 2, radiusLarge: 4, density: 0.9,
    light: {
      page: "#fbf8ef", side: "#f3eedf", card: "#fffdf6", bubble: "#efe7d1", line: "#ddd3b8",
      text: "#2a2619", muted: "#7a7158", accent: "#a86f00", onAccent: "#fffdf6", error: "#c0392b",
      keyword: "#a0522d", string: "#4f7a28", number: "#a86f00",
      shadow: "none",
    },
    dark: {
      page: "#0c0b08", side: "#100f0b", card: "#15140f", bubble: "#1e1c14", line: "#2f2c20",
      text: "#f2e8c9", muted: "#8f8870", accent: "#ffbf00", onAccent: "#0c0b08", error: "#ff5f56",
      keyword: "#cd7f32", string: "#a6d36a", number: "#ffd700",
      shadow: "none",
    },
  },
  { // a newspaper's: serif type, square corners, black rules and red ink
    name: "Newsprint",
    font: `"Charter", "Iowan Old Style", "Source Serif 4", Georgia, serif`,
    displayFont: `"Bodoni Moda", "Didot", "Playfair Display", Georgia, serif`, displayWeight: 700,
    codeFont: `"Courier Prime", "Courier New", ${MONO}`, size: 17, radius: 0, radiusLarge: 0,
    density: 1.1,
    light: {
      page: "#f4f1ea", side: "#ebe6da", card: "#fffdf8", bubble: "#e6dfcf", line: "#1a1a1a",
      text: "#111", muted: "#5c574d", accent: "#c8281a", onAccent: "#fff", error: "#c8281a",
      keyword: "#8a1c7c", string: "#2f6b2f", number: "#9a5b00",
      shadow: "4px 4px 0 #111",
    },
    dark: {
      page: "#161513", side: "#0f0e0d", card: "#1f1d1a", bubble: "#2a2724", line: "#e9e4d8",
      text: "#eee9dc", muted: "#a59f91", accent: "#ff5a3c", onAccent: "#111", error: "#ff5a3c",
      keyword: "#e48fd6", string: "#9ccc8a", number: "#f2b45c",
      shadow: "4px 4px 0 #e9e4d8",
    },
  },
  { // Nord's palette, its Snow Storm and Polar Night
    name: "Nord",
    font: `"Inter", ${SANS}`, displayFont: `"Inter", ${SANS}`, displayWeight: 600,
    radius: 8, radiusLarge: 16,
    light: {
      page: "#eceff4", side: "#e5e9f0", card: "#f8f9fb", bubble: "#dde3ec", line: "#d3d9e3",
      text: "#2e3440", muted: "#5e6a80", accent: "#5e81ac", onAccent: "#fff", error: "#bf616a",
      keyword: "#81659b", string: "#5f8040", number: "#c0703a",
      shadow: "0 4px 16px rgb(46 52 64 / 0.08)",
    },
    dark: {
      page: "#2e3440", side: "#2a2f3a", card: "#3b4252", bubble: "#434c5e", line: "#434c5e",
      text: "#eceff4", muted: "#9aa3b5", accent: "#88c0d0", onAccent: "#2e3440", error: "#d57780",
      keyword: "#81a1c1", string: "#a3be8c", number: "#b48ead",
      shadow: "0 4px 16px rgb(0 0 0 / 0.2)",
    },
  },
  { // Solarized's sixteen colors
    name: "Solarized",
    font: `"Source Sans 3", "Source Sans Pro", ${SANS}`,
    displayFont: `"Source Serif 4", "Source Serif Pro", ui-serif, Georgia, serif`, displayWeight: 600,
    codeFont: `"Source Code Pro", ${MONO}`, radius: 6, radiusLarge: 14,
    light: {
      page: "#fdf6e3", side: "#eee8d5", card: "#fffbef", bubble: "#eee8d5", line: "#ddd6c1",
      text: "#586e75", muted: "#93a1a1", accent: "#268bd2", onAccent: "#fdf6e3", error: "#dc322f",
      keyword: "#859900", string: "#2aa198", number: "#d33682",
      shadow: "0 2px 10px rgb(0 43 54 / 0.08)",
    },
    dark: {
      page: "#002b36", side: "#00252f", card: "#073642", bubble: "#0b4150", line: "#0e4452",
      text: "#93a1a1", muted: "#657b83", accent: "#268bd2", onAccent: "#fdf6e3", error: "#dc322f",
      keyword: "#859900", string: "#2aa198", number: "#d33682",
      shadow: "0 2px 10px rgb(0 0 0 / 0.25)",
    },
  },
  { // Catppuccin's Latte and Mocha: pastel, soft and round
    name: "Catppuccin",
    font: `"Nunito", "Quicksand", ${SANS}`, displayFont: `"Nunito", "Quicksand", ${SANS}`,
    displayWeight: 700, radius: 14, radiusLarge: 28, density: 1.05,
    light: {
      page: "#eff1f5", side: "#e6e9ef", card: "#f8f9fb", bubble: "#dce0e8", line: "#ccd0da",
      text: "#4c4f69", muted: "#6c6f85", accent: "#8839ef", onAccent: "#eff1f5", error: "#d20f39",
      keyword: "#8839ef", string: "#40a02b", number: "#fe640b",
      shadow: "0 4px 20px rgb(136 57 239 / 0.08)",
    },
    dark: {
      page: "#1e1e2e", side: "#181825", card: "#292a3c", bubble: "#383a4e", line: "#45475a",
      text: "#cdd6f4", muted: "#a6adc8", accent: "#cba6f7", onAccent: "#1e1e2e", error: "#f38ba8",
      keyword: "#cba6f7", string: "#a6e3a1", number: "#fab387",
      shadow: "0 4px 20px rgb(0 0 0 / 0.25)",
    },
  },
  { // gruvbox's retro browns and oranges, tightly set
    name: "Gruvbox",
    font: `"IBM Plex Sans", ${SANS}`, displayFont: `"IBM Plex Serif", ui-serif, Georgia, serif`,
    displayWeight: 500, codeFont: `"IBM Plex Mono", ${MONO}`, radius: 4, radiusLarge: 10,
    density: 0.95,
    light: {
      page: "#fbf1c7", side: "#f2e5bc", card: "#f9f5d7", bubble: "#ebdbb2", line: "#d5c4a1",
      text: "#3c3836", muted: "#7c6f64", accent: "#af3a03", onAccent: "#fbf1c7", error: "#9d0006",
      keyword: "#9d0006", string: "#79740e", number: "#8f3f71",
      shadow: "0 2px 0 #d5c4a1",
    },
    dark: {
      page: "#282828", side: "#1d2021", card: "#32302f", bubble: "#3c3836", line: "#504945",
      text: "#ebdbb2", muted: "#a89984", accent: "#fe8019", onAccent: "#282828", error: "#fb4934",
      keyword: "#fb4934", string: "#b8bb26", number: "#d3869b",
      shadow: "0 2px 0 #1d2021",
    },
  },
  { // dark alone: Dracula's purples and pinks
    name: "Dracula",
    displayWeight: 600, radius: 8, radiusLarge: 18,
    dark: {
      page: "#282a36", side: "#21222c", card: "#343746", bubble: "#44475a", line: "#3c3f51",
      text: "#f8f8f2", muted: "#9ea3c4", accent: "#bd93f9", onAccent: "#282a36", error: "#ff5555",
      keyword: "#ff79c6", string: "#f1fa8c", number: "#bd93f9",
      shadow: "0 4px 24px rgb(0 0 0 / 0.3)",
    },
  },
  { // Rosé Pine's Dawn and its night: muted, with an italic serif
    name: "Rosé Pine",
    displayFont: `"Cormorant Garamond", "EB Garamond", ui-serif, Georgia, serif`, displayWeight: 500,
    radius: 10, radiusLarge: 20,
    light: {
      page: "#faf4ed", side: "#f2e9de", card: "#fffaf3", bubble: "#f2e9e1", line: "#dfdad9",
      text: "#575279", muted: "#797593", accent: "#d7827e", onAccent: "#fffaf3", error: "#b4637a",
      keyword: "#907aa9", string: "#56949f", number: "#ea9d34",
      shadow: "0 4px 20px rgb(87 82 121 / 0.08)",
    },
    dark: {
      page: "#191724", side: "#1f1d2e", card: "#1f1d2e", bubble: "#2a273f", line: "#403d52",
      text: "#e0def4", muted: "#908caa", accent: "#ebbcba", onAccent: "#191724", error: "#eb6f92",
      keyword: "#c4a7e7", string: "#9ccfd8", number: "#f6c177",
      shadow: "0 4px 20px rgb(0 0 0 / 0.3)",
    },
  },
].map(complete)];

// a theme as a file has it, checked, and what it leaves out Leat's; or why it cannot be one
export function complete(theme) {
  if (typeof theme !== "object" || theme === null || Array.isArray(theme)) {
    throw new Error("A theme is an object, of its name, fonts, sizes and colors");
  }
  const { name } = theme;
  if (typeof name !== "string" || !name.trim() || name.length > 40) {
    throw new Error("A theme needs a name, of at most 40 characters");
  }
  const done = { name: name.trim() };
  for (const [key, kind] of Object.entries(SHAPE)) {
    const value = theme[key] ?? LEAT[key];
    if (kind === "font" ? !valid("font-family", value)
      : typeof value !== "number" || !(value >= kind[0] && value <= kind[1])) {
      throw new Error(kind === "font" ? `${key} is no font: ${value}`
        : `${key} is a number from ${kind[0]} to ${kind[1]}`);
    }
    done[key] = value;
  }
  const schemes = SCHEMES.filter((s) => theme[s] !== undefined);
  for (const scheme of schemes.length ? schemes : SCHEMES) {
    const colors = theme[scheme] ?? {};
    if (typeof colors !== "object" || colors === null) throw new Error(`${scheme} is an object of colors`);
    done[scheme] = {};
    for (const key of COLORS) {
      const value = colors[key] ?? LEAT[scheme][key];
      if (!valid(key === "shadow" ? "box-shadow" : "color", value)) {
        throw new Error(`${scheme}'s ${key} is no ${key === "shadow" ? "shadow" : "color"}: ${value}`);
      }
      done[scheme][key] = value;
    }
  }
  return done;
}

// whether a value is one of a property's, and nothing more: as a browser parses it, where there is
// one to ask, and never ending its declaration
function valid(property, value) {
  if (typeof value !== "string" || value.length > 200 || /[;{}<>\\]|url\(/i.test(value)) return false;
  return globalThis.CSS?.supports(property, value) ?? true;
}

// the scheme a theme is shown in: its one, or as the mode says, "system" the device's
export function scheme(theme, mode) {
  if (!theme.light || !theme.dark) return theme.light ? "light" : "dark";
  if (mode !== "system") return mode;
  return globalThis.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

// a theme's custom properties in a scheme, for an element to take them alone, as a sample does
export function properties(theme, scheme) {
  return { ...declared(theme, Object.keys(SHAPE)), ...declared(theme[scheme], COLORS),
    "color-scheme": scheme };
}

// the stylesheet that makes the app a theme, in a mode: "system", "light" or "dark"
export function stylesheet(theme, mode) {
  const rule = (scheme) => `:root { ${block(properties(theme, scheme))} }`;
  if (mode !== "system" || !theme.light || !theme.dark) return rule(scheme(theme, mode));
  return `${rule("light")}\n@media (prefers-color-scheme: dark) { ${rule("dark")} }`;
}

// a theme as a file has it, to make another of
export function file(theme) {
  return JSON.stringify(theme, null, 2) + "\n";
}

function declared(values, keys) {
  return Object.fromEntries(keys.map((key) => [PROPERTIES[key] ?? `--${key}`,
    PIXELS.has(key) ? `${values[key]}px` : String(values[key])]));
}

function block(declarations) {
  return Object.entries(declarations).map(([property, value]) => `${property}: ${value};`).join(" ");
}
