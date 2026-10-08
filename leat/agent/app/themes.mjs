// The app's themes: each the few things that make it look as it does, its fonts, its corners, how
// much room it leaves, and its colors, light and dark. A theme of one scheme alone is always that;
// whatever a theme leaves out is Leat's own, the first. Leat comes with a few looks of its own;
// colors come of the themes people already have, VS Code's and shadcn/ui's, which Settings opens,
// as it opens Leat's own JSON, as these would be written.

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

// Leat's own: black on white, and white on black, its corners sharp, its lines rather than shadows
const LEAT = {
  name: "Leat",
  font: `"Geist", "Inter", ${SANS}`, displayFont: `"Geist", "Inter", ${SANS}`, displayWeight: 600,
  codeFont: `"Geist Mono", ${MONO}`, size: 16, radius: 6, radiusLarge: 12, density: 1,
  light: {
    page: "#fff", side: "#fafafa", card: "#fff", bubble: "#f2f2f2", line: "#e6e6e6",
    text: "#0a0a0a", muted: "#6e6e6e", accent: "#0a0a0a", onAccent: "#fff", error: "#d92d20",
    keyword: "#7c3aed", string: "#15803d", number: "#b45309", shadow: "none",
  },
  dark: {
    page: "#0a0a0a", side: "#000", card: "#111", bubble: "#1a1a1a", line: "#262626",
    text: "#ededed", muted: "#8f8f8f", accent: "#ededed", onAccent: "#0a0a0a", error: "#ff6369",
    keyword: "#c4a1ff", string: "#86efac", number: "#fdba74", shadow: "none",
  },
};

export const THEMES = [LEAT, ...[
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
  if (typeof value !== "string" || value.length > 400 || /[;{}<>\\]|url\(|var\(/i.test(value)) return false;
  return globalThis.CSS?.supports(property, value) ?? true;
}

// a theme of a file, `named` so: Leat's own, a VS Code color theme, or one of shadcn/ui's, as
// tweakcn makes them, CSS or a registry's JSON; or why it is none. What a theme of VS Code's or
// shadcn's has that cannot be one of Leat's is left out, as what it does not have
export function read(text, named) {
  const name = titled(named.replace(/\.[^.]*$/, ""));
  if (!/^\s*[[{]/.test(text)) return complete(shadcn(stylesheetVariables(text), name));
  const json = parse(text);
  if (json?.cssVars) {
    const { theme, light, dark } = json.cssVars;
    const schemes = { light: { ...theme, ...light }, ...(dark && { dark: { ...theme, ...light, ...dark } }) };
    return complete(shadcn(schemes, json.name ? titled(json.name) : name));
  }
  if (json?.colors || json?.tokenColors) return complete(code(json, name));
  if (!Object.keys(json ?? {}).some((key) => Object.hasOwn(SHAPE, key) || SCHEMES.includes(key))) {
    throw new Error("it is no theme of Leat's, VS Code's or shadcn/ui's");
  }
  return complete(json);
}

// a name as a file's or a registry's: "neo-brutalism", "Neo Brutalism"
function titled(name) {
  return name.replace(/[-_]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()).trim().slice(0, 40);
}

// JSON, with the comments and trailing commas VS Code's themes are often written with
function parse(text) {
  const string = /"(?:\\.|[^"\\])*"/.source;
  return JSON.parse(text
    .replace(new RegExp(`(${string})|//[^\\n]*|/\\*[\\s\\S]*?\\*/`, "g"), (_, kept) => kept ?? "")
    .replace(new RegExp(`(${string})|,(?=\\s*[}\\]])`, "g"), (_, kept) => kept ?? ""));
}

// a stylesheet's custom properties: :root's light, and .dark's dark, over :root's
function stylesheetVariables(text) {
  const own = { light: {} };
  for (const [, selector, body] of text.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/([^{}]*)\{([^{}]*)\}/g)) {
    const scheme = /\.dark\b/.test(selector) ? "dark" : /:root/.test(selector) ? "light" : null;
    if (!scheme) continue;
    own[scheme] ??= {};
    for (const [, key, value] of body.matchAll(/--([\w-]+)\s*:\s*([^;]+)/g)) own[scheme][key] = value.trim();
  }
  if (!Object.keys(own.light).length) throw new Error("it has no :root of shadcn/ui's variables");
  return { light: own.light, ...(own.dark && { dark: { ...own.light, ...own.dark } }) };
}

// a value of an imported theme's, if it is one of the property's
function usable(property, value) {
  return valid(property, value) ? value : undefined;
}

// a theme of shadcn/ui's variables, each scheme's: its fonts, its radius, the larger as Leat's
// are to its own, its spacing, and its colors, with Leat's code's
function shadcn(schemes, name) {
  const first = schemes.light ?? schemes.dark;
  const pixels = (length) => parseFloat(length) * (/rem\s*$/.test(length) ? 16 : 1);
  const radius = Math.min(40, Math.round(pixels(first.radius ?? "0.625rem"))) || 0;
  const colors = (v) => {
    const color = (key, property = "color") => {
      const value = v[key]?.replace(/var\(--([\w-]+)\)/g, (match, other) => v[other] ?? match);
      // Tailwind 3's hues, saturations and lightnesses, bare
      return usable(property, /^[\d.]+\s+[\d.]+%\s+[\d.]+%(\s*\/\s*[\d.]+%?)?$/.test(value ?? "") ? `hsl(${value})` : value);
    };
    return { page: color("background"), side: color("sidebar") ?? color("sidebar-background"),
      card: color("card"), bubble: color("muted"), line: color("border"), text: color("foreground"),
      muted: color("muted-foreground"), accent: color("primary"), onAccent: color("primary-foreground"),
      error: color("destructive"), shadow: color("shadow-lg", "box-shadow") ?? color("shadow", "box-shadow") };
  };
  const spacing = first.spacing && pixels(first.spacing) / 4;
  return {
    name, font: usable("font-family", first["font-sans"]), displayFont: usable("font-family", first["font-sans"]),
    displayWeight: 600, codeFont: usable("font-family", first["font-mono"]), radius,
    radiusLarge: Math.min(40, Math.round(radius * 2.4)),
    density: spacing ? Math.min(1.5, Math.max(0.75, spacing)) : undefined,
    ...Object.fromEntries(Object.entries(schemes).map(([scheme, v]) => [scheme, colors(v)])),
  };
}

// a theme of a VS Code color theme's, which is of one scheme and colors alone: its editor's, its
// sidebar's, inputs', hovers', borders', buttons' and errors', what is not said mixed of its text
// and its page, and its keywords', strings' and numbers' of its token colors
function code(theme, name) {
  const c = theme.colors ?? {};
  const color = (...keys) => keys.map((key) => usable("color", c[key])).find(Boolean);
  const bright = (hex) => /^#[0-9a-f]{6}/i.test(hex ?? "")
    && [1, 3, 5].reduce((sum, at, i) => sum + parseInt(hex.slice(at, at + 2), 16) * [0.3, 0.59, 0.11][i], 0) > 128;
  const scheme = theme.type ? (/light/i.test(theme.type) ? "light" : "dark")
    : bright(c["editor.background"]) ? "light" : "dark";
  const page = color("editor.background") ?? (scheme === "light" ? "#fff" : "#1e1e1e");
  const text = color("editor.foreground", "foreground") ?? (scheme === "light" ? "#333" : "#ccc");
  const mix = (share) => `color-mix(in srgb, ${text} ${share}%, ${page})`;
  const rules = Array.isArray(theme.tokenColors) ? theme.tokenColors : [];
  const scopes = (rule) => [rule.scope ?? []].flat().flatMap((s) => String(s).split(",")).map((s) => s.trim());
  const token = (scope) => {
    const colored = (match) => rules.find((r) => usable("color", r.settings?.foreground) && scopes(r).some(match));
    return (colored((s) => s === scope) ?? colored((s) => s.startsWith(`${scope}.`)))?.settings.foreground;
  };
  return { name: typeof theme.name === "string" ? titled(theme.name) : name, [scheme]: {
    page, side: color("sideBar.background") ?? mix(3), card: color("input.background") ?? mix(5),
    bubble: color("list.hoverBackground") ?? mix(8),
    line: color("panel.border", "sideBar.border", "editorGroup.border") ?? mix(14), text,
    muted: color("descriptionForeground") ?? mix(60),
    accent: color("button.background", "focusBorder", "textLink.foreground"),
    onAccent: color("button.foreground"), error: color("errorForeground", "editorError.foreground"),
    keyword: token("keyword"), string: token("string"), number: token("constant.numeric"),
    shadow: color("widget.shadow") && `0 4px 20px ${color("widget.shadow")}`,
  } };
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
