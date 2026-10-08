"""The app's themes as themes.mjs checks, completes and reads them of VS Code's and shadcn/ui's, run
by Node: every case in one run."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node.js")
APP = Path(__file__).parents[1] / "leat" / "agent" / "app"

# themes as files have them, and the error each makes, or None for none
REFUSED = [
    ({"name": "Mint"}, None),
    ([], "A theme is an object"),
    ({}, "A theme needs a name"),
    ({"name": "x" * 41}, "A theme needs a name"),
    ({"name": "Big", "size": 30}, "size is a number from 12 to 22"),
    ({"name": "Round", "radius": "8px"}, "radius is a number from 0 to 40"),
    ({"name": "Font", "font": "Inter; color: red"}, "font is no font"),
    (
        {"name": "Out", "light": {"accent": "red; } body { display: none"}},
        "light's accent is no color",
    ),
    (
        {"name": "Far", "dark": {"shadow": "0 0 4px url(https://x.io)"}},
        "dark's shadow is no shadow",
    ),
    ({"name": "Odd", "dark": "black"}, "dark is an object of colors"),
]

# files of others' themes, and what they read as
CODE = """{
  // Solarized's, which says not whether it is light or dark
  "name": "Sunny (light)",
  "colors": {
    "editor.background": "#FDF6E3", "editor.foreground": "#657B83", "button.background": "#2AA198",
    "textLink.foreground": "https://x.io/not-a-comment", "input.background": "var(--x)",
  },
  /* its syntax's */
  "tokenColors": [
    {"settings": {"foreground": "#657B83"}},
    {"scope": "keyword.control, storage", "settings": {"foreground": "#859900"}},
    {"scope": ["string.quoted", "string"], "settings": {"foreground": "#2AA198"}},
  ],
}"""
REGISTRY = {
    "name": "neo-brutalism",
    "cssVars": {
        "theme": {"font-sans": "DM Sans, sans-serif", "radius": "0px"},
        "light": {"background": "oklch(1 0 0)", "primary": "oklch(0.65 0.24 27)",
                  "spacing": "0.3rem", "shadow-lg": "4px 4px 0px 0px hsl(0 0% 0% / 1)"},
        "dark": {"background": "oklch(0 0 0)"},
    },
}  # fmt: skip
CSS = """@layer base {
  :root { --background: 0 0% 100%; --foreground: 222.2 84% 4.9%; --radius: 0.5rem; }
  /* dark, as Tailwind 3's were */
  .dark { --background: 222.2 84% 4.9%; --muted: var(--background); }
}"""
READ = [
    (CODE, "sunny.json"),
    (json.dumps(REGISTRY), "theme.json"),
    (CSS, "slate-blue.css"),
    ('{"a": 1}', "package.json"),
    ("body { color: red }", "plain.css"),
]


@pytest.fixture(scope="module")
def run():
    module = json.dumps((APP / "themes.mjs").as_uri())
    script = f"""
    import {{ readFileSync }} from "fs";
    import {{ THEMES, complete, read, stylesheet }} from {module};
    const given = JSON.parse(readFileSync(0, "utf8"));
    const refused = given.refused.map((theme) => {{
      try {{ complete(theme); return null; }} catch (error) {{ return error.message; }}
    }});
    const opened = given.read.map(([text, name]) => {{
      try {{ return read(text, name); }} catch (error) {{ return error.message; }}
    }});
    const mint = complete({{ name: " Mint ", radius: 4, light: {{ accent: "#0f9d76" }} }});
    console.log(JSON.stringify({{ themes: THEMES, again: THEMES.map(complete), refused, mint,
      read: opened, system: stylesheet(THEMES[0], "system"), dark: stylesheet(THEMES[0], "dark"),
      alone: stylesheet(mint, "dark") }}));
    """
    given = json.dumps({"refused": [theme for theme, _ in REFUSED], "read": READ})
    run = subprocess.run([NODE, "--input-type=module", "-e", script], input=given,
                         capture_output=True, text=True, check=True)  # fmt: skip
    return json.loads(run.stdout)


def test_themes(run):
    themes = run["themes"]
    names = [t["name"] for t in themes]
    assert names == ["Leat", "Graphite", "Void", "Hermes", "Newsprint"]
    assert run["again"] == themes  # each complete already, as a file saved of it would be
    assert all("light" in t or "dark" in t for t in themes)
    assert [t["name"] for t in themes if "light" not in t] == ["Void"]  # dark alone


@pytest.mark.parametrize(("index", "error"), list(enumerate(error for _, error in REFUSED)))
def test_refused(run, index, error):
    message = run["refused"][index]
    assert message is None if error is None else message.startswith(error)


def test_partial(run):
    # what a theme leaves out is Leat's, and a theme of one scheme has that alone
    mint, leat = run["mint"], run["themes"][0]
    assert mint["name"] == "Mint" and mint["radius"] == 4 and mint["font"] == leat["font"]
    assert mint["light"] == leat["light"] | {"accent": "#0f9d76"} and "dark" not in mint
    assert "--accent: #0f9d76;" in run["alone"] and "color-scheme: light;" in run["alone"]
    assert "@media" not in run["alone"]


def test_stylesheet(run):
    leat = run["themes"][0]
    system, dark = run["system"], run["dark"]
    assert "@media (prefers-color-scheme: dark)" in system and "@media" not in dark
    assert f"--accent: {leat['dark']['accent']};" in dark and "color-scheme: dark;" in dark
    assert f"--on-accent: {leat['dark']['onAccent']};" in dark
    assert f"--round-large: {leat['radiusLarge']}px;" in dark
    assert f"--font-display: {leat['displayFont']};" in dark


def test_defaults(run):
    # style.css's own, before a theme is applied, are Leat's, light and dark
    css = (APP / "style.css").read_text()
    root = re.search(r"^:root \{(.*?)^\}", css, re.S | re.M)[1]
    dark = re.search(r"prefers-color-scheme: dark\) \{\s*:root \{(.*?)\}", css, re.S)[1]
    declared = lambda block: dict(re.findall(r"(--[\w-]+): ([^;]+);", block))  # noqa: E731
    leat = run["themes"][0]
    for scheme, block in (("light", root), ("dark", dark)):
        for key, value in leat[scheme].items():
            assert declared(block)["--on-accent" if key == "onAccent" else f"--{key}"] == value
    shape = {
        "--font": leat["font"],
        "--font-display": leat["displayFont"],
        "--display-weight": str(leat["displayWeight"]),
        "--font-mono": leat["codeFont"],
        "--size": f"{leat['size']}px",
        "--round": f"{leat['radius']}px",
        "--round-large": f"{leat['radiusLarge']}px",
        "--density": str(leat["density"]),
    }
    assert declared(root).items() >= shape.items()


def test_code(run):
    # VS Code's: of one scheme, its own by its editor's background; what it lacks of its text and
    # page, or Leat's, and what cannot be one of Leat's left out; its syntax of its scopes
    theme, leat = run["read"][0], run["themes"][0]
    assert theme["name"] == "Sunny (Light)" and "dark" not in theme
    light = theme["light"]
    assert (
        light["page"] == "#FDF6E3" and light["text"] == "#657B83" and light["accent"] == "#2AA198"
    )
    assert light["card"] == "color-mix(in srgb, #657B83 5%, #FDF6E3)"
    assert light["keyword"] == "#859900" and light["string"] == "#2AA198"
    assert light["number"] == leat["light"]["number"] and theme["font"] == leat["font"]


def test_shadcn(run):
    # shadcn/ui's, of tweakcn's registry: its fonts, corners, spacing, shadow, and dark over light
    theme = run["read"][1]
    assert theme["name"] == "Neo Brutalism" and theme["font"] == "DM Sans, sans-serif"
    assert theme["radius"] == theme["radiusLarge"] == 0 and theme["density"] == pytest.approx(1.2)
    assert theme["light"]["accent"] == "oklch(0.65 0.24 27)" == theme["dark"]["accent"]
    assert theme["light"]["shadow"] == "4px 4px 0px 0px hsl(0 0% 0% / 1)"
    assert theme["dark"]["page"] == "oklch(0 0 0)"
    # of a stylesheet: Tailwind 3's bare colors, and .dark's over :root's
    theme = run["read"][2]
    assert theme["name"] == "Slate Blue" and theme["radius"] == 8 and theme["radiusLarge"] == 19
    assert theme["light"]["page"] == "hsl(0 0% 100%)"
    assert theme["dark"]["page"] == theme["dark"]["bubble"] == "hsl(222.2 84% 4.9%)"
    assert theme["dark"]["text"] == "hsl(222.2 84% 4.9%)"  # :root's


def test_unread(run):
    assert run["read"][3] == "it is no theme of Leat's, VS Code's or shadcn/ui's"
    assert run["read"][4] == "it has no :root of shadcn/ui's variables"
