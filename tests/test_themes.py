"""The app's themes as themes.mjs checks and completes them, run by Node: every case in one run."""

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


@pytest.fixture(scope="module")
def run():
    script = f"""
    import {{ readFileSync }} from "fs";
    import {{ THEMES, complete, stylesheet }} from {json.dumps((APP / "themes.mjs").as_uri())};
    const refused = JSON.parse(readFileSync(0, "utf8")).map((theme) => {{
      try {{ complete(theme); return null; }} catch (error) {{ return error.message; }}
    }});
    const mint = complete({{ name: " Mint ", radius: 4, light: {{ accent: "#0f9d76" }} }});
    console.log(JSON.stringify({{ themes: THEMES, again: THEMES.map(complete), refused, mint,
      system: stylesheet(THEMES[0], "system"), dark: stylesheet(THEMES[0], "dark"),
      alone: stylesheet(mint, "dark") }}));
    """
    refused = json.dumps([theme for theme, _ in REFUSED])
    run = subprocess.run([NODE, "--input-type=module", "-e", script], input=refused,
                         capture_output=True, text=True, check=True)  # fmt: skip
    return json.loads(run.stdout)


def test_themes(run):
    themes = run["themes"]
    names = [t["name"] for t in themes]
    assert names[0] == "Leat" and len(set(names)) == len(names) >= 10
    assert run["again"] == themes  # each complete already, as a file saved of it would be
    assert all("light" in t or "dark" in t for t in themes)
    assert [t["name"] for t in themes if "light" not in t] == ["Void", "Dracula"]  # dark alone


@pytest.mark.parametrize(("index", "error"), enumerate(error for _, error in REFUSED))
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
    assert f"--accent: {leat['dark']['accent']};" in dark and "--round-large: 24px;" in dark
    assert "--on-accent: #fff;" in dark and "--font-display: ui-serif, Georgia, serif;" in dark


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
