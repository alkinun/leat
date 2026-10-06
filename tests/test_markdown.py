"""The chat app's Markdown as leat/markdown.mjs parses it, run by Node: every case in one run."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node.js")
MODULE = (Path(__file__).parents[1] / "leat" / "markdown.mjs").as_uri()

# text, and its tree: JsonML, a string or [tag, attributes?, ...children]
CASES = [
    ("", []),
    ("hello", [["p", "hello"]]),
    ("  one\n  two  ", [["p", "one\ntwo"]]),
    ("one\n\n\ntwo", [["p", "one"], ["p", "two"]]),
    # code blocks, unclosed while streaming, their info string a language
    ("```\nx = 1\n\ny = 2\n```", [["pre", ["code", "x = 1\n\ny = 2"]]]),
    ("```py\nx\n```\nafter", [["pre", ["code", {"class": "language-py"}, "x"]], ["p", "after"]]),
    ("text\n```\nx", [["p", "text"], ["pre", ["code", "x"]]]),
    ("~~~~\n```\n~~~~", [["pre", ["code", "```"]]]),
    ("  ```\n  x\n    y\n  ```", [["pre", ["code", "x\n  y"]]]),
    # code spans, between runs of as many backticks
    ("a `b` c", [["p", "a ", ["code", "b"], " c"]]),
    ("`` a`b ``", [["p", ["code", "a`b"]]]),
    ("`a\nb`", [["p", ["code", "a b"]]]),
    ("`open", [["p", "`open"]]),
    ("`**a**`", [["p", ["code", "**a**"]]]),
    ("**bold** and **`code`**", [["p", ["strong", "bold"], " and ", ["strong", ["code", "code"]]]]),
]  # fmt: skip


@pytest.fixture(scope="module")
def parsed() -> dict[str, list]:
    script = f"""
        import {{ parse }} from "{MODULE}";
        let input = "";
        for await (const chunk of process.stdin) input += chunk;
        console.log(JSON.stringify(JSON.parse(input).map(parse)));
    """
    texts = [text for text, _ in CASES]
    run = subprocess.run([NODE, "--input-type=module", "-e", script], input=json.dumps(texts),
                         capture_output=True, text=True, check=True)  # fmt: skip
    return dict(zip(texts, json.loads(run.stdout), strict=True))


@pytest.mark.parametrize(("text", "tree"), CASES, ids=[repr(text) for text, _ in CASES])
def test_parse(parsed, text, tree):
    assert parsed[text] == tree
