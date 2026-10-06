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
    # headings, their closing hashes dropped, and rules, each breaking into a paragraph
    ("# One\n###### Six ##", [["h1", "One"], ["h6", "Six"]]),
    ("text\n## **Bold** #5#\nmore",
     [["p", "text"], ["h2", ["strong", "Bold"], " #5#"], ["p", "more"]]),
    ("#hashtag\n####### seven", [["p", "#hashtag\n####### seven"]]),
    ("#", [["h1"]]),
    ("one\n***\n- - -\n___", [["p", "one"], ["hr"], ["hr"], ["hr"]]),
    ("--", [["p", "--"]]),
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
    # emphasis, opened before a non-space and closed after one, an _ neither within a word
    ("*a* _b_ **c** __d__ ~~e~~", [["p", ["em", "a"], " ", ["em", "b"], " ", ["strong", "c"], " ",
                                   ["strong", "d"], " ", ["del", "e"]]]),
    ("***a***", [["p", ["em", ["strong", "a"]]]]),
    ("*a **b** c*", [["p", ["em", "a ", ["strong", "b"], " c"]]]),
    ("**a *b* c**", [["p", ["strong", "a ", ["em", "b"], " c"]]]),
    ("*a **b***", [["p", ["em", "a ", ["strong", "b"]]]]),
    ("***a** b*", [["p", ["em", ["strong", "a"], " b"]]]),
    ("***a* b**", [["p", ["strong", ["em", "a"], " b"]]]),
    ("**a*", [["p", "*", ["em", "a"]]]),
    ("**bold** and **`code`**", [["p", ["strong", "bold"], " and ", ["strong", ["code", "code"]]]]),
    ("**a `**` b**", [["p", ["strong", "a ", ["code", "**"], " b"]]]),
    ("**Note:**text", [["p", ["strong", "Note:"], "text"]]),
    ("2 * 3 * 4, a ** b, ** c **", [["p", "2 * 3 * 4, a ** b, ** c **"]]),
    ("_snake_case_ and _x_y_", [["p", ["em", "snake_case"], " and ", ["em", "x_y"]]]),
    ("~5 to ~10, ~~~", [["p", "~5 to ~10, ~~~"]]),
    ("**a\nb**", [["p", ["strong", "a\nb"]]]),
    ("**unclosed", [["p", "**unclosed"]]),
    # escapes, of ASCII punctuation only
    ("\\*a\\* \\` \\\\ \\a", [["p", "*a* ` \\ \\a"]]),
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
