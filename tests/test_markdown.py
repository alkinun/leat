"""The app's Markdown as markdown.mjs parses it, run by Node: every case in one run."""

import functools
import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node.js")
MODULE = (Path(__file__).parents[1] / "leat" / "agent" / "app" / "markdown.mjs").as_uri()


def token(kind: str, text: str) -> list:
    """A highlighted token of a code block."""
    return ["span", {"class": kind}, text]


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
    # quotes, of blocks of their own
    ("> one\n>two\n>\n> # three", [["blockquote", ["p", "one\ntwo"], ["h1", "three"]]]),
    ("text\n> quote\n\n> again\nafter",
     [["p", "text"], ["blockquote", ["p", "quote"]], ["blockquote", ["p", "again"]],
      ["p", "after"]]),
    ("> > nested\n> ```\n> code", [["blockquote", ["blockquote", ["p", "nested"]],
                                     ["pre", ["code", "code"]]]]),
    ("a > b", [["p", "a > b"]]),
    # lists, tight but for a blank line between items or blocks of one
    ("- a\n- b **c**\n+ d",
     [["ul", ["li", "a"], ["li", "b ", ["strong", "c"]]], ["ul", ["li", "d"]]]),
    ("text:\n1. a\n2. b\n\n3) c", [["p", "text:"], ["ol", ["li", "a"], ["li", "b"]],
                                    ["ol", {"start": 3}, ["li", "c"]]]),
    ("- a\n\n- b", [["ul", ["li", ["p", "a"]], ["li", ["p", "b"]]]]),
    ("1. a\n\n   more\n2. b", [["ol", ["li", ["p", "a"], ["p", "more"]], ["li", ["p", "b"]]]]),
    ("- a\n  b\nlazy\n\nafter", [["ul", ["li", "a\nb\nlazy"]], ["p", "after"]]),
    ("- a\n  - b\n    - c\n- d", [["ul", ["li", "a", ["ul", ["li", "b", ["ul", ["li", "c"]]]]],
                                    ["li", "d"]]]),
    ("1. a\n  - b\n  - c\n2. d", [["ol", ["li", "a", ["ul", ["li", "b"], ["li", "c"]]],
                                    ["li", "d"]]]),
    ("- a\n  ```\n  x\n  ```\n- b", [["ul", ["li", "a", ["pre", ["code", "x"]]], ["li", "b"]]]),
    ("- > quote\n-\n- # heading", [["ul", ["li", ["blockquote", ["p", "quote"]]], ["li"],
                                     ["li", ["h1", "heading"]]]]),
    ("* * *\n- - -\n-5 1.5 +1 **b**", [["hr"], ["hr"], ["p", "-5 1.5 +1 ", ["strong", "b"]]]),
    # tables, their rows to a blank line or a block, as many cells as the header has
    ("| a | *b* |\n|---|:-:|\n| 1 |\n| 2 | 3 | 4 |\n\nafter",
     [["table", ["thead", ["tr", ["th", "a"], ["th", {"align": "center"}, ["em", "b"]]]],
       ["tbody", ["tr", ["td", "1"], ["td", {"align": "center"}]],
        ["tr", ["td", "2"], ["td", {"align": "center"}, "3"]]]], ["p", "after"]]),
    ("text\na | b\n:- | -:\n`\\|` | c\n# end",
     [["p", "text"], ["table", ["thead", ["tr", ["th", {"align": "left"}, "a"],
                                           ["th", {"align": "right"}, "b"]]],
                      ["tbody", ["tr", ["td", {"align": "left"}, ["code", "|"]],
                                 ["td", {"align": "right"}, "c"]]]], ["h1", "end"]]),
    ("|a|\n|-|", [["table", ["thead", ["tr", ["th", "a"]]]]]),
    ("| a | b |\n|---|", [["p", "| a | b |\n|---|"]]),
    ("| a | b |", [["p", "| a | b |"]]),
    # math, a lone $ before a non-space and after one, and before no digit
    ("Euler: $e^{i\\pi} + 1 = 0$.", [["p", "Euler: ", ["math", "e^{i\\pi} + 1 = 0"], "."]]),
    ("\\(a_1\\) and $$b$$", [["p", ["math", "a_1"], " and ", ["math", {"display": "block"}, "b"]]]),
    ("$5 to $10, or US$5, then x$", [["p", "$5 to $10, or US$5, then x$"]]),
    ("$ x $ $x$5 \\$x$", [["p", "$ x $ $x$5 $x$"]]),
    ("$a\\$b$", [["p", ["math", "a\\$b"]]]),
    ("*a $b_1*c$ d* [$]$](https://x.io)",
     [["p", ["em", "a ", ["math", "b_1*c"], " d"], " ",
       ["a", {"href": "https://x.io"}, ["math", "]"]]]]),
    ("text\n$$\n\\int_0^1 x\\,dx\n$$\nafter",
     [["p", "text"], ["math", {"display": "block"}, "\\int_0^1 x\\,dx"], ["p", "after"]]),
    ("\\[ a^2 \\] and $$b$$ c", [["math", {"display": "block"}, "a^2"],
                               ["p", "and ", ["math", {"display": "block"}, "b"], " c"]]),
    ("$$\\frac{a}{", [["math", {"display": "block"}, "\\frac{a}{"]]),
    ("`$x$`", [["p", ["code", "$x$"]]]),
    # code blocks, unclosed while streaming, their info string a language
    ("```\nx = 1\n\ny = 2\n```", [["pre", ["code", "x = 1\n\ny = 2"]]]),
    ("```md\nx\n```\nafter", [["pre", ["code", {"class": "language-md"}, "x"]], ["p", "after"]]),
    ("```constructor\nx\n```", [["pre", ["code", {"class": "language-constructor"}, "x"]]]),
    ("```__proto__\nx\n```", [["pre", ["code", {"class": "language-__proto__"}, "x"]]]),
    ("text\n```\nx", [["p", "text"], ["pre", ["code", "x"]]]),
    ("~~~~\n```\n~~~~", [["pre", ["code", "```"]]]),
    ("  ```\n  x\n    y\n  ```", [["pre", ["code", "x\n  y"]]]),
    # highlighting, of the languages models write most
    ("```py\nx = 1  # a\nreturn '#'\n```",
     [["pre", ["code", {"class": "language-py"}, "x = ", token("number", "1"), "  ",
               token("comment", "# a"), "\n", token("keyword", "return"), " ",
               token("string", "'#'")]]]),
    ("```Rust\nfn f<'a>() -> char { 'x' }",
     [["pre", ["code", {"class": "language-Rust"}, token("keyword", "fn"), " f<'a>() -> ",
               token("keyword", "char"), " { ", token("string", "'x'"), " }"]]]),
    ("```sql\nSELECT a -- b",
     [["pre", ["code", {"class": "language-sql"}, token("keyword", "SELECT"), " a ",
               token("comment", "-- b")]]]),
    ("```js\n`${a}` /* b",
     [["pre", ["code", {"class": "language-js"}, token("string", "`${a}`"), " ",
               token("comment", "/* b")]]]),
    ("```sh\necho $# # c",
     [["pre", ["code", {"class": "language-sh"}, "echo $# ", token("comment", "# c")]]]),
    ("```klingon\nif x\n```", [["pre", ["code", {"class": "language-klingon"}, "if x"]]]),
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
    # a run that cannot open is text whole, as CommonMark's: its second _ opens nothing either
    ("foo__bar__ and snake__case or other___x___", [["p", "foo__bar__ and snake__case or "
                                                         "other___x___"]]),
    ("__a_ and obj.__dict__", [["p", "_", ["em", "a"], " and obj.", ["strong", "dict"]]]),
    ("~5 to ~10, ~~~", [["p", "~5 to ~10, ~~~"]]),
    ("**a\nb**", [["p", ["strong", "a\nb"]]]),
    ("**unclosed", [["p", "**unclosed"]]),
    # HTML, as its text
    ('<b>a</b><img src=x onerror="alert(1)">', [["p", '<b>a</b><img src=x onerror="alert(1)">']]),
    # links, to the web or mail only
    ("[a **b**](https://x.io)", [["p", ["a", {"href": "https://x.io"}, "a ", ["strong", "b"]]]]),
    ('[a](<https://x.io/a b> "title")', [["p", ["a", {"href": "https://x.io/a b"}, "a"]]]),
    ("[a](https://w.org/A_(b)) c", [["p", ["a", {"href": "https://w.org/A_(b)"}, "a"], " c"]]),
    ("[[1]](mailto:a@x.io)", [["p", ["a", {"href": "mailto:a@x.io"}, "[1]"]]]),
    # citations, of a source's number, which the app links; no link's text cites
    ("Paris [1][2], and [12].",
     [["p", "Paris ", ["sup", {"class": "cite"}, "1"], ["sup", {"class": "cite"}, "2"], ", and ",
       ["sup", {"class": "cite"}, "12"], "."]]),
    ("[1234], [a], [1, 2] and [1](javascript:x)",
     [["p", "[1234], [a], [1, 2] and ", ["sup", {"class": "cite"}, "1"], "(javascript:x)"]]),
    # and as gpt-oss cites, at times with a part of the source
    ("Paris【1】, Lyon【2†L4-L9】, 【x】, 【1234】 and 【3",
     [["p", "Paris", ["sup", {"class": "cite"}, "1"], ", Lyon", ["sup", {"class": "cite"}, "2"],
       ", 【x】, 【1234】 and 【3"]]),
    ("[`]`](https://x.io)", [["p", ["a", {"href": "https://x.io"}, ["code", "]"]]]]),
    ("![a cat](https://x.io/cat.png)", [["p", ["a", {"href": "https://x.io/cat.png"}, "a cat"]]]),
    ("[a](javascript:alert(1)) [b](data:text/html,x) [c](/path) [d](HTTPS://X.IO)",
     [["p", "[a](javascript:alert(1)) [b](data:text/html,x) [c](/path) ",
       ["a", {"href": "HTTPS://X.IO"}, "d"]]]),
    ("[a] (b) [c](d [e]", [["p", "[a] (b) [c](d [e]"]]),
    ("<https://x.io/a_b_c> <javascript:x>",
     [["p", ["a", {"href": "https://x.io/a_b_c"}, "https://x.io/a_b_c"], " <javascript:x>"]]),
    ("see https://x.io/a_b_c.", [["p", "see ", ["a", {"href": "https://x.io/a_b_c"},
                                                "https://x.io/a_b_c"], "."]]),
    ("(https://w.org/A_(b)), **https://x.io**",
     [["p", "(", ["a", {"href": "https://w.org/A_(b)"}, "https://w.org/A_(b)"], "), ",
       ["strong", ["a", {"href": "https://x.io"}, "https://x.io"]]]]),
    ("xhttps://x.io https:// https://. http", [["p", "xhttps://x.io https:// https://. http"]]),
    # escapes, of ASCII punctuation only
    ("\\*a\\* \\` \\\\ \\a", [["p", "*a* ` \\ \\a"]]),
    # runs that open spans none closes, as C's pointers, in time linear in them, not exponential
    ("int *a, " * 59 + "int *a", [["p", "int *a, " * 59 + "int *a"]]),
    ("a _b **c " * 39 + "a _b **c", [["p", "a _b **c " * 39 + "a _b **c"]]),
    # blocks deeper than a page shows, text
    ("> " * 40 + "deep", [functools.reduce(lambda tree, _: ["blockquote", tree], range(32),
                                           ["p", "> " * 8 + "deep"])]),
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
                         capture_output=True, text=True, check=True, timeout=60)  # fmt: skip
    return dict(zip(texts, json.loads(run.stdout), strict=True))


@pytest.mark.parametrize(("text", "tree"), CASES, ids=[repr(text) for text, _ in CASES])
def test_parse(parsed, text, tree):
    assert parsed[text] == tree


# texts a reply may hold, or a page it read, that took seconds or overflowed the stack: spans that
# never close, runs regular expressions backtrack over, and blocks or spans thousands deep
HARD = {
    "pointers": "int *a, " * 5000,
    "unclosed code": "*a " * 3000 + "``" + " `x`" * 3000,
    "alternating": "**a ~~b " * 500 + "c" + "~~ d** " * 500,
    "heading": "# a" + " " * 40000 + "b",
    "url": "https://x" + ")" * 40000,
    "code": "` " + "a" * 40000 + "`",
    "delimiter": "a|b\n-" + " " * 40000 + "x",
    "quotes": ">" * 4000 + " deep",
    "lists": "- + " * 1300,
    "links": "[" * 3000 + "x" + "](https://x.io)" * 3000,
    "within words": "x_y__z " * 3000,
}


def test_hard_texts_parse_quickly():
    script = f"""
        import {{ parse }} from "{MODULE}";
        let input = "";
        for await (const chunk of process.stdin) input += chunk;
        console.log(JSON.stringify(JSON.parse(input).map((text) => {{
            const start = performance.now();
            parse(text);
            return performance.now() - start;
        }})));
    """
    run = subprocess.run([NODE, "--input-type=module", "-e", script],
                         input=json.dumps(list(HARD.values())), capture_output=True, text=True,
                         check=True, timeout=60)  # fmt: skip
    times = dict(zip(HARD, json.loads(run.stdout), strict=True))
    assert all(ms < 300 for ms in times.values()), times  # each takes 30 ms at most here
