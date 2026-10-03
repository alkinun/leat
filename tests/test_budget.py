from pathlib import Path

# leat stays small on purpose. Raise this deliberately, never to make a change fit.
LINE_BUDGET = 4000


def test_line_budget():
    package = Path(__file__).parents[1] / "leat"
    lines = sum(
        1
        for path in package.rglob("*.py")
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert lines <= LINE_BUDGET, f"leat/ has {lines} code lines, the budget is {LINE_BUDGET}"
