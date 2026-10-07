"""Measures how the agent does what people ask of it: whether it calls the tools it should, says
what it should, and remembers and forgets what it should, each case run several times, in a state
of its own. Prints a Markdown table of each case's passes and mean time.

    uv run python scripts/evaluate.py [--engine http://127.0.0.1:8080] [-n 3] [--think] [-k name]

The agent runs in this process, with leat agent's tools, against leat serve at --engine and the
SearXNG at --search; the states are temporary, and the user's own untouched. A prompt's or a tool's
change is measured here before it is kept.
"""

import argparse
import re
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leat.agent.agent import Agent  # noqa: E402
from leat.agent.client import Client  # noqa: E402
from leat.agent.store import Store  # noqa: E402
from leat.agent.tools import weather, web  # noqa: E402

TIMEOUT = 300  # seconds a turn may take


@dataclass(frozen=True)
class Outcome:
    """What came of a case's message: the tools it called, the answer, the memories after."""

    tools: list[str]
    answer: str
    memories: list[str]
    seconds: float


Check = Callable[[Outcome], str | None]  # why an outcome fails, or None if it passes


def called(name: str, times: int = 1) -> Check:
    n = lambda o: o.tools.count(name)  # noqa: E731
    return lambda o: None if n(o) >= times else f"called {name} {n(o)} times, not {times}"


def uncalled(*names: str) -> Check:
    return lambda o: next((f"called {name}" for name in names if name in o.tools), None)


def says(pattern: str) -> Check:
    return lambda o: None if re.search(pattern, o.answer, re.I) else f"did not say /{pattern}/"


def remembers(pattern: str) -> Check:
    found = lambda o: any(re.search(pattern, m, re.I) for m in o.memories)  # noqa: E731
    return lambda o: None if found(o) else f"does not remember /{pattern}/"


def forgot(pattern: str) -> Check:
    return lambda o: f"still remembers /{pattern}/" if remembers(pattern)(o) is None else None


@dataclass(frozen=True)
class Case:
    name: str
    message: str  # in a new conversation
    checks: list[Check]
    memories: list[str] = field(default_factory=list)  # remembered before it
    before: list[str] = field(default_factory=list)  # messages of earlier conversations, each one's


NONE = ("search", "fetch", "weather", "remember", "forget", "recall")  # every tool
CASES = [
    Case("chat", "Write a haiku about autumn.", [uncalled(*NONE)]),
    Case("arithmetic", "What is 17 * 23?", [says(r"391"), uncalled(*NONE)]),
    Case("known fact", "What is the capital of Australia?",
         [says("Canberra"), uncalled("remember")]),
    Case("news", "What's in the news today about space exploration?", [called("search")]),
    Case("reads pages", "When does the British Museum open tomorrow? Check its website.",
         [called("fetch")]),
    Case("weather", "Will I need an umbrella in London tomorrow?",
         [called("weather"), uncalled("fetch")]),
    Case("introduction", "Hi! I'm Sam, I work as a nurse, and my kids are called Mia and Leo.",
         [called("remember", 2), remembers("Sam"), remembers("nurse"), remembers("Mia|Leo")]),
    Case("preference", "I'm vegetarian, keep that in mind for recipes.",
         [called("remember"), remembers("vegetarian")]),
    Case("knows", "What's my dog called?", [says("Rex"), uncalled("search", "recall")],
         memories=["The user's dog is called Rex."]),
    Case("forget", "Please forget about my dog.",
         [called("forget"), forgot("Rex"), remembers("Izmir")],
         memories=["The user's dog is called Rex.", "The user lives in Izmir."]),
    Case("recall", "What did I ask you about tulips the other day?",
         [called("recall"), says("plant")],
         before=["When should I plant tulip bulbs? One sentence."]),
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", default="http://127.0.0.1:8080", help="leat serve's address")
    parser.add_argument("--search", default="http://127.0.0.1:8888", help="a SearXNG's address")
    parser.add_argument("-n", "--runs", type=int, default=3, help="runs of each case")
    parser.add_argument("--think", action="store_true", help="have the model think first")
    parser.add_argument("-k", help="the cases whose names hold this alone")
    args = parser.parse_args()
    cases = [case for case in CASES if not args.k or args.k in case.name]
    rows, passed = [], 0
    for case in cases:
        failures, seconds = [], []
        for _ in range(args.runs):
            outcome = _run(case, args)
            seconds.append(outcome.seconds)
            why = [w for check in case.checks if (w := check(outcome))]
            failures += why[:1]
            print(f"{case.name}: {why[0] if why else 'passed'}", file=sys.stderr, flush=True)
        passed += args.runs - len(failures)
        failed = "; ".join(sorted(set(failures)))
        rows.append(f"| {case.name} | {args.runs - len(failures)}/{args.runs} | "
                    f"{statistics.mean(seconds):.1f} | {failed} |")  # fmt: skip
    mode = "thinking" if args.think else "not thinking"
    print(f"\n{passed} of {len(cases) * args.runs} passed, {mode}\n")
    print("| case | passed | mean s | failures |\n|---|---:|---:|---|")
    print("\n".join(rows))


def _run(case: Case, args: argparse.Namespace) -> Outcome:
    # the case, in an agent of a state of its own
    with tempfile.TemporaryDirectory() as data:
        tools = [*web.tools(args.search), *weather.tools()]
        agent = Agent(Store(Path(data) / "leat.db"), Client(args.engine), tools)
        for memory in case.memories:
            agent.remember(memory)
        for message in case.before:
            _wait(agent, agent.send(None, message, args.think))
        start = time.monotonic()
        messages = _wait(agent, agent.send(None, case.message, args.think))
        seconds = time.monotonic() - start
        tools_called = [m["name"] for m in messages if m["role"] == "tool"]
        answer = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else ""
        memories = [m["text"] for m in agent.memories()]
        return Outcome(tools_called, answer or "", memories, seconds)


def _wait(agent: Agent, id: str) -> list[dict]:
    # a conversation's messages once its turn has ended, or none if it failed and took it back
    end = time.monotonic() + TIMEOUT
    while (c := agent.conversation(id)) is not None and c["running"]:
        if time.monotonic() > end:
            agent.stop(id)
        time.sleep(0.1)
    return c["messages"] if c else []


if __name__ == "__main__":
    main()
