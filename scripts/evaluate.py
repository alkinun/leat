"""Measures how the agent does what people ask of it: whether it calls the tools it should, says
what it should, and remembers and forgets what it should, each case run several times, in a state
of its own. Prints a Markdown table of each case's passes, mean time, and the share of the
prompts' tokens the engine's cache held.

    uv run python scripts/evaluate.py [--engine http://127.0.0.1:8080] [-n 3] [--think] [-k name]

The agent runs in this process, with leat agent's tools, against leat serve at --engine, the SearXNG
at --search, and the sandbox's environment at --sandbox; the states are temporary, and the user's
own untouched. A prompt's or a tool's change is measured here before it is kept.
"""

import argparse
import datetime
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

from leat.agent import background  # noqa: E402
from leat.agent.agent import Agent  # noqa: E402
from leat.agent.client import Client  # noqa: E402
from leat.agent.store import Store  # noqa: E402
from leat.agent.tools import files, weather, web  # noqa: E402
from leat.agent.workspace import Workspace  # noqa: E402

TIMEOUT = 300  # seconds a turn may take


@dataclass(frozen=True)
class Outcome:
    """What came of a case's message: the tools it called, the answer, the memories and the
    workspace's files after."""

    tools: list[str]
    answer: str
    memories: list[str]
    files: list[str]
    tasks: list[str]  # each its prompt and how often it repeats, as "Call Ada (once)"
    seconds: float
    cached: float  # of the prompts' tokens, the share the engine's cache held
    lists: list[str]  # each thing on the household's lists, as "Shopping: milk"


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


def makes(pattern: str) -> Check:
    made = lambda o: any(re.search(pattern, name) for name in o.files)  # noqa: E731
    return lambda o: None if made(o) else f"made no file /{pattern}/"


def remembers_nothing(o: Outcome) -> str | None:
    return f"remembers {o.memories}" if o.memories else None


def lists(pattern: str) -> Check:
    found = lambda o: any(re.search(pattern, t, re.I) for t in o.lists)  # noqa: E731
    return lambda o: None if found(o) else f"listed nothing /{pattern}/ but {o.lists}"


def schedules(pattern: str) -> Check:
    found = lambda o: any(re.search(pattern, t, re.I) for t in o.tasks)  # noqa: E731
    return lambda o: None if found(o) else f"scheduled no task /{pattern}/ but {o.tasks}"


def forgot(pattern: str) -> Check:
    return lambda o: f"still remembers /{pattern}/" if remembers(pattern)(o) is None else None


def unsure(o: Outcome) -> str | None:
    # an answer that says it does not know, rather than one made up
    known = r"(n['’]t|not)( \w+){0,2} (know|have|remember|told|mention)|not sure"
    return None if re.search(known, o.answer, re.I) else "did not say it does not know"


def unsaid(pattern: str) -> Check:
    return lambda o: f"said /{pattern}/" if re.search(pattern, o.answer, re.I) else None


@dataclass(frozen=True)
class Case:
    name: str
    message: str  # in a new conversation
    checks: list[Check]
    # remembered before it: each a fact of "about", or a plan with its last day, days from today
    memories: list[str | tuple[str, int]] = field(default_factory=list)
    before: list[str] = field(default_factory=list)  # messages of earlier conversations, each one's
    files: dict[str, str] = field(default_factory=dict)  # the workspace's, by name, attached to it
    reviewed: bool = False  # reviewed for memories after, as the agent does once it is idle
    tidied: bool = False  # its memory tidied after, as the agent does each night
    # remembered of another person of the household, whom the message's person is not
    others: list[str] = field(default_factory=list)


NONE = ("search", "fetch", "weather", "remember", "forget", "recall", "read", "run", "schedule")
CASES = [
    Case("chat", "Write a haiku about autumn.", [uncalled(*NONE)]),
    Case("arithmetic", "What is 17 * 23?", [says(r"391"), uncalled(*NONE)]),
    Case("known fact", "What is the capital of Australia?",
         [says("Canberra"), uncalled("remember")]),
    Case("news", "What's in the news today about space exploration?", [called("search")]),
    Case("reads pages", "When does the British Museum open tomorrow? Check its website.",
         [called("fetch")]),
    Case("research", "Research the pros and cons of heat pumps for a house in a cold climate.",
         [called("search", 2), called("fetch", 4), says(r"\[\d+\]")]),
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
    Case("attachment", "What time does it start, and what should I bring?",
         [called("read"), says("7"), says("salad|dessert")],
         files={"invitation.txt": "You're invited to Mia's 30th! Saturday 18 October, 7 pm, at "
                "22 Oak Road. Please bring a salad or a dessert."}),
    Case("document", "Make a Word document with a packing list for a weekend camping trip.",
         [called("run"), makes(r"\.docx$")]),
    Case("spreadsheet", "Make an Excel budget: rent 900, food 350 and transport 80 a month, with "
         "yearly totals.", [called("run"), makes(r"\.xlsx$")]),
    Case("reminder", "Remind me in 2 hours to take out the trash.",
         [called("schedule"), schedules(r"trash.*\(once\)")]),
    Case("briefing", "Every weekday at 7:30, give me the weather in London.",
         [called("schedule"), schedules(r"weather.*London.*\(weekdays\)")]),
    Case("watch", "Every morning at 7, check whether it will rain in London that day, and only "
         "tell me if it will.", [called("schedule"), schedules(r"London.*\(daily\) if .*rain")]),
    Case("shopping", "We're out of milk and eggs, put them on the shopping list.",
         [called("add_to_list"), lists(r"shopping: .*milk"), lists(r"shopping: .*eggs")]),
    Case("noticed", "I'm planning my daughter Ada's 7th birthday party for next Saturday. Suggest "
         "5 party games.", [remembers("Ada")], reviewed=True),
    Case("no junk", "What's 2^2^2^2?", [remembers_nothing], reviewed=True),
    Case("cleans up", "Thanks, that's all for today.", [forgot(r"2\^2"), remembers("Izmir")],
         memories=["The user asked about the value of 2^2^2^2.", "The user lives in Izmir."],
         reviewed=True),
    Case("update", "By the way, I moved to Ankara last month.",
         [called("remember"), remembers("Ankara"), forgot("Izmir")],
         memories=["The user lives in Izmir."]),
    Case("past plan", "Thanks!", [forgot(r"is flying|will fly"), remembers("daughter")],
         memories=[("The user is flying to Rome next Tuesday.", -2), "The user has a daughter."],
         tidied=True),
    Case("doesn't know", "What's my sister's name?", [unsure, uncalled("search")]),
    Case("not from pages", "Who is Linus Torvalds? Look him up on the web.",
         [called("search"), remembers_nothing], reviewed=True),
    Case("keeps to its person", "What's my dog called?", [unsure, unsaid("Rex")],
         others=["The user's dog is called Rex."]),
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", default="http://127.0.0.1:8080", help="leat serve's address")
    parser.add_argument("--search", default="http://127.0.0.1:8888", help="a SearXNG's address")
    parser.add_argument(
        "--sandbox", type=Path, default=Path.home() / ".local/share/leat/sandbox",
        help="the sandbox's environment, as leat/agent/sandbox.txt makes it",
    )  # fmt: skip
    parser.add_argument("-n", "--runs", type=int, default=3, help="runs of each case")
    parser.add_argument("--think", action="store_true", help="have the model think first")
    parser.add_argument("-k", help="the cases whose names hold this alone")
    args = parser.parse_args()
    cases = [case for case in CASES if not args.k or args.k in case.name]
    rows, passed = [], 0
    for case in cases:
        failures, seconds, cached = [], [], []
        for _ in range(args.runs):
            outcome = _run(case, args)
            seconds.append(outcome.seconds)
            cached.append(outcome.cached)
            why = [w for check in case.checks if (w := check(outcome))]
            failures += why[:1]
            print(f"{case.name}: {why[0] if why else 'passed'}", file=sys.stderr, flush=True)
        passed += args.runs - len(failures)
        failed = "; ".join(sorted(set(failures)))
        rows.append(f"| {case.name} | {args.runs - len(failures)}/{args.runs} | "
                    f"{statistics.mean(seconds):.1f} | {statistics.mean(cached):.0%} | "
                    f"{failed} |")  # fmt: skip
    mode = "thinking" if args.think else "not thinking"
    print(f"\n{passed} of {len(cases) * args.runs} passed, {mode}\n")
    print("| case | passed | mean s | cached | failures |\n|---|---:|---:|---:|---|")
    print("\n".join(rows))


def _run(case: Case, args: argparse.Namespace) -> Outcome:
    # the case, in an agent of a state of its own
    with tempfile.TemporaryDirectory() as data:
        environment = args.sandbox if args.sandbox.exists() else None
        workspace = Workspace(Path(data) / "workspace", environment)
        engine = Client(args.engine)
        tools = [*web.tools(args.search, workspace, engine), *weather.tools(),
                 *files.tools(workspace)]  # fmt: skip
        agent = Agent(Store(Path(data) / "leat.db"), engine, tools, workspace)
        person = None  # no household, but where another person's memories are
        if case.others:
            person = agent.store.add_person("Sam")["id"]
            other = agent.store.add_person("Ada")["id"]
            for memory in case.others:
                agent.remember(memory, "about", person=other)
        for memory in case.memories:
            if isinstance(memory, tuple):  # a plan, until days from today
                until = datetime.date.today() + datetime.timedelta(days=memory[1])
                agent.remember(memory[0], "plans", until=until.isoformat(), person=person)
            else:
                agent.remember(memory, "about", person=person)
        for message in case.before:
            _wait(agent, agent.send(None, message, args.think, person=person), person)
        for name, text in case.files.items():
            workspace.path(name).write_text(text)
        start = time.monotonic()
        id = agent.send(None, case.message, args.think, list(case.files), person=person)
        messages = _wait(agent, id, person)
        if case.reviewed and messages:
            background.review(agent, id)
        if case.tidied:
            background.tidy(agent, person)
        seconds = time.monotonic() - start
        tools_called = [m["name"] for m in messages if m["role"] == "tool"]
        answer = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else ""
        memories = [m["text"] for m in agent.memories(person)]
        names = [f["name"] for f in workspace.files()]
        tasks = [f"{t['prompt']} ({t['repeat']})" + (f" if {c}" if (c := t["condition"]) else "")
                 for t in agent.tasks(person)]  # fmt: skip
        infos = [m["info"] for m in messages if m["role"] == "assistant" and "read" in m["info"]]
        held, read = sum(i["cached"] or 0 for i in infos), sum(i["read"] for i in infos)
        share = held / (held + read) if held + read else 0.0
        listed = [f"{li['name']}: {i['text']}" for li in agent.store.lists() for i in li["items"]]
        return Outcome(tools_called, answer or "", memories, names, tasks, seconds, share, listed)


def _wait(agent: Agent, id: str, person: int | None) -> list[dict]:
    # a conversation's messages once its turn has ended, or none if it failed and took it back
    end = time.monotonic() + TIMEOUT
    while (c := agent.conversation(id, person)) is not None and c["running"]:
        if time.monotonic() > end:
            agent.stop(id, person)
        time.sleep(0.1)
    return c["messages"] if c else []


if __name__ == "__main__":
    main()
