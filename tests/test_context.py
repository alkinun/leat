import datetime

from leat.agent import context

SYSTEM = {"role": "system", "content": "You are Leat."}
CALL = {"id": "a", "type": "function", "function": {"name": "fetch", "arguments": '{"url": "u"}'}}
MESSAGES = [
    SYSTEM,
    {"role": "user", "content": "Read it", "info": {"files": ["plan.md"]}},
    {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [CALL]},
    {"role": "tool", "tool_call_id": "a", "name": "fetch", "content": "x" * 900},
    {"role": "assistant", "content": "Read.", "reasoning_content": "Hmm.", "info": {"tokens": 2}},
]


def test_prompt():
    # whole, but for what only people see; the attachments named
    read = context.prompt(MESSAGES, {})
    attached = "Read it\n\n(Attached, in the workspace: plan.md)"
    assert read[1] == {"role": "user", "content": attached}
    assert read[2] == {"role": "assistant", "content": "", "tool_calls": [CALL]}
    assert read[4] == {"role": "assistant", "content": "Read.", "reasoning_content": "Hmm."}
    # old long answers cleared; a summary in the system prompt, going on at a reply
    read = context.prompt(MESSAGES, {"cleared": 4, "summary": "Goal: it.", "summarized": 2})
    assert read[0]["content"] == "You are Leat." + context.SUMMARY.format(summary="Goal: it.")
    assert read[1] == {"role": "user", "content": context.GOING_ON}
    assert read[3]["content"] == context.CLEARED and len(read) == 5
    # one saved in the workspace says where, to read again
    saved = [*MESSAGES[:3], MESSAGES[3] | {"info": {"saved": ".web/a.md"}}, *MESSAGES[4:]]
    read = context.prompt(saved, {"cleared": 4})
    assert read[3]["content"] == context.SAVED.format(saved=".web/a.md")
    assert MESSAGES[3]["content"] == "x" * 900  # the conversation's own, untouched


def test_stamp():
    # a user's message begins with when it was sent, a task's too; a reply's does not
    at = datetime.datetime(2026, 10, 8, 14, 5).timestamp()
    task = {"role": "user", "content": "Brief me", "info": {"at": at, "task": 3}}
    due = "(Your scheduled task [3] is due now: Brief me)"
    assert context.message(task)["content"] == f"[Thursday 8 October 2026, 14:05]\n{due}"
    reply = {"role": "assistant", "content": "Done.", "info": {"at": at}}
    assert context.message(reply)["content"] == "Done."


def test_estimate():
    # the engine's count of the last, and an estimate of what came after it
    assert context.estimate(MESSAGES, {"used": 1000, "at": 5}) == 1000
    assert context.estimate(MESSAGES, {"used": 1000, "at": 4}) > 1000
    assert context.estimate(MESSAGES, {}) > 300  # the page's 900 characters, as a third of tokens


def test_compact():
    # a long old answer is cleared before anything is summarized; a short stretch is not summarized
    messages = [*MESSAGES, {"role": "user", "content": "And?"}]
    summarized = []

    def summarize(before, span, tokens):
        summarized.append(span)
        return "Goal: it."

    state = context.compact(messages, {}, 600, 0, summarize)
    assert state["cleared"] == 4 and summarized == []
    assert context.compact(messages, state, 600, 0, summarize, force=True) == state | {"used": None}
    # a long stretch, summarized, with the summary before, to the turn running
    long = [SYSTEM] + [{"role": r, "content": "y" * 600} for r in ("user", "assistant") * 4]
    state = context.compact(long, {"summary": "Before."}, 1000, 0, summarize)
    assert state["summary"] == "Goal: it." and state["summarized"] == len(long) - 2
    assert summarized[-1] == long[1:-2]


def test_compact_keeps_the_turn():
    # the turn running keeps the answers it found, the earlier turns' cleared first; and its own
    # are cleared when that is not enough
    def page(i: int) -> list[dict]:
        call = {"id": f"c{i}", "type": "function", "function": {"name": "fetch", "arguments": "{}"}}
        return [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"c{i}", "name": "fetch", "content": "p" * 900},
        ]

    earlier = [SYSTEM, {"role": "user", "content": "One"}, *page(1), *page(2),
               {"role": "assistant", "content": "Done."}]  # fmt: skip
    running = [*earlier, {"role": "user", "content": "Two"}, *page(3), *page(4), *page(5)]
    state = context.compact(running, {}, 2600, 0, lambda *a: "Goal: it.")
    view = context.prompt(running, state)
    assert [m["content"] for m in view if m["role"] == "tool"] == [context.CLEARED] * 2 + [
        "p" * 900
    ] * 3
    state = context.compact(running, {}, 1500, 0, lambda *a: "Goal: it.")
    assert context.prompt(running, state)[-3]["content"] == context.CLEARED  # its own, at last


def test_transcript():
    assert context.transcript(MESSAGES[1:], shown=10) == (
        "User: Read it\n\n(Attached, in the workspace: plan.md)\n\n"
        'Assistant called fetch with {"url": "u"}\n\n'
        "fetch answered: xxxxxxxxxx …\n\n"
        "Assistant: Read."
    )


def test_images():
    # the images a message shows, attached or read, named for the agent to show, each estimated
    # as IMAGE tokens; a tool's old answer of one cleared as a long one is
    user = {"role": "user", "content": "What is it?", "info": {"files": ["cat.png", "plan.md"]}}
    read = {"role": "tool", "tool_call_id": "a", "name": "read", "content": "The image a.jpg.",
            "info": {"images": ["a.jpg"]}}  # fmt: skip
    messages = [SYSTEM, user, MESSAGES[2] | {"tool_calls": [CALL]}, read]
    view = context.prompt(messages, {})
    assert view[1]["images"] == ["cat.png"] and view[3]["images"] == ["a.jpg"]
    plain = [m | {"info": {}} for m in messages]
    images = context.estimate(messages, {}) - context.estimate(plain, {})
    assert images >= 2 * context.IMAGE
    cleared = context.prompt(messages, {"cleared": 4})[3]
    assert cleared["content"] == context.CLEARED and "images" not in cleared
    long = [*messages[:3], read | {"content": "x" * (context.KEEP + 1)}]  # cleared for its length
    assert "images" not in context.prompt(long, {"cleared": 4})[3]
