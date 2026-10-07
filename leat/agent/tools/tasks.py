"""Tasks, as ChatGPT's and Hermes Agent's: something the agent does later, once or again and again,
as a reminder or a morning's briefing, set by the model as it talks. At its time the task is sent
in the conversation it was set in, as if the user had asked it then, and the agent does it.

Times are the box's own, its wall clock's: a task each day at 08:00 keeps to 08:00 as the clocks
change.
"""

import calendar
import datetime
import re
from typing import TYPE_CHECKING, Any

from leat.agent.tools import Context, Result, Tool, schema

if TYPE_CHECKING:
    from leat.agent.agent import Agent

# how often a task runs, and each's words
REPEATS = {
    "once": "once", "hourly": "every hour", "daily": "every day", "weekdays": "every weekday",
    "weekly": "every week", "monthly": "every month",
}  # fmt: skip
MOST = 20  # tasks at most, as ChatGPT bounds its
LONGEST = 500  # characters of a task
_RELATIVE = re.compile(r"in (\d+) (minute|hour|day|week)s?", re.I)
_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def tools(agent: "Agent") -> list[Tool]:
    """schedule, unschedule and tasks, of the agent's tasks."""
    return [
        Tool(
            "schedule",
            "Do something later, once or again and again, as a reminder or a daily briefing: at "
            "its time you are given the task in this conversation, and do it",
            schema(
                required=2,
                task=(
                    "string",
                    "what to do then, as you would be asked it, as 'Remind the user "
                    "to call their mother' or 'Give the weather in Izmir and the news'",
                ),
                at=(
                    "string",
                    "when, the first time: 'YYYY-MM-DD HH:MM' in the user's time, or "
                    "'in 20 minutes'",
                ),
                repeat=("string", "how often", list(REPEATS)),
            ),  # fmt: skip
            lambda context, task, at, repeat="once": schedule(agent, context, task, at, repeat),
        ),
        Tool(
            "unschedule",
            "Cancel a scheduled task",
            schema(number=("integer", "the task's number")),
            lambda context, number: unschedule(agent, number),
        ),
        Tool(
            "tasks",
            "List the scheduled tasks",
            {"type": "object", "properties": {}},
            lambda context: listed(agent),
        ),
    ]


def schedule(agent: "Agent", context: Context, task: str, at: str, repeat: str) -> Result:
    now = datetime.datetime.now()
    t = agent.schedule(task, when(at, now), repeat, context.conversation)
    return Result(f"Scheduled, as task [{t['id']}]: {t['schedule']}.", {"task": t})


def unschedule(agent: "Agent", number: int | str) -> Result:
    t = agent.unschedule(int(number))
    return Result(f"Cancelled task [{t['id']}]: {t['prompt']}", {"task": t})


def listed(agent: "Agent") -> Result:
    lines = [f"[{t['id']}] {t['prompt']} ({t['schedule']})" for t in agent.tasks()]
    return Result("\n".join(lines) or "No task is scheduled.", {"tasks": len(lines)})


def checked(prompt: str, at: datetime.datetime, repeat: str, now: datetime.datetime) -> str:
    """A task's prompt, its spaces made one, if it may be one, at a time to come. Raises
    ValueError, saying why not."""
    prompt = " ".join(prompt.split())
    if not prompt:
        raise ValueError("there is nothing to do")
    if len(prompt) > LONGEST:
        raise ValueError(f"a task is said in {LONGEST} characters at most: say it shorter")
    if repeat not in REPEATS:
        raise ValueError(f"repeat must be one of {', '.join(REPEATS)}")
    if at <= now:
        raise ValueError(f"that time has passed: it is {_day(now)}, {now:%H:%M} now")
    return prompt


def first(at: datetime.datetime, repeat: str, now: datetime.datetime) -> datetime.datetime:
    """When a task first runs: at its time, or, of one that repeats, at the first of its times
    after now, as "every morning at 8" set in the evening begins tomorrow."""
    while repeat in REPEATS and repeat != "once" and at <= now:
        at = following(at, repeat, at.day) or at
    return at


def when(at: str, now: datetime.datetime) -> datetime.datetime:
    """The time `at` says, of `now`'s clock: 'YYYY-MM-DD HH:MM', 'HH:MM' the next time the clock
    shows it, or 'in 20 minutes', hours, days or weeks. Raises ValueError for other words."""
    at = at.strip()
    if relative := _RELATIVE.fullmatch(at):
        return now + datetime.timedelta(**{relative[2].lower() + "s": int(relative[1])})
    for form in _FORMATS:
        try:
            return datetime.datetime.strptime(at, form)
        except ValueError:
            pass
    if clock := re.fullmatch(r"(\d{1,2}):(\d{2})", at):
        t = now.replace(hour=int(clock[1]), minute=int(clock[2]), second=0, microsecond=0)
        return t if t > now else t + datetime.timedelta(days=1)
    raise ValueError(f"the time {at!r} is not 'YYYY-MM-DD HH:MM', nor 'in 20 minutes'")


def following(t: datetime.datetime, repeat: str, day: int) -> datetime.datetime | None:
    """The time a task runs after `t`, by how often it repeats, a monthly one on `day` of each month
    or its last; None for one that runs once."""
    if repeat == "hourly":
        return t + datetime.timedelta(hours=1)
    if repeat in ("daily", "weekdays"):
        t += datetime.timedelta(days=1)
        while repeat == "weekdays" and t.weekday() >= 5:
            t += datetime.timedelta(days=1)
        return t
    if repeat == "weekly":
        return t + datetime.timedelta(weeks=1)
    if repeat == "monthly":
        year, month = (t.year + 1, 1) if t.month == 12 else (t.year, t.month + 1)
        return t.replace(year=year, month=month, day=min(day, calendar.monthrange(year, month)[1]))
    return None


def describe(task: dict[str, Any]) -> str:
    """A task's times, in words, as "every day at 08:00, next Thursday 8 October"."""
    first, due = (datetime.datetime.fromtimestamp(task[key]) for key in ("first", "next"))
    clock, on = f"{due:%H:%M}", f"{due:%A} {due.day} {due:%B}"
    if task["repeat"] == "once":
        return f"once, on {on} at {clock}"
    if task["repeat"] == "hourly":
        return f"every hour, at {due:%M} past, next at {clock}"
    if task["repeat"] == "weekly":
        return f"every {first:%A} at {clock}, next {on}"
    if task["repeat"] == "monthly":
        return f"every month on the {_ordinal(first.day)} at {clock}, next {on}"
    return f"{REPEATS[task['repeat']]} at {clock}, next {on}"


def _day(t: datetime.datetime) -> str:
    return f"{t:%A} {t.day} {t:%B %Y}"


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
