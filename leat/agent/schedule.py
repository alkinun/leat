"""When a workflow runs by itself: its schedule, as a few words of when, in the box's local time.

    daily 08:00          every day
    weekdays 08:00       Monday to Friday
    weekly mon 08:00     each Monday, or tue, wed, thu, fri, sat or sun
    monthly 1 08:00      each month on its 1st, or on its last day where it has fewer days
    files                when files arrive in its space, uploaded, put on the network drive or
                         synced from a folder

A schedule of times counts from a time: when it was set, or the workflow last ran. It is due once
the first time it names after that has come, so that a box that was off runs a workflow it missed
once, as it starts, not once for each time it missed.
"""

import calendar
import datetime
import re

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
ARRIVING = "files"  # the schedule of a workflow that runs as files arrive
_SCHEDULE = re.compile(
    rf"(?P<every>daily|weekdays|weekly (?P<day>{'|'.join(DAYS)})|monthly (?P<date>\d{{1,2}}))"
    r" (?P<hour>\d{2}):(?P<minute>\d{2})"
)
AHEAD = 62  # days looked ahead for a schedule's next time, more than any's between two


def parsed(schedule: str) -> re.Match[str]:
    """A schedule's parts. Raises ValueError if it is not one."""
    match = _SCHEDULE.fullmatch(" ".join(schedule.split()).lower())
    if match is None or int(match["hour"]) > 23 or int(match["minute"]) > 59:
        raise ValueError(
            f"a schedule is 'daily 08:00', 'weekdays 08:00', 'weekly mon 08:00', "
            f"'monthly 1 08:00' or 'files', not {schedule!r}"
        )
    if match["date"] and not 1 <= int(match["date"]) <= 31:
        raise ValueError(f"a month's day is 1 to 31, not {match['date']}")
    return match


def normal(schedule: str) -> str:
    """A schedule as it is kept: its words in lowercase, one space between them. Raises
    ValueError if it is not one."""
    if schedule.strip().lower() == ARRIVING:
        return ARRIVING
    return parsed(schedule).group(0)


def following(schedule: str, after: float) -> float:
    """The first time a schedule names after a time, both as seconds since the epoch."""
    match = parsed(schedule)
    start = datetime.datetime.fromtimestamp(after)
    at = datetime.time(int(match["hour"]), int(match["minute"]))
    for ahead in range(AHEAD):
        day = start.date() + datetime.timedelta(days=ahead)
        when = datetime.datetime.combine(day, at)
        if when > start and _names(match, day):
            return when.timestamp()
    raise AssertionError(f"{schedule!r} names no day in {AHEAD}")


def due(schedule: str, since: float, now: float) -> bool:
    """Whether a schedule counted from `since` has named a time by `now`: never of ARRIVING,
    which no time makes due."""
    return schedule != ARRIVING and following(schedule, since) <= now


def _names(match: re.Match[str], day: datetime.date) -> bool:
    # whether a schedule names a day
    if match["every"] == "daily":
        return True
    if match["every"] == "weekdays":
        return day.weekday() < 5
    if match["day"]:
        return day.weekday() == DAYS.index(match["day"])
    last = calendar.monthrange(day.year, day.month)[1]
    return day.day == min(int(match["date"]), last)
