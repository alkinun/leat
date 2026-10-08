import datetime

import pytest

from leat.agent.tools import tasks

NOW = datetime.datetime(2026, 10, 7, 21, 14)  # a Wednesday


@pytest.mark.parametrize(
    "at, expected",
    [
        ("2026-10-08 17:00", datetime.datetime(2026, 10, 8, 17)),
        ("2026-10-08T07:30", datetime.datetime(2026, 10, 8, 7, 30)),
        ("in 20 minutes", NOW + datetime.timedelta(minutes=20)),
        ("in 1 hour", NOW + datetime.timedelta(hours=1)),
        ("In 2 Days", NOW + datetime.timedelta(days=2)),
        ("22:00", datetime.datetime(2026, 10, 7, 22)),  # later today
        ("08:00", datetime.datetime(2026, 10, 8, 8)),  # passed today: tomorrow's
    ],
)
def test_when(at, expected):
    assert tasks.when(at, NOW) == expected


def test_when_refused():
    with pytest.raises(ValueError, match="not 'YYYY-MM-DD HH:MM', nor 'in 20 minutes'"):
        tasks.when("tomorrow morning", NOW)


def test_following():
    friday = datetime.datetime(2026, 10, 9, 7, 30)
    assert tasks.following(friday, "daily", 9) == datetime.datetime(2026, 10, 10, 7, 30)
    assert tasks.following(friday, "weekdays", 9) == datetime.datetime(2026, 10, 12, 7, 30)
    assert tasks.following(friday, "weekly", 9) == datetime.datetime(2026, 10, 16, 7, 30)
    assert tasks.following(friday, "hourly", 9) == datetime.datetime(2026, 10, 9, 8, 30)
    assert tasks.following(friday, "once", 9) is None
    # monthly on the 31st: the last of a shorter month, then the 31st again
    february = tasks.following(datetime.datetime(2026, 1, 31, 9), "monthly", 31)
    assert february == datetime.datetime(2026, 2, 28, 9)
    assert tasks.following(february, "monthly", 31) == datetime.datetime(2026, 3, 31, 9)
    december = datetime.datetime(2026, 12, 15, 9)
    assert tasks.following(december, "monthly", 15) == datetime.datetime(2027, 1, 15, 9)


def test_first():
    # a task that repeats, set after its time today, begins at its next; one that runs once, not
    morning = datetime.datetime(2026, 10, 7, 8)
    assert tasks.first(morning, "daily", NOW) == datetime.datetime(2026, 10, 8, 8)
    assert tasks.first(morning, "weekly", NOW) == datetime.datetime(2026, 10, 14, 8)
    assert tasks.first(morning, "once", NOW) == morning
    # of weekdays, one set for a weekend begins on Monday
    saturday = datetime.datetime(2026, 10, 10, 8)
    assert tasks.first(saturday, "weekdays", NOW) == datetime.datetime(2026, 10, 12, 8)
    # a monthly one on the 31st, set long after, keeps to it past shorter months
    long_ago = datetime.datetime(2026, 1, 31, 8)
    assert tasks.first(long_ago, "monthly", NOW) == datetime.datetime(2026, 10, 31, 8)


def test_checked():
    at = NOW + datetime.timedelta(hours=1)
    assert tasks.checked("  Remind   them ", at, "once", NOW) == "Remind them"
    for prompt, when, repeat, error in [
        ("", at, "once", "nothing to do"),
        ("x" * 501, at, "once", "500 characters at most"),
        ("Remind them", at, "yearly", "repeat must be one of"),
        ("Remind them", NOW, "once", "that time has passed: it is Wednesday 7 October 2026, 21:14"),
    ]:
        with pytest.raises(ValueError, match=error):
            tasks.checked(prompt, when, repeat, NOW)


def test_describe():
    def task(first: datetime.datetime, due: datetime.datetime, repeat: str) -> dict:
        return {"first": first.timestamp(), "next": due.timestamp(), "repeat": repeat}

    friday = datetime.datetime(2026, 10, 9, 8)
    assert tasks.describe(task(friday, friday, "once")) == "once, on Friday 9 October at 08:00"
    assert (
        tasks.describe(task(friday, friday, "daily")) == "every day at 08:00, next Friday 9 October"
    )
    assert tasks.describe(task(friday, friday, "weekdays")) == (
        "every weekday at 08:00, next Friday 9 October")  # fmt: skip
    assert tasks.describe(task(friday, friday, "weekly")) == (
        "every Friday at 08:00, next Friday 9 October")  # fmt: skip
    assert tasks.describe(task(friday, friday, "monthly")) == (
        "every month on the 9th at 08:00, next Friday 9 October")  # fmt: skip
    assert tasks.describe(task(friday, friday, "hourly")) == "every hour, at 00 past, next at 08:00"
