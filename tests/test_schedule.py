"""When workflows run by themselves: their schedules' times, in the box's local time."""

import datetime

import pytest

from leat.agent import schedule


def at(*parts: int) -> float:
    return datetime.datetime(*parts).timestamp()  # type: ignore[arg-type]


def test_following():
    # the first time each schedule names after Saturday 10 October 2026, 22:00
    saturday = at(2026, 10, 10, 22, 0)
    expected = {
        "daily 08:00": at(2026, 10, 11, 8, 0),
        "weekdays 08:00": at(2026, 10, 12, 8, 0),
        "weekly mon 09:30": at(2026, 10, 12, 9, 30),
        "weekly sat 23:00": at(2026, 10, 10, 23, 0),  # later the same day
        "monthly 1 00:00": at(2026, 11, 1, 0, 0),
        "monthly 31 08:00": at(2026, 10, 31, 8, 0),
    }
    assert {s: schedule.following(s, saturday) for s in expected} == expected
    # a day past a month's last, on its last
    assert schedule.following("monthly 31 08:00", at(2026, 11, 1)) == at(2026, 11, 30, 8, 0)
    assert schedule.following("monthly 30 08:00", at(2027, 2, 1)) == at(2027, 2, 28, 8, 0)


def test_due():
    # due once its next time has come, however long ago: missed times run once
    since = at(2026, 10, 10, 22, 0)
    assert not schedule.due("daily 08:00", since, at(2026, 10, 11, 7, 59))
    assert schedule.due("daily 08:00", since, at(2026, 10, 11, 8, 0))
    assert schedule.due("daily 08:00", since, at(2026, 10, 20))


def test_normal():
    assert schedule.normal(" Weekly  MON 08:00 ") == "weekly mon 08:00"
    for wrong in ("hourly", "daily 8:00", "daily 24:00", "monthly 32 08:00", "weekly x 08:00"):
        with pytest.raises(ValueError):
            schedule.normal(wrong)
