"""The weather: now, and for the next week, at a place, from Open-Meteo, which asks for no key.

Its forecasts are what a weather site's pages show, which fetch cannot read: they draw them in the
browser, or refuse a program.
"""

import datetime
import json
import urllib.parse
import urllib.request
from typing import Any

from leat.agent.tools import Result, Tool, strings

GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 15  # seconds a request may take
# the weather's codes, as the WMO numbers them, in words
CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
    48: "freezing fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle", 61: "light rain", 63: "rain",
    65: "heavy rain", 66: "freezing rain", 67: "heavy freezing rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}  # fmt: skip
CURRENT = "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m"
DAILY = (
    "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
    "precipitation_sum,wind_speed_10m_max"
)


def tools() -> list[Tool]:
    return [
        Tool(
            "weather",
            "The weather at a place: now, and each day of the next week",
            strings(place="a town or city, and its country if it is ambiguous, as 'Paris, France'"),
            lambda context, place: weather(place),
        )
    ]


def weather(place: str) -> Result:
    name, _, hint = (part.strip() for part in place.partition(","))
    found = _get(GEOCODING, {"name": name, "count": 10, "language": "en"}).get("results") or []
    # the first that the hint names, as a country or a region, or the first
    hinted = [f for f in found if hint.lower() in " ".join(_where(f)).lower()]
    if not (at := (hinted or found or [None])[0]):
        raise ValueError(f"there is no place called {place!r}")
    w = _get(FORECAST, {
        "latitude": at["latitude"], "longitude": at["longitude"], "current": CURRENT,
        "daily": DAILY, "timezone": "auto", "forecast_days": 7,
    })  # fmt: skip
    where, now, days = ", ".join(_where(at)), w["current"], w["daily"]
    lines = [
        f"{where}, at {now['time'].replace('T', ' ')} local time: "
        f"{CODES.get(now['weather_code'], 'unknown')}, {_degrees(now['temperature_2m'])}, "
        f"feeling {_degrees(now['apparent_temperature'])}, humidity "
        f"{now['relative_humidity_2m']}%, wind {now['wind_speed_10m']:.0f} km/h."
    ]
    keys = ("time", "weather_code", "temperature_2m_min", "temperature_2m_max",
            "precipitation_probability_max", "precipitation_sum", "wind_speed_10m_max")  # fmt: skip
    for day, code, low, high, chance, rain, wind in zip(*(days[k] for k in keys), strict=True):
        date = datetime.date.fromisoformat(day)
        lines.append(
            f"{date:%A} {date.day} {date:%B}: {CODES.get(code, 'unknown')}, {_degrees(low)} to "
            f"{_degrees(high)}, {chance}% chance of rain ({rain} mm), wind up to {wind:.0f} km/h."
        )
    return Result("\n".join(lines), {"place": where})


def _where(found: dict[str, Any]) -> list[str]:
    # a place's name, its region and its country, as many as it has, once each
    names = (found.get(key) for key in ("name", "admin1", "country"))
    return list(dict.fromkeys(name for name in names if name))


def _degrees(celsius: float) -> str:
    return f"{celsius:.0f} °C ({celsius * 9 / 5 + 32:.0f} °F)"


def _get(url: str, query: dict[str, Any]) -> dict[str, Any]:
    address = f"{url}?{urllib.parse.urlencode(query)}"
    with urllib.request.urlopen(address, timeout=TIMEOUT) as response:
        return json.loads(response.read())
