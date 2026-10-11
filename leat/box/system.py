"""The box's system, changed for leat agent, which runs without root: a few things alone, each
checked, by a helper systemd runs as root for each request, on a socket leat's group alone reaches.

    {"timezone": "Europe/Istanbul"}                       sets the box's time zone
    {"wifi": "scan"}                                       the Wi-Fi networks in reach
    {"wifi": "join", "name": "Office", "passphrase": "…"}  joins one, and again whenever in reach
    {"wifi": "forget", "name": "Office"}                   forgets one
    {"power": "off"}                                       switches the box off, or "restart"

Each answer is a JSON object: {} or what was asked, or {"error": why}. The agent asks by ask().

Wi-Fi is iwd's, its addresses systemd-networkd's, as a wired link's are: a network joined is one of
iwd's known networks, a file of /var/lib/iwd, which iwd joins by itself whenever it is in reach; a
wired link stays the box's way out while it has one.
"""

import json
import re
import socket
import subprocess
import sys
import time
import zoneinfo
from pathlib import Path
from typing import Any

SOCKET = "/run/leat/system.sock"
IWD = Path("/var/lib/iwd")  # iwd's known networks
LONGEST = 4096  # bytes of a request at most
SCANNING = 6  # seconds a scan for networks is waited for at most
_PLAIN = re.compile(r"[A-Za-z0-9 _-]+")  # a network's name iwd names its file by, as it is


class Refused(Exception):
    """The request is not one the helper does."""


def handle(request: Any) -> dict[str, Any]:
    """The answer to a request, as the helper gives it."""
    try:
        if not isinstance(request, dict):
            raise Refused("a request is a JSON object")
        if "timezone" in request:
            return set_timezone(request["timezone"])
        wifi = request.get("wifi")
        if wifi == "scan":
            return scan()
        if wifi == "join":
            return join(request.get("name"), request.get("passphrase"))
        if wifi == "forget":
            return forget(request.get("name"))
        if "power" in request:
            return power(request["power"])
        raise Refused(f"there is no such request: {json.dumps(request)[:100]}")
    except (Refused, OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}


def set_timezone(zone: Any) -> dict[str, Any]:
    """Sets the box's time zone, one of the IANA database's, by timedated."""
    if not isinstance(zone, str) or zone not in zoneinfo.available_timezones():
        raise Refused(f"there is no time zone {zone!r}")
    subprocess.run(["timedatectl", "set-timezone", zone], check=True, capture_output=True,
                   timeout=10)  # fmt: skip
    return {}


def scan() -> dict[str, Any]:
    """The Wi-Fi networks in reach, the strongest first, each its name, whether it asks for a
    passphrase, its strength of 0 to 4 and whether the box is on it, and which the box is on."""
    station = _station()
    if station is None:
        return {"radio": False, "networks": [], "connected": None}
    _bus("call", station, "net.connman.iwd.Station", "Scan", check=False)
    for _ in range(SCANNING * 4):
        if not _bus("get-property", station, "net.connman.iwd.Station", "Scanning")["data"]:
            break
        time.sleep(0.25)
    ordered = _bus("call", station, "net.connman.iwd.Station", "GetOrderedNetworks")["data"][0]
    objects = _objects()
    networks, connected = [], None
    for path, signal in ordered:
        network = _properties(objects.get(path, {}).get("net.connman.iwd.Network", {}))
        if not network.get("Name"):
            continue
        connected_now = bool(network.get("Connected"))
        networks.append({"name": network["Name"], "secure": network.get("Type") != "open",
                         "strength": _strength(signal), "connected": connected_now})  # fmt: skip
        if connected_now:
            connected = network["Name"]
    return {"radio": True, "networks": networks, "connected": connected}


def join(name: Any, passphrase: Any = None) -> dict[str, Any]:
    """Joins a Wi-Fi network, open of no passphrase, as one of iwd's known networks, which it joins
    by itself whenever it is in reach."""
    _checked_name(name)
    if passphrase is None:
        text, kind = "[Settings]\nAutoConnect=true\n", "open"
    else:
        if (
            not isinstance(passphrase, str)
            or not 8 <= len(passphrase) <= 63
            or not all(" " <= c <= "~" for c in passphrase)
        ):
            raise Refused("a passphrase is 8 to 63 letters, digits and signs")
        text, kind = f"[Security]\nPassphrase={passphrase}\n", "psk"
    forget(name)  # of another kind, if it was known
    IWD.mkdir(mode=0o700, exist_ok=True)
    path = IWD / f"{_file(name)}.{kind}"
    path.touch(mode=0o600)
    path.write_text(text)
    return {}


def forget(name: Any) -> dict[str, Any]:
    """Forgets a Wi-Fi network, which the box leaves and joins no more."""
    _checked_name(name)
    for kind in ("psk", "open", "8021x"):
        (IWD / f"{_file(name)}.{kind}").unlink(missing_ok=True)
    return {}


def power(how: Any) -> dict[str, Any]:
    """Switches the box off, or restarts it, by systemd, as cleanly as ever: its answer given
    first, the box going a moment after."""
    if how not in ("off", "restart"):
        raise Refused('the box\'s power is "off" or "restart"')
    verb = "poweroff" if how == "off" else "reboot"
    subprocess.run(["systemctl", "--no-block", verb], check=True, capture_output=True, timeout=10)
    return {}


def zone() -> str | None:
    """The box's time zone, of the IANA database, as /etc/localtime names it, or None."""
    localtime = str(Path("/etc/localtime").resolve())
    return localtime.split("/zoneinfo/", 1)[1] if "/zoneinfo/" in localtime else None


def ask(request: dict[str, Any], path: str = SOCKET, timeout: float = 30) -> dict[str, Any]:
    """Asks the helper; returns its answer. Raises OSError if it cannot be reached, as off a box,
    and ValueError of a refusal, saying why."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(path)
        s.sendall(json.dumps(request).encode())
        s.shutdown(socket.SHUT_WR)
        answer = json.loads(b"".join(iter(lambda: s.recv(65536), b"")) or b"{}")
    if "error" in answer:
        raise ValueError(answer["error"])
    return answer


def _checked_name(name: Any) -> None:
    # a Wi-Fi network's name, of 1 to 32 bytes. Raises Refused
    if not isinstance(name, str) or not 1 <= len(name.encode()) <= 32:
        raise Refused("a Wi-Fi network's name is 1 to 32 bytes")


def _file(name: str) -> str:
    # the name of iwd's file of a network: its own, of letters, digits, spaces, - and _ alone, or
    # "=" and its bytes in hex
    return name if _PLAIN.fullmatch(name) else "=" + name.encode().hex()


def _strength(signal: int) -> int:
    # a network's strength, 0 to 4, of its signal in hundredths of a dBm
    return sum(signal >= limit * 100 for limit in (-80, -70, -60, -50))


def _bus(*args: str, check: bool = True) -> Any:
    # iwd's answer over D-Bus, by busctl, as JSON
    done = subprocess.run(["busctl", "--json=short", args[0], "net.connman.iwd", *args[1:]],
                          capture_output=True, text=True, timeout=10, check=check)  # fmt: skip
    return json.loads(done.stdout) if done.stdout.strip() else {}


def _objects() -> dict[str, dict[str, Any]]:
    # iwd's objects, each its interfaces' properties
    managed = _bus("call", "/", "org.freedesktop.DBus.ObjectManager", "GetManagedObjects")
    return managed["data"][0] if managed.get("data") else {}


def _station() -> str | None:
    # the path of iwd's first Wi-Fi station, or None without a Wi-Fi device
    for path, interfaces in sorted(_objects().items()):
        if "net.connman.iwd.Station" in interfaces:
            return path
    return None


def _properties(interface: dict[str, Any]) -> dict[str, Any]:
    # an interface's properties, their values without their types
    return {name: value.get("data") for name, value in interface.items()}


def main() -> None:
    # a request on stdin, the socket systemd accepted, and its answer on stdout
    raw = sys.stdin.buffer.read(LONGEST + 1)
    try:
        request = json.loads(raw) if len(raw) <= LONGEST else None
    except ValueError:
        request = None
    sys.stdout.write(json.dumps(handle(request)))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
