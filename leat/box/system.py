"""The box's system, changed for leat agent, which runs without root: a few things alone, each
checked, by a helper systemd runs as root for each request, on a socket leat's group alone reaches.

    {"timezone": "Europe/Istanbul"}                       sets the box's time zone
    {"wifi": "scan"}                                       the Wi-Fi networks in reach
    {"wifi": "join", "name": "Office", "passphrase": "…"}  joins one, and again whenever in reach
    {"wifi": "forget", "name": "Office"}                   forgets one
    {"hotspot": "status"}                                  whether it has Wi-Fi, its own hotspot's
                                                           and the networks it knows
    {"hotspot": "start", "name": "Leat-7J32",              opens its own Wi-Fi, a hotspot
     "passphrase": "…"}
    {"hotspot": "scan"}                                    the networks in reach of the hotspot
    {"hotspot": "stop"}                                    closes it, its Wi-Fi joining networks
    {"network": {"address": "192.168.1.50/24",             a fixed address of its wired link's,
     "gateway": "192.168.1.1", "dns": ["192.168.1.1"]}}    from which it falls back to DHCP if its
                                                           gateway answers not
    {"network": "dhcp"}                                    its address of DHCP's again
    {"network": "status"}                                  which, and whether it fell back
    {"power": "off"}                                       switches the box off, or "restart"
    {"reset": true}                                        erases the box's state, and restarts
    {"update": "status"}                                   its version, and an update ready, if one
    {"update": "check"}                                    looks for an update now
    {"diagnostics": true}                                  the box's system, as text, for support

Each answer is a JSON object: {} or what was asked, or {"error": why}. The agent asks by ask().

What it sets that the system's partition holds, as the time zone, it keeps on the state's too, and
puts back at boot, by --restore, as an update replaces the system's.

Wi-Fi is iwd's, its addresses systemd-networkd's, as a wired link's are: a network joined is one of
iwd's known networks, a file of /var/lib/iwd, which iwd joins by itself whenever it is in reach; a
wired link stays the box's way out while it has one. Its radio is iwd's access point instead while
the box opens a hotspot of its own, as leat.box.hotspot does to be set up without a cable.
"""

import contextlib
import ipaddress
import json
import os
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
# what the helper set that the system's partition holds, as the time zone's /etc/localtime, kept on
# the state's, as the system's is replaced by an update, and put back at boot
KEPT = Path("/var/lib/leat-system")
# the flag of a factory reset, on the ESP: empty, as systemd-repart heeds its being there alone
RESET = Path("/boot/leat-reset")
# a fixed address of the wired link's, as systemd-networkd reads it, made anew at boot of what is
# kept
FIXED = Path("/run/systemd/network/10-leat-fixed.network")
CHECK = 40  # seconds after a fixed address is set before its gateway must answer, or DHCP's again
UKIS = Path("/boot/EFI/Linux")  # the box's UKIs, leat_<version>[+tries].efi, each its system's
OS_RELEASE = Path("/etc/os-release")
LONGEST = 4096  # bytes of a request at most
SCANNING = 6  # seconds a scan for networks is waited for at most
_PLAIN = re.compile(r"[A-Za-z0-9 _-]+")  # a network's name iwd names its file by, as it is
_HOTSPOT = re.compile(r"Leat-[A-Z0-9]{4}")  # the name of the box's own hotspot
_AP = "net.connman.iwd.AccessPoint"


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
        if "network" in request:
            return network(request["network"])
        if "hotspot" in request:
            return hotspot(request["hotspot"], request.get("name"), request.get("passphrase"))
        if "power" in request:
            return power(request["power"])
        if request.get("reset") is True:
            return reset()
        if request.get("update") == "status":
            return update_status()
        if request.get("diagnostics") is True:
            return {"text": diagnostics()}
        if request.get("update") == "check":
            subprocess.run(["systemctl", "--no-block", "start", "systemd-sysupdate.service"],
                           check=True, capture_output=True, timeout=10)  # fmt: skip
            return {}
        raise Refused(f"there is no such request: {json.dumps(request)[:100]}")
    except (Refused, OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}


def set_timezone(zone: Any) -> dict[str, Any]:
    """Sets the box's time zone, one of the IANA database's, by timedated."""
    if not isinstance(zone, str) or zone not in zoneinfo.available_timezones():
        raise Refused(f"there is no time zone {zone!r}")
    subprocess.run(["timedatectl", "set-timezone", zone], check=True, capture_output=True,
                   timeout=10)  # fmt: skip
    KEPT.mkdir(exist_ok=True)  # systemd's StateDirectory, on a box
    _write(KEPT / "timezone", f"{zone}\n", 0o644)
    return {}


def restore() -> None:
    """Puts back what the helper set that the system's partition holds, as an update replaced it,
    or that is of a boot alone: the time zone, of its kept name; the wired link's fixed address."""
    with contextlib.suppress(FileNotFoundError, Refused):  # none kept, or no zone the system has
        if (kept := (KEPT / "timezone").read_text().strip()) != zone():
            set_timezone(kept)
    with contextlib.suppress(FileNotFoundError, ValueError, OSError, subprocess.SubprocessError):
        _fix(json.loads((KEPT / "network.json").read_text()))


def network(how: Any) -> dict[str, Any]:
    """The wired link's address: fixed, of {"address": "192.168.1.50/24", "gateway": …, "dns": […]},
    which the box falls back from to DHCP's if its gateway answers not CHECK seconds after, as an
    office's IT may mistype one and lose the box else; DHCP's again, of "dhcp"; or which it is now,
    and whether it fell back, of "status"."""
    kept, fell = KEPT / "network.json", KEPT / "network-fell-back"
    if how == "status":
        with contextlib.suppress(FileNotFoundError, ValueError):
            return {"fixed": json.loads(kept.read_text()), "fell_back": False}
        return {"fixed": None, "fell_back": fell.exists()}
    fell.unlink(missing_ok=True)
    if how == "dhcp":
        kept.unlink(missing_ok=True)
        _fix(None)
        return {}
    fixed = _checked_network(how)
    KEPT.mkdir(exist_ok=True)
    _write(kept, json.dumps(fixed) + "\n", 0o644)
    _fix(fixed)
    subprocess.run(["systemctl", "stop", "leat-network-check.timer"], capture_output=True,
                   timeout=10)  # an earlier check's, of an address set again since  # fmt: skip
    subprocess.run(["systemd-run", f"--on-active={CHECK}", "--unit=leat-network-check",
                    "--description=Whether the box's fixed address reaches its gateway",
                    sys.executable, "-m", "leat.box.system", "--check-network"],
                   check=True, capture_output=True, timeout=10)  # fmt: skip
    return {}


def check_network() -> None:
    """Falls back to DHCP's address if the fixed one's gateway answers not, as `network` asks once
    one is set."""
    with contextlib.suppress(FileNotFoundError, ValueError):
        fixed = json.loads((KEPT / "network.json").read_text())
        if not _answers(fixed["gateway"]):
            network("dhcp")
            (KEPT / "network-fell-back").touch()


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
        _checked_passphrase(passphrase)
        text, kind = f"[Security]\nPassphrase={passphrase}\n", "psk"
    forget(name)  # of another kind, if it was known
    IWD.mkdir(mode=0o700, exist_ok=True)
    path = IWD / f"{_file(name)}.{kind}"
    _write(path, text, 0o600)
    return {}


def forget(name: Any) -> dict[str, Any]:
    """Forgets a Wi-Fi network, which the box leaves and joins no more."""
    _checked_name(name)
    for kind in ("psk", "open", "8021x"):
        (IWD / f"{_file(name)}.{kind}").unlink(missing_ok=True)
    return {}


def hotspot(action: Any, name: Any = None, passphrase: Any = None) -> dict[str, Any]:
    """The box's own hotspot, of its Wi-Fi as iwd's access point: whether the box has Wi-Fi, the
    hotspot is open, and the networks it knows, of "status"; opened as `name`, of `passphrase`, of
    "start"; the networks in reach, as scan() has them, and whether the box knows each, of "scan";
    closed, the radio joining networks again, of "stop"."""
    device = _device()
    if action == "status":
        started = device is not None and _AP in _objects().get(device, {})
        return {"radio": device is not None, "started": started, "known": _known()}
    if device is None:
        raise Refused("this box has no Wi-Fi")
    if action == "start":
        if not isinstance(name, str) or not _HOTSPOT.fullmatch(name):
            raise Refused("a hotspot's name is Leat- and four letters or digits")
        _checked_passphrase(passphrase)
        _bus("set-property", device, "net.connman.iwd.Device", "Mode", "s", "ap")
        for _ in range(SCANNING * 4):  # its access point's interface, once iwd has it
            if _AP in _objects().get(device, {}):
                break
            time.sleep(0.25)
        _bus("call", device, _AP, "Start", "ss", name, passphrase)
        return {}
    if action == "scan":
        _bus("call", device, _AP, "Scan", check=False)
        for _ in range(SCANNING * 4):
            if not _bus("get-property", device, _AP, "Scanning", check=False).get("data"):
                break
            time.sleep(0.25)
        ordered = _bus("call", device, _AP, "GetOrderedNetworks")["data"][0]
        known, networks = set(_known()), list[dict[str, Any]]()
        for found in map(_properties, ordered):
            if (network := found.get("Name")) and network not in {n["name"] for n in networks}:
                networks.append({"name": network, "secure": found.get("Type") != "open",
                                 "strength": _strength(found.get("SignalStrength") or -10000),
                                 "known": network in known})  # fmt: skip
        return {"networks": networks}
    if action == "stop":
        _bus("call", device, _AP, "Stop", check=False)
        _bus("set-property", device, "net.connman.iwd.Device", "Mode", "s", "station")
        return {}
    raise Refused(f"there is no hotspot request {action!r}")


def power(how: Any) -> dict[str, Any]:
    """Switches the box off, or restarts it, by systemd, as cleanly as ever: its answer given
    first, the box going a moment after."""
    if how not in ("off", "restart"):
        raise Refused('the box\'s power is "off" or "restart"')
    verb = "poweroff" if how == "off" else "reboot"
    subprocess.run(["systemctl", "--no-block", verb], check=True, capture_output=True, timeout=10)
    return {}


def reset() -> dict[str, Any]:
    """Erases the box's state, everything of the office's and its settings, keeping its identity:
    a flag on the ESP, of which systemd-repart, on the boot the box restarts to, deletes the state's
    partition and makes it anew, of a new key, before anything unlocks it. systemd's own factory
    reset is none of the box's, as it marks itself done in the initrd, which makes no partitions."""
    _write(RESET, "", 0o600)
    return power("restart")


# the system's services, whose logs say nothing of anyone's work, as leat agent's errors might of a
# file's name, and Caddy's requests of the path asked for: Caddy's own of its certificate alone
LOGGED = ["leat-address", "leat-screen", "leat-health", "leat-fallback", "leat-provision",
          "leat-system@*", "systemd-sysupdate*", "systemd-repart", "systemd-cryptsetup@*",
          "systemd-networkd", "iwd", "systemd-timesyncd"]  # fmt: skip
DIAGNOSED = [
    ("Version", ["cat", "/etc/os-release"]),
    ("Up", ["uptime"]),
    ("Failed units", ["systemctl", "--failed", "--no-legend", "--no-pager"]),
    ("The box's units", ["systemctl", "list-units", "--all", "--no-legend", "--no-pager", "leat-*",
                         "systemd-sysupdate*", "systemd-repart*", "systemd-cryptsetup@*"]),
    ("Timers", ["systemctl", "list-timers", "--all", "--no-legend", "--no-pager"]),
    ("Boot entries", ["ls", "-l", "/boot/EFI/Linux"]),
    ("Disks", ["lsblk", "-o", "NAME,SIZE,TYPE,PARTLABEL,FSTYPE,MOUNTPOINTS"]),
    ("Room", ["df", "-h", "/", "/etc/leat", "/var", "/boot"]),
    ("Memory", ["free", "-h"]),
    ("Network", ["ip", "-br", "address"]),
    ("Routes", ["ip", "route"]),
    ("DNS", ["resolvectl", "status", "--no-pager"]),
    ("Identity", ["cat", "/etc/leat/box.env"]),
    ("Logs", ["journalctl", "--boot", "--no-pager", "--lines=400", "--output=short-iso",
              *(f"--unit={u}" for u in LOGGED)]),
    ("Certificate", ["journalctl", "--boot", "--no-pager", "--output=cat", "--unit=leat-proxy",
                     "--grep", '"logger":"tls']),
    ("Kernel errors", ["journalctl", "--boot", "--no-pager", "--dmesg", "--priority=err"]),
]  # fmt: skip


def diagnostics() -> str:
    """The box's system, as text an admin sends us when something is wrong: its version, units,
    disks, network and the logs of its system's services; nothing of anyone's work, nor the office's
    name, nor who uses it."""
    parts = []
    for title, command in DIAGNOSED:
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            said = (done.stdout + done.stderr).strip()
        except (OSError, subprocess.SubprocessError) as e:
            said = f"({e})"
        parts.append(f"## {title}\n\n{said}\n")
    return "\n".join(parts)


def update_status() -> dict[str, Any]:
    """The box's version; the newer one an update installed, which the box boots into as it next
    restarts, at night by systemd-sysupdate-reboot.timer, if one is; one that failed, its tries
    spent, the box back on the one before, if one did; whether it is looking for an update now; and
    when it last looked, in seconds since the epoch, if it has."""
    version = next((line.split("=", 1)[1].strip('"') for line in OS_RELEASE.read_text().splitlines()
                    if line.startswith("IMAGE_VERSION=")), None)  # fmt: skip
    ukis = list(UKIS.glob("leat_*.efi"))
    failed = [_version(p.name) for p in ukis if "+0-" in p.name]  # of no tries left
    installed = [_version(p.name) for p in ukis if "+0-" not in p.name]
    newer = max((v for v in installed if version and _order(v) > _order(version)), default=None,
                key=_order)  # fmt: skip
    checking = _shown("systemd-sysupdate.service", "ActiveState") in ("activating", "active")
    # of its last look, the timer's or an admin's
    last = _shown("systemd-sysupdate.service", "ExecMainExitTimestamp").removeprefix("@")
    return {"version": version, "ready": newer, "checking": checking,
            "checked": int(last) if last.isdigit() else None,
            "failed": max(failed, default=None, key=_order)}  # fmt: skip


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


def _write(path: Path, text: str, mode: int) -> None:
    # a file whole, of its mode from the start, on the disk before it takes its name, as a power cut
    # just after left a file written but not yet on the disk empty
    temporary = path.with_name(f".{path.name}.new")
    with open(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _shown(unit: str, name: str) -> str:
    # a unit's property, as systemctl shows it, a time as "@<seconds since the epoch>"
    return subprocess.run(["systemctl", "show", unit, f"--property={name}", "--value",
                           "--timestamp=unix"], capture_output=True, text=True, timeout=10,
                          check=False).stdout.strip()  # fmt: skip


def _version(name: str) -> str:
    # a UKI's version, of its file's name: leat_<version>[+left[-done]].efi
    return name.removeprefix("leat_").removesuffix(".efi").split("+", 1)[0]


def _order(version: str) -> tuple[int, ...]:
    # a version's order: of its minute, as mkosi.version writes it, 2026.10.11.2241-<commit>
    return tuple(int(n) for n in version.split("-", 1)[0].split(".") if n.isdigit())


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
    # the path of iwd's first Wi-Fi station, or None without a Wi-Fi device, or while it is the
    # box's hotspot
    for path, interfaces in sorted(_objects().items()):
        if "net.connman.iwd.Station" in interfaces:
            return path
    return None


def _device() -> str | None:
    # the path of iwd's first Wi-Fi device, a station or an access point, or None without one
    for path, interfaces in sorted(_objects().items()):
        if "net.connman.iwd.Device" in interfaces:
            return path
    return None


def _known() -> list[str]:
    # the names of the Wi-Fi networks the box knows, which iwd joins whenever they are in reach
    return sorted(_properties(i["net.connman.iwd.KnownNetwork"]).get("Name") or ""
                  for i in _objects().values() if "net.connman.iwd.KnownNetwork" in i)  # fmt: skip


def _checked_network(how: Any) -> dict[str, Any]:
    # a fixed address as networkd is to have it, checked. Raises Refused of one that is no address
    if not isinstance(how, dict):
        raise Refused('a network is "dhcp", or a fixed address, its gateway and DNS servers')
    try:
        address = ipaddress.IPv4Interface(how.get("address"))
        gateway = ipaddress.IPv4Address(how.get("gateway"))
        dns = [str(ipaddress.IPv4Address(d)) for d in how.get("dns") or [str(gateway)]]
    except (ValueError, TypeError) as e:
        raise Refused(f"that is no address: {e}") from None
    net = address.network
    if not 8 <= net.prefixlen <= 30 or address.ip in (net.network_address, net.broadcast_address):
        raise Refused(f"{address} is no address of a host on a network")
    if gateway not in net or gateway == address.ip:
        raise Refused(f"the gateway {gateway} is not on {net}")
    if len(dns) > 3:
        raise Refused("three DNS servers at most")
    return {"address": str(address), "gateway": str(gateway), "dns": dns}


def _fix(fixed: dict[str, Any] | None) -> None:
    # the wired links' fixed address, as networkd is to have it, or none, and networkd told
    if fixed is None:
        FIXED.unlink(missing_ok=True)
    else:
        dns = "".join(f"DNS={d}\n" for d in fixed["dns"])
        FIXED.parent.mkdir(parents=True, exist_ok=True)
        _write(FIXED, f"# leat.box.system's, of an admin's fixed address\n[Match]\nType=ether\n\n"
               f"[Network]\nAddress={fixed['address']}\nGateway={fixed['gateway']}\n{dns}"
               "MulticastDNS=yes\n", 0o644)  # fmt: skip
    subprocess.run(["networkctl", "reload"], check=True, capture_output=True, timeout=10)
    for link in Path("/sys/class/net").iterdir():  # each wired link configured anew
        if (link / "type").read_text().strip() == "1" and not (link / "wireless").exists() and (
                link / "device").exists():  # fmt: skip
            subprocess.run(["networkctl", "reconfigure", link.name], capture_output=True,
                           timeout=10)  # fmt: skip


def _answers(gateway: str, wait: float = 5) -> bool:
    # whether a gateway answers on the link, as its neighbour's entry shows once a packet was sent
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(b"", (gateway, 9))
    for _ in range(int(wait * 4)):
        told = subprocess.run(["ip", "-j", "neigh", "show", gateway], capture_output=True,
                              text=True, timeout=5).stdout  # fmt: skip
        states = {s for entry in json.loads(told or "[]") for s in entry.get("state", [])}
        if states & {"REACHABLE", "STALE", "DELAY", "PROBE", "PERMANENT"}:
            return True
        time.sleep(0.25)
    return False


def _checked_passphrase(passphrase: Any) -> None:
    # raises Refused of what is no WPA2 passphrase
    if (
        not isinstance(passphrase, str)
        or not 8 <= len(passphrase) <= 63
        or not all(" " <= c <= "~" for c in passphrase)
    ):
        raise Refused("a passphrase is 8 to 63 letters, digits and signs")


def _properties(interface: dict[str, Any]) -> dict[str, Any]:
    # an interface's properties, their values without their types
    return {name: value.get("data") for name, value in interface.items()}


def main() -> None:
    # a request on stdin, the socket systemd accepted, and its answer on stdout; or, of --restore,
    # at boot, what the helper set put back
    if sys.argv[1:] == ["--restore"]:
        return restore()
    if sys.argv[1:] == ["--check-network"]:
        return check_network()
    raw = sys.stdin.buffer.read(LONGEST + 1)
    try:
        request = json.loads(raw) if len(raw) <= LONGEST else None
    except ValueError:
        request = None
    sys.stdout.write(json.dumps(handle(request)))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
