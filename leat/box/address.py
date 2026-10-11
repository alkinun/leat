"""The box's name follows its address: the registry is told the box's addresses on its office's
network, its IPv4 one and an IPv6 unique local one if it has one, whenever they change, and once a
day as they are, so that a name lost is had again.

    python -m leat.box.address [--root /etc/leat] [--state /var/lib/leat-address]

A systemd timer runs it at boot and every few minutes; it asks the registry only when there is
something to tell. The box's secret comes as a systemd credential, registry-secret, or from the
root's file of that name.
"""

import argparse
import base64
import ipaddress
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from leat.box.provision import ROOT

STATE = Path("/var/lib/leat-address")
AGAIN = 86400  # seconds after which the addresses are told again, unchanged
TIMEOUT = 20  # seconds the registry may take
# where packets would go, to learn the address they would leave from: none is sent
_OUT = ("192.0.2.1", 9)  # TEST-NET-1, of no one
_ULA = ipaddress.ip_network("fc00::/7")


def addresses() -> dict[str, str | None]:
    """The box's addresses on its network: the IPv4 one it reaches others from, and an IPv6 unique
    local one of the same link, if it has one, not a VPN's, as Tailscale's. Raises OSError if it
    has no IPv4 address."""
    found = ipv4()
    return {"ipv4": found, "ipv6": _ula(found)}


def ipv4() -> str:
    """The IPv4 address the box reaches others from. Raises OSError if it has none."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(_OUT)  # chooses a route, sending nothing
        return s.getsockname()[0]


def _ula(ipv4: str) -> str | None:
    # the first stable IPv6 unique local address of the link that holds ipv4, as iproute2 lists it
    try:
        listed = subprocess.run(
            ["ip", "-j", "addr", "show"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
        links = json.loads(listed)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    for link in links:
        infos = link.get("addr_info", [])
        if not any(info.get("local") == ipv4 for info in infos):
            continue
        for info in infos:
            if info.get("family") != "inet6" or info.get("temporary") or info.get("deprecated"):
                continue
            if ipaddress.ip_address(info["local"]) in _ULA:
                return info["local"]
    return None


def tell(registry: str, id: str, secret: str, found: dict[str, str | None]) -> dict[str, Any]:
    """Tells the registry a box's addresses; returns its answer. Raises OSError of a registry
    that cannot be reached or refuses them, saying why."""
    body = json.dumps({k: v for k, v in found.items() if v}).encode()
    auth = base64.b64encode(f"{id}:{secret}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/json"}
    request = urllib.request.Request(f"{registry}/address", body, headers, method="PUT")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        try:
            reason = json.loads(e.read())["error"]
        except (ValueError, KeyError, TypeError):
            reason = e.reason
        raise OSError(f"the registry refused the addresses ({e.code}): {reason}") from None


def update(
    registry: str, id: str, secret: str, state: Path, found: dict[str, str | None],
    now: float | None = None,
) -> bool:  # fmt: skip
    """Tells the registry the addresses `found` if they changed since it was last told, or a day
    has gone by; whether it was told. Raises OSError."""
    now = time.time() if now is None else now
    last = state / "told.json"
    try:
        told = json.loads(last.read_text())
    except (OSError, ValueError):
        told = {}
    if told.get("addresses") == found and now - told.get("at", 0) < AGAIN:
        return False
    tell(registry, id, secret, found)
    state.mkdir(parents=True, exist_ok=True)
    last.write_text(json.dumps({"addresses": found, "at": now}))
    return True


def settings(root: Path) -> tuple[str, str, str]:
    """The box's ID, registry and secret, of its provisioned files and systemd's credentials."""
    env = dict(
        line.split("=", 1) for line in (root / "box.env").read_text().splitlines() if "=" in line
    )
    credentials = os.environ.get("CREDENTIALS_DIRECTORY")
    secret = Path(credentials, "registry-secret") if credentials else root / "registry-secret"
    return env["LEAT_ID"], env["LEAT_REGISTRY"], secret.read_text().strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m leat.box.address", description="Point the box's name at its address."
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="the box's provisioned files")
    parser.add_argument("--state", type=Path, default=STATE, help="where what was told is kept")
    args = parser.parse_args(argv)
    try:
        id, registry, secret = settings(args.root)
        found = addresses()
        if update(registry, id, secret, args.state, found):
            shown = ", ".join(a for a in found.values() if a)
            print(f"{id}'s name points at {shown}", flush=True)
    except (OSError, KeyError) as e:
        print(f"the box's name could not be pointed at its address: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
