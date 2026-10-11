"""The box's screen: what a display plugged into it shows, in place of a login, which no box has.

    python -m leat.box.screen      # by leat-screen.service, on tty1

It says, in a few large words, whether the box is ready, and the address to open, with a QR code of
it for a phone: starting, without a network, not provisioned, ready for its first person to set it
up, or ready. Ready is said once a browser would find it so: the box's certificate, checked as a
browser checks it, of its name, and the app answering behind it. It is drawn again only as it
changes, every few seconds.
"""

import contextlib
import functools
import json
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from leat.box.address import ipv4
from leat.box.provision import ROOT

EVERY = 3  # seconds between looks
BOLD, PLAIN = "\033[1;97m", "\033[0m"  # of the words that matter: bright white, bold
AGENT = "http://127.0.0.1:8000"  # leat agent, behind Caddy


@dataclass(frozen=True)
class State:
    """What the screen says: the box's name, if provisioned; its IPv4 address, if it has one;
    whether its certificate holds for its name; whether the app answers; and whether the box is yet
    to be set up by its first person."""

    name: str | None
    address: str | None
    certified: bool = False
    answering: bool = False
    empty: bool = False


def look(root: Path = ROOT) -> State:
    """The box's state, as it is now."""
    name = _name(root)
    address = _address()
    if name is None or address is None:
        return State(name, address)
    answering, empty = _app(name)
    return State(name, address, _certified(name), answering, empty)


def render(state: State, qr: str = "") -> list[str]:
    """The screen's lines, of a state, and the QR code of its address, if any."""
    title = f"{BOLD}Leat{PLAIN}"
    if state.address is None:
        return [title, "", f"{BOLD}No network{PLAIN}", "", "Plug the network cable into the box."]
    if state.name is None:
        return [
            title,
            "",
            f"{BOLD}Not set up{PLAIN}",
            "",
            "This box has no identity yet: call Leat.",
        ]
    url = f"https://{state.name}"
    if not (state.certified and state.answering):
        return [title, "", f"{BOLD}Starting...{PLAIN}", "", url]
    said = "Open this address to set up your Leat" if state.empty else "Ready"
    lines = [title, "", f"{BOLD}{said}{PLAIN}", "", url, ""]
    lines += qr.splitlines() if qr else []
    return [*lines, "", "or type leat.local in a browser on the office's network"]


def draw(lines: list[str], width: int, height: int) -> str:
    """The lines, each centred, the block of them in the middle of the screen, which is cleared."""
    top = max(0, (height - len(lines)) // 2)
    longest = max((_shown(line) for line in lines), default=0)
    left = max(0, (width - longest) // 2)
    body = "\n".join(
        " " * max(left, (width - _shown(line)) // 2) + line if line else "" for line in lines
    )
    return "\033[2J\033[H\033[?25l" + "\n" * top + body


@functools.lru_cache(maxsize=4)
def qr(text: str) -> str:
    """A QR code of the text, of block characters, as qrencode draws one; or "" without it."""
    if shutil.which("qrencode") is None:
        return ""
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        drawn = subprocess.run(
            ["qrencode", "-t", "UTF8", "-m", "2", text],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout  # fmt: skip
        return drawn.rstrip("\n")
    return ""


def _shown(line: str) -> int:
    # the characters a line takes on the screen, its styles' codes but none
    return len(line.replace(BOLD, "").replace(PLAIN, ""))


def _name(root: Path) -> str | None:
    # the box's name, of its provisioned box.env
    with contextlib.suppress(OSError):
        for line in (root / "box.env").read_text().splitlines():
            if line.startswith("LEAT_NAME="):
                return line.split("=", 1)[1].strip() or None
    return None


def _address() -> str | None:
    # the IPv4 address the box reaches others from, or None without a network
    with contextlib.suppress(OSError):
        return ipv4()
    return None


def _certified(name: str) -> bool:
    # whether the box's own HTTPS, on this machine, has a certificate a browser takes for its name
    context = ssl.create_default_context()
    with (
        contextlib.suppress(OSError),
        socket.create_connection(("127.0.0.1", 443), timeout=3) as raw,
        context.wrap_socket(raw, server_hostname=name),
    ):
        return True
    return False


def _app(name: str) -> tuple[bool, bool]:
    # whether the app answers, and whether the box is yet to be set up, as it tells a device that
    # has not joined: its 401 says so
    request = urllib.request.Request(f"{AGENT}/api/me", headers={"Host": name})
    try:
        with urllib.request.urlopen(request, timeout=3):
            return True, False
    except urllib.error.HTTPError as e:
        with contextlib.suppress(ValueError):
            return True, bool(json.loads(e.read()).get("empty"))
        return True, False
    except OSError:
        return False, False


def main() -> None:
    shown: tuple[list[str], int, int] | None = None
    while True:
        state = look()
        lines = render(state, qr(f"https://{state.name}") if state.name else "")
        width, height = shutil.get_terminal_size((80, 25))
        if (lines, width, height) != shown:
            sys.stdout.write(draw(lines, width, height))
            sys.stdout.flush()
            shown = (lines, width, height)
        time.sleep(EVERY)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
