"""The box's own Wi-Fi, a hotspot, as an office without a free network port sets it up: once the box
has gone without a network for a minute, of a box that knows no Wi-Fi network, or five, of one that
does, which may yet come back, as a router restarting, its radio is an access point of its own,
"Leat-7J32", of a passphrase the box's screen shows, with a QR code that joins it. A phone on it is
shown a page of the office's networks, whatever it asks, as a hotel's Wi-Fi shows its own: every
name is the box's, of the DNS answered here, and Caddy sends the hotspot's every request here. A
network chosen, the hotspot closes, and the box joins it; one it cannot join in half a minute, as of
a passphrase mistyped, is forgotten, and the hotspot opens again, saying so. The hotspot closes too
once the box has a network of its own, a cable plugged in, or a known network back in reach, as the
hotspot looks every minute.

Its address and its devices' are systemd-networkd's, of 82-hotspot.network; the radio's mode the
box's system helper's to change, as root, of leat.box.system.
"""

import contextlib
import functools
import html
import json
import secrets
import socket
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from leat import locales
from leat.box import system
from leat.box.address import ipv4
from leat.box.provision import ROOT

ADDRESS = "10.42.0.1"  # the box's on its hotspot, of 82-hotspot.network
PAGE = ("127.0.0.1", 8090)  # the hotspot's page, which Caddy serves its devices whatever they ask
STATE = Path("/run/leat-hotspot/state.json")  # its name and passphrase, of the screen's to show
AGENT = "http://127.0.0.1:8000"  # leat agent, of the box's language
LOOK = 5  # seconds between looks
AFTER = 60  # seconds without a network before the hotspot opens, of a box that knows none
KNOWN_AFTER = 300  # of one that knows a Wi-Fi network, which may come back
RESCAN = 60  # seconds between the hotspot's scans, for a known network back in reach; LOOK, till
# it found any, as one just opened finds none
JOINING = 30  # seconds a network chosen has to give the box an address
CLOSING = 2  # seconds a page has to reach its phone before the hotspot closes
WORDS = "abcdefghijkmnpqrstuvwxyz23456789"  # of a passphrase: none mistaken for another
LANGUAGE = "de"  # the page's, of a box whose admins chose none yet, as its screen's


class Hotspot:
    """The box's hotspot, `name`, opened and closed as the box goes without a network, as `online`
    tells, and its helper, `ask`, opens it, at `clock`'s times: open while it has a passphrase; and
    the networks it found in reach, and the one the box last could not join, as its page and the
    box's screen, of the file `state`, say."""

    def __init__(
        self, name: str, ask: Callable[[dict[str, Any]], dict[str, Any]] = system.ask,
        online: Callable[[], bool] = lambda: _online(), clock: Callable[[], float] = time.monotonic,
        state: Path = STATE,
    ):  # fmt: skip
        self.name, self.ask, self.online, self.clock, self.state = name, ask, online, clock, state
        self.passphrase: str | None = None  # while it is open
        self.networks: list[dict[str, Any]] = []
        self.failed: str | None = None  # the network chosen last that the box could not join
        self._since: float | None = None  # since when the box has gone without a network
        self._scanned = 0.0
        # its passphrase, the same while the box goes without a network, as a phone joined to it
        # before a network chosen failed joins it again; and the known networks that came back in
        # reach and could not be joined, which it closes for no more till the box has a network
        self._key = _passphrase()
        self._unjoinable: set[str] = set()
        # the network the box joins now, if any: its name, since when, and whether it was chosen,
        # rather than a known one back in reach
        self._joining: tuple[str, float, bool] | None = None
        self._lock = threading.Lock()

    def look(self) -> None:
        """Looks at the box now: opens the hotspot, or closes it, as it has gone without a network,
        or has one again."""
        with self._lock:
            now = self.clock()
            if self.online():
                self._since = self._joining = self.failed = None
                self._key, self._unjoinable = _passphrase(), set()
                if self.passphrase:
                    self._close()
                return
            if self._joining:
                name, since, chosen = self._joining
                if now - since < JOINING:
                    return
                self._joining = None
                if chosen:  # its passphrase mistyped, as like as not: forgotten, to type again
                    self.failed = name
                    self.ask({"wifi": "forget", "name": name})
                else:  # its passphrase changed, or its DHCP down: not left for again
                    self._unjoinable.add(name)
                return self._open()
            if self.passphrase:
                if not self.ask({"hotspot": "status"}).get("started"):  # as iwd restarted
                    return self._open()
                if now - self._scanned >= (RESCAN if self.networks else LOOK):
                    self._scan()
                    back = [n["name"] for n in self.networks
                            if n["known"] and n["name"] not in self._unjoinable]  # fmt: skip
                    if back:
                        self._close()  # the box joins it by itself
                        self._joining = (back[0], now, False)
                return
            status = self.ask({"hotspot": "status"})
            if not status.get("radio"):
                return
            if status.get("started"):  # of this service before it restarted: closed, to open anew
                self.ask({"hotspot": "stop"})
            self._since = now if self._since is None else self._since
            if now - self._since >= (KNOWN_AFTER if status.get("known") else AFTER):
                self._open()

    def join(self, name: str, passphrase: str | None) -> None:
        """Joins a network a phone on the hotspot chose, closing the hotspot CLOSING seconds from
        now, as its page reaches the phone first. Raises ValueError of a name or passphrase that
        is not one, or while the hotspot is closed, as a phone off it has no network to choose."""
        with self._lock:
            if self.passphrase is None:
                raise ValueError("the box's hotspot is closed")
            self.ask({"wifi": "join", "name": name, "passphrase": passphrase})
            self._joining, self.failed = (name, self.clock() + CLOSING, True), None

        def closing() -> None:
            time.sleep(CLOSING)
            with self._lock:
                if self.passphrase:
                    self._close()

        threading.Thread(target=closing, daemon=True).start()

    def _open(self) -> None:
        self.ask({"hotspot": "start", "name": self.name, "passphrase": self._key})
        self.passphrase = self._key  # once it is open
        self._scanned = self.clock() - RESCAN  # its networks found at its next look
        self._write()

    def _close(self) -> None:
        self.ask({"hotspot": "stop"})
        self.passphrase = None
        self._write()

    def _scan(self) -> None:
        self.networks = self.ask({"hotspot": "scan"}).get("networks", [])
        self._scanned = self.clock()

    def _write(self) -> None:
        # the hotspot's name and passphrase, and the network last not joined, for the screen; none
        # while it is closed
        if self.passphrase is None:
            self.state.unlink(missing_ok=True)
            return
        told = {"name": self.name, "passphrase": self.passphrase, "failed": self.failed}
        partial = self.state.with_suffix(".partial")
        partial.write_text(json.dumps(told))
        partial.chmod(0o640)
        partial.replace(self.state)


def answer(query: bytes, address: str = ADDRESS) -> bytes | None:
    """The DNS answer to a query, of every name `address`, as a hotspot's devices are to find the
    box whatever they ask; none of a packet that is no query."""
    if len(query) < 12:
        return None
    ident, flags, questions = struct.unpack("!HHH", query[:6])
    if flags & 0x8000 or questions != 1:
        return None
    end = 12
    while end < len(query) and query[end]:  # the name's labels, uncompressed in a query
        end += query[end] + 1
    if end + 5 > len(query):
        return None
    kind, klass = struct.unpack("!HH", query[end + 1 : end + 5])
    found = kind == 1 and klass == 1  # an IPv4 address: the box's; of anything else, none
    header = struct.pack("!HHHHHH", ident, 0x8480 | flags & 0x0100, 1, int(found), 0, 0)
    record = struct.pack("!HHHIH", 0xC00C, 1, 1, 10, 4) + socket.inet_aton(address)
    return header + query[12 : end + 5] + (record if found else b"")


def answering(where: tuple[str, int] = (ADDRESS, 53)) -> None:
    """Answers every DNS query at `where`, the hotspot's address, which is there only while it is
    open: bound before, of IP_FREEBIND."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.IPPROTO_IP, getattr(socket, "IP_FREEBIND", 15), 1)
        s.bind(where)
        while True:
            query, sender = s.recvfrom(512)
            if reply := answer(query, where[0]):
                with contextlib.suppress(OSError):
                    s.sendto(reply, sender)


class Page(BaseHTTPRequestHandler):
    """The hotspot's page, of every address a phone on it asks for: the networks in reach, a
    passphrase, and the box joining the one chosen."""

    server: "PageServer"

    def do_GET(self) -> None:
        self._page()

    def do_POST(self) -> None:
        length = min(int(self.headers.get("Content-Length") or 0), 4096)
        form = urllib.parse.parse_qs(self.rfile.read(length).decode(errors="replace"))
        name = (form.get("other", [""])[0] or form.get("name", [""])[0]).strip()
        passphrase = form.get("passphrase", [""])[0] or None
        if not name:
            return self._page()
        try:
            self.server.hotspot.join(name, passphrase)
        except (ValueError, OSError) as e:
            return self._page(str(e))
        self._page(joining=name)

    def _page(self, refused: str = "", joining: str = "") -> None:
        language = _language()
        body = page(self.server.hotspot, language, locales.translated(refused, language), joining)
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # what the office's phones ask is no one's to keep


class PageServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, hotspot: Hotspot, where: tuple[str, int] = PAGE):
        self.hotspot = hotspot
        super().__init__(where, Page)


def page(hotspot: Hotspot, language: str, refused: str = "", joining: str = "") -> str:
    """The hotspot's page, in a language: the networks in reach, the strongest first, each chosen
    by a tap, and its passphrase; or that the box joins the one chosen."""
    t = functools.partial(locales.t, language=language)
    if joining:
        said = t("The box is joining {name}. Once it is on it, its screen shows how to set Leat "
                 "up. Your phone goes back to its own Wi-Fi.",
                 name=html.escape(joining))  # fmt: skip
        return _html(language, t("Joining…"), f"<p>{said}</p>")
    rows = "".join(
        f'<label><input type="radio" name="name" value="{html.escape(n["name"], quote=True)}"'
        f'{" checked" if i == 0 else ""}><span>{html.escape(n["name"])}</span>'
        f'<small>{"▂▄▆█"[: max(1, n["strength"])]}{" 🔒" if n["secure"] else ""}</small></label>'
        for i, n in enumerate(hotspot.networks)
    )  # fmt: skip
    failed = t("The box couldn't join {name}. Is its password right?",
               name=html.escape(hotspot.failed or ""))  # fmt: skip
    failed = f'<p class="error">{failed}</p>' if hotspot.failed else ""
    error = f'<p class="error">{html.escape(refused)}</p>' if refused else ""
    found = rows or f"<p>{t('Looking for networks…')}</p>"
    again = "" if rows else '<meta http-equiv="refresh" content="5">'  # till it found them
    body = f"""{again}
<p>{t("The box joins it, and then shows on its screen how to set Leat up.")}</p>{failed}{error}
<form method="post">{found}
<input name="other" placeholder="{t("Or type a network's name")}" autocomplete="off">
<input name="passphrase" type="password" placeholder="{t("Its password")}" autocomplete="off">
<button>{t("Join")}</button></form>"""
    return _html(language, t("Choose the office's Wi-Fi"), body)


def _html(language: str, title: str, body: str) -> str:
    # a page of the hotspot's, as small as a phone shows at once
    return f"""<!doctype html><html lang="{language}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Leat · {title}</title>
<style>body{{font:16px system-ui,sans-serif;max-width:28rem;margin:2rem auto;padding:0 1rem;
color:#0a0a0a}}h1{{font-size:1.5rem}}label{{display:flex;gap:.6rem;align-items:center;
padding:.7rem 0;border-bottom:1px solid #e6e6e6}}label span{{flex:1}}small{{color:#6e6e6e}}
input[type=password],input[name=other]{{display:block;width:100%;box-sizing:border-box;
margin-top:1rem;padding:.7rem;font:inherit;border:1px solid #ccc;border-radius:6px}}
button{{margin-top:1rem;width:100%;padding:.8rem;font:inherit;font-weight:600;color:#fff;
background:#0a0a0a;border:0;border-radius:6px}}.error{{color:#d92d20}}</style></head>
<body><h1>{title}</h1>{body}</body></html>"""


def ssid(root: Path = ROOT) -> str:
    """The hotspot's name: Leat- and the first four of the box's ID, as its name has it, or of its
    machine's ID, of a box not yet provisioned."""
    with contextlib.suppress(OSError):
        for line in (root / "box.env").read_text().splitlines():
            if line.startswith("LEAT_ID="):
                return f"Leat-{line.split('=', 1)[1].strip()[:4].upper()}"
    with contextlib.suppress(OSError):
        return f"Leat-{Path('/etc/machine-id').read_text()[:4].upper()}"
    return "Leat-0000"


def _passphrase() -> str:
    # a passphrase of the hotspot's, typed easily off the screen: "kq7x-2mpa"
    word = lambda: "".join(secrets.choice(WORDS) for _ in range(4))  # noqa: E731
    return f"{word()}-{word()}"


def _online() -> bool:
    # whether the box has a network of its own, a way out, as a cable or a Wi-Fi joined gives it
    try:
        ipv4()
        return True
    except OSError:
        return False


def _language() -> str:
    # the box's language, as leat agent tells it, or the page's own
    with (
        contextlib.suppress(OSError, ValueError),
        urllib.request.urlopen(f"{AGENT}/api/box", timeout=3) as response,
    ):
        return json.loads(response.read()).get("language") or LANGUAGE
    return LANGUAGE


def main() -> None:
    hotspot = Hotspot(ssid())
    threading.Thread(target=answering, daemon=True).start()
    threading.Thread(target=PageServer(hotspot).serve_forever, daemon=True).start()
    while True:
        try:
            hotspot.look()
        except (OSError, ValueError) as e:  # the helper away, or refusing: looked at again
            print(f"leat-hotspot: {e}", file=sys.stderr, flush=True)
        time.sleep(LOOK)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
