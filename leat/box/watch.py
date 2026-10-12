"""What of a box needs attention, which its screen and its admins are told, each a few words and a
sentence of what to do: its name not found on its office's network, as a router's DNS rebinding
protection hides it; the internet out of its reach; its certificate expiring, as it renews it over
the internet; its data's disk nearly full; its model out of reach; an update that failed, the box
back on the version before.

Looked at once a minute, in a thread of its own, so that whoever asks, as the box's screen every few
seconds, has what was last seen at once, never waiting on DNS or the box's helper.
"""

import shutil
import socket
import ssl
import threading
import time
from collections.abc import Callable
from pathlib import Path

from leat.box import system
from leat.box.address import ipv4

LOOK = 60  # seconds between looks
HELD = 600  # seconds the name goes unfound, or the internet unreached, before it is said, as a new
# address takes minutes, and a router restarting some
EXPIRING = 14  # days of the certificate left, or fewer, which are said: Caddy renews it at 30
FULL = 0.1  # of the data's disk left, or less, which is said
FRITZ = "Heimnetz › Netzwerk › Netzwerkeinstellungen › DNS-Rebind-Schutz"
KNOWN = "leatbox.net"  # a name found whenever the box's DNS answers at all
REACHED = ("api.leatbox.net", 443)  # reached whenever the box reaches the internet: its registry
HTTPS = ("127.0.0.1", 443)  # the box's front door, Caddy, whose certificate is looked at
TIMEOUT = 5  # seconds each look at the network may take
EXPIRED = 10  # OpenSSL's X509_V_ERR_CERT_HAS_EXPIRED


class Watch:
    """What of a box needs attention, of its name, its data's folder, and why its model is out of
    reach, if it is."""

    def __init__(self, name: str, data: Path, model_error: Callable[[], str | None],
                 clock: Callable[[], float] = time.monotonic):  # fmt: skip
        self.name, self.data, self.model_error, self.clock = name, data, model_error, clock
        self._since: dict[str, float] = {}  # since when each of what is said once held has held
        self.warnings: list[dict[str, str]] = []  # as last seen: each its key, title and sentence

    def start(self) -> None:
        """Looks once a minute from now on, in a thread of its own."""

        def looking() -> None:
            while True:
                self.look()
                time.sleep(LOOK)

        threading.Thread(target=looking, daemon=True).start()

    def look(self) -> list[dict[str, str]]:
        """Looks at the box now; returns what needs attention, as warnings then says."""
        found = []
        if self._held("name", _found(self.name) is False):
            hidden = ("The router hides addresses inside the office from the devices here. On a "
                      f"FRITZ!Box, add leatbox.net under {FRITZ}.")  # fmt: skip
            found.append(_warning("name", "Leat can't be found by its name", hidden))
        if self._held("internet", not _reached()):
            found.append(_warning("internet", "The box can't reach the internet",
                                  "Leat works in the office, but web searches, updates and its "
                                  "certificate wait. Check the box's cable or Wi-Fi, and the "
                                  "office's firewall."))  # fmt: skip
        left = _days_left(self.name)
        if left is not None and left <= EXPIRING:
            title = (f"Leat's certificate expires in {left} days" if left > 0
                     else "Leat's certificate has expired")  # fmt: skip
            found.append(_warning("certificate", title,
                                  "The box renews it over the internet, and couldn't so far; "
                                  "browsers warn of Leat once it expires. Check the box reaches "
                                  "the internet."))  # fmt: skip
        usage = shutil.disk_usage(self.data)
        if usage.free <= usage.total * FULL:
            found.append(_warning("disk", "The box is nearly full",
                                  f"{usage.free / 1e9:.0f} GB are left. Delete files Leat no "
                                  "longer needs."))  # fmt: skip
        if self.model_error():
            found.append(
                _warning(
                    "model",
                    "Leat can't reach its model",
                    "Chats get no answer until it is back. Restart the box.",
                )
            )
        if failed := self._failed():
            found.append(_warning("update", "An update failed",
                                  f"The box went back to the version before, as {failed} did not "
                                  "start. Leat works as it did."))  # fmt: skip
        self.warnings = found
        return found

    def _held(self, key: str, holds: bool) -> bool:
        # whether something has held for HELD seconds, as of this look and those before
        if not holds:
            self._since.pop(key, None)
            return False
        return self.clock() - self._since.setdefault(key, self.clock()) >= HELD

    def _failed(self) -> str | None:
        # the version of an update that failed, if one did, as the box's helper sees its UKIs
        try:
            return system.ask({"update": "status"}).get("failed")
        except (OSError, ValueError):
            return None


def _found(name: str) -> bool | None:
    # whether a name is found as the box's address, by the DNS the office's network gives the box;
    # None if it finds no name at all, as an office without internet, which is told of that instead
    try:
        found = {a[4][0] for a in socket.getaddrinfo(name, 443, socket.AF_INET)}
        return ipv4() in found
    except OSError:
        pass
    try:
        socket.getaddrinfo(KNOWN, 443, socket.AF_INET)
        return False
    except OSError:
        return None


def _reached() -> bool:
    # whether the box reaches the internet, as its registry
    try:
        socket.create_connection(REACHED, TIMEOUT).close()
        return True
    except OSError:
        return False


def _days_left(name: str) -> int | None:
    # the days the certificate the box serves its name with has left, 0 once expired; None if it
    # serves none yet, as Caddy has its first some seconds after the box's first boot, or another
    # that browsers do not trust, as of Let's Encrypt's staging CA, which a box under test has
    context = ssl.create_default_context()
    try:
        with (socket.create_connection(HTTPS, TIMEOUT) as connection,
              context.wrap_socket(connection, server_hostname=name) as tls):  # fmt: skip
            expires = ssl.cert_time_to_seconds(tls.getpeercert()["notAfter"])
    except ssl.SSLCertVerificationError as e:
        return 0 if e.verify_code == EXPIRED else None
    except OSError:
        return None
    return max(0, int((expires - time.time()) // 86400))


def _warning(key: str, title: str, said: str) -> dict[str, str]:
    return {"key": key, "title": title, "said": said}
