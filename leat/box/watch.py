"""What of a box needs attention, which its screen and its admins are told, each a few words and a
sentence of what to do: its name not found on its office's network, as a router's DNS rebinding
protection hides it; its data's disk nearly full; its model out of reach; an update that failed,
the box back on the version before.

Looked at once a minute, in a thread of its own, so that whoever asks, as the box's screen every few
seconds, has what was last seen at once, never waiting on DNS or the box's helper.
"""

import shutil
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

from leat.box import system
from leat.box.address import ipv4

LOOK = 60  # seconds between looks
UNFOUND = 600  # seconds the name goes unfound before it is said, as a new address takes minutes
FULL = 0.1  # of the data's disk left, or less, which is said
FRITZ = "Heimnetz › Netzwerk › Netzwerkeinstellungen › DNS-Rebind-Schutz"
KNOWN = "leatbox.net"  # a name found whenever the box's DNS answers at all


class Watch:
    """What of a box needs attention, of its name, its data's folder, and why its model is out of
    reach, if it is."""

    def __init__(self, name: str, data: Path, model_error: Callable[[], str | None],
                 clock: Callable[[], float] = time.monotonic):  # fmt: skip
        self.name, self.data, self.model_error, self.clock = name, data, model_error, clock
        self._unfound: float | None = None  # since when the name has gone unfound
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
        if self._unfound_for() >= UNFOUND:
            hidden = ("The router hides addresses inside the office from the devices here. On a "
                      f"FRITZ!Box, add leatbox.net under {FRITZ}.")  # fmt: skip
            found.append(_warning("name", "Leat can't be found by its name", hidden))
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

    def _unfound_for(self) -> float:
        # how long the name has gone unfound, by the DNS the office's network gives the box, while
        # that DNS answers at all, as an office without internet finds no name and is told of none
        found = _found(self.name)
        if found is None:
            return 0.0
        if found:
            self._unfound = None
            return 0.0
        self._unfound = self.clock() if self._unfound is None else self._unfound
        return self.clock() - self._unfound

    def _failed(self) -> str | None:
        # the version of an update that failed, if one did, as the box's helper sees its UKIs
        try:
            return system.ask({"update": "status"}).get("failed")
        except (OSError, ValueError):
            return None


def _found(name: str) -> bool | None:
    # whether a name is found as the box's address; None if the DNS finds no name at all
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


def _warning(key: str, title: str, said: str) -> dict[str, str]:
    return {"key": key, "title": title, "said": said}
