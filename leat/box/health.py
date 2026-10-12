"""The box's health: whether it came up as it should, once an update booted it, and what of it needs
attention, which its screen and its admins are told.

    python -m leat.box.health     # by leat-health.service, before boot-complete.target

systemd-boot tries an update's UKI three times, and systemd-bless-boot marks its boot good once
boot-complete.target is reached, which this is required by: once Leat answers, as the box's screen
asks it, within a few minutes. A trial boot whose Leat never answers is restarted, so that
systemd-boot, its tries spent, boots the system before. A box not provisioned, which runs no Leat of
its own, is healthy of that alone.

What needs attention, Watch says: the box's name not found on its office's network, as a router's
DNS rebinding protection hides it; its data's disk nearly full; its model out of reach; an update
that failed, the box back on the version before. Each a few words and a sentence of what to do.
"""

import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from leat.box import screen, system
from leat.box.address import ipv4
from leat.box.provision import ROOT

WAIT = 300  # seconds Leat has to answer
EVERY = 5  # seconds between asks
# set by systemd-boot of a boot whose UKI counts its tries: an update's, not yet marked good
TRIAL = Path("/sys/firmware/efi/efivars/LoaderBootCountPath-4a67b082-0a4c-41cf-b6c7-440b29bb8c4f")


LOOK = 60  # seconds between looks at what needs attention
UNFOUND = 600  # seconds the name goes unfound before it is said, as a new address takes minutes
FULL = 0.1  # of the data's disk left, or less, which is said
FRITZ = "Heimnetz › Netzwerk › Netzwerkeinstellungen › DNS-Rebind-Schutz"


class Watch:
    """What of a box needs attention, of its name and its data's folder, looked at once a minute at
    most, however often it is asked."""

    def __init__(self, name: str, data: Path, clock: Callable[[], float] = time.monotonic):
        self.name, self.data, self.clock = name, data, clock
        self._looked: float | None = None
        self._unfound: float | None = None  # since when the name has gone unfound
        self._warnings: list[dict[str, str]] = []

    def warnings(
        self, model_error: Callable[[], str | None] = lambda: None
    ) -> list[dict[str, str]]:
        """What needs attention, each its key, its title and a sentence of what to do; of the model,
        why it is out of reach, if it is, as asked when the box is looked at."""
        now = self.clock()
        if self._looked is not None and now - self._looked < LOOK:
            return self._warnings
        self._looked, found = now, []
        if self._found():
            self._unfound = None
        else:
            self._unfound = now if self._unfound is None else self._unfound
            if now - self._unfound >= UNFOUND:
                found.append(_warning("name", "Leat can't be found by its name",
                                      "The router hides addresses inside the office from the "
                                      "devices here. On a FRITZ!Box, add leatbox.net under "
                                      f"{FRITZ}."))  # fmt: skip
        usage = shutil.disk_usage(self.data)
        if usage.free <= usage.total * FULL:
            found.append(_warning("disk", "The box is nearly full",
                                  f"{usage.free / 1e9:.0f} GB are left. Delete files Leat no "
                                  "longer needs."))  # fmt: skip
        if model_error():
            found.append(
                _warning(
                    "model",
                    "Leat can't reach its model",
                    "Chats get no answer until it is back. Restart the box.",
                )
            )
        failed = self._failed()
        if failed:
            found.append(_warning("update", "An update failed",
                                  f"The box went back to the version before, as {failed} did not "
                                  "start. Leat works as it did."))  # fmt: skip
        self._warnings = found
        return found

    def _found(self) -> bool:
        # whether the box's name is found, by the DNS its office's network gives it, as its address
        try:
            found = {a[4][0] for a in socket.getaddrinfo(self.name, 443, socket.AF_INET)}
            return ipv4() in found
        except OSError:  # unfound, or the box of no address, which its screen says
            return False

    def _failed(self) -> str | None:
        # the version of an update that failed, if one did, as the box's helper sees its UKIs
        try:
            return system.ask({"update": "status"}).get("failed")
        except (OSError, ValueError):
            return None


def _warning(key: str, title: str, said: str) -> dict[str, str]:
    return {"key": key, "title": title, "said": said}


def healthy(root: Path = ROOT, wait: float = WAIT, every: float = EVERY) -> bool:
    """Whether Leat answers within `wait` seconds, or the box runs none, not provisioned."""
    if not (root / "box.env").exists():
        return True
    deadline = time.monotonic() + wait
    while screen.box() is None:
        if time.monotonic() > deadline:
            return False
        time.sleep(every)
    return True


def main() -> None:
    if healthy():
        return
    print(f"Leat did not answer in {WAIT} s", file=sys.stderr)
    if TRIAL.exists():  # an update's boot, restarted to be tried again, or given up on
        print("restarting, as the boot is an update's trial", file=sys.stderr)
        subprocess.run(["systemctl", "--no-block", "reboot"], check=False)
    sys.exit(1)


if __name__ == "__main__":
    main()
