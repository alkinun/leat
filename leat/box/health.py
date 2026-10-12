"""Whether the box came up as it should, once an update booted it: its partitions mounted, and Leat
answering, as the box's screen asks it, within a few minutes.

    python -m leat.box.health     # by leat-health.service, before boot-complete.target

systemd-boot tries an update's UKI three times, and systemd-bless-boot marks its boot good once
boot-complete.target is reached, which this is required by. A trial boot whose Leat never answers
is restarted, so that systemd-boot, its tries spent, boots the system before. A box not provisioned,
which runs no Leat of its own, is healthy of its partitions alone.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from leat.box import screen
from leat.box.provision import ROOT

WAIT = 300  # seconds Leat has to answer
MOUNTS = (ROOT, Path("/var"))  # the identity's and the state's partitions, which must be mounted
EVERY = 5  # seconds between asks
# set by systemd-boot of a boot whose UKI counts its tries: an update's, not yet marked good
TRIAL = Path("/sys/firmware/efi/efivars/LoaderBootCountPath-4a67b082-0a4c-41cf-b6c7-440b29bb8c4f")


def healthy(root: Path = ROOT, wait: float = WAIT, every: float = EVERY,
            mounts: tuple[Path, ...] = MOUNTS) -> bool:  # fmt: skip
    """Whether the box's partitions are mounted, and Leat answers within `wait` seconds, or the box
    runs none, not provisioned."""
    if not all(os.path.ismount(m) for m in mounts):  # as an identity not there is none provisioned
        return False
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
