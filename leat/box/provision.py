"""Provisioning: the files a box keeps of its registration, written when we build it.

    ssh leatbox leatbox-registry register | python -m leat.box.provision [--root /etc/leat]
    python -m leat.box.provision --take /boot/leat-box.json

A box takes its registration in on its first boot, by leat-provision.service, of a file beside its
image, on its boot partition, which any computer writes: the file is deleted once taken in, so
that its secrets are kept nowhere but the box's root.

Of the registration's JSON, it writes into the root:

    box.env          LEAT_ID, LEAT_NAME and LEAT_REGISTRY, which the box's services read
    registry-secret  the box's secret, by which leat.box.address sets its name's address
    acme-dns.json    its acme-dns account, by which Caddy has its certificate

Each secret is root's alone to read; the services are given them by systemd, as credentials.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from leat.box.system import write

ROOT = Path("/etc/leat")
# an ID as the registry gives one
_ID = re.compile(r"[2-9a-z]{6}")


def provision(box: dict[str, Any], root: Path = ROOT) -> None:
    """Writes a box's registration into `root`. Raises ValueError if it is not one."""
    try:
        id, name, secret, registry = box["id"], box["name"], box["secret"], box["registry"]
        account = {k: box["acme_dns"][k] for k in ("username", "password", "subdomain")}
        account["server_url"] = box["acme_dns"]["server_url"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"not a box's registration: {e!r} is missing") from None
    if not _ID.fullmatch(id) or not name.startswith(f"{id}."):
        raise ValueError(f"not a box's ID and name: {id!r}, {name!r}")
    root.mkdir(parents=True, exist_ok=True)
    write(root / "box.env", f"LEAT_ID={id}\nLEAT_NAME={name}\nLEAT_REGISTRY={registry}\n", 0o644)
    write(root / "registry-secret", secret, 0o600)
    write(root / "acme-dns.json", json.dumps(account, indent=2) + "\n", 0o600)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m leat.box.provision",
        description="Write a box's registration, as leatbox-registry register prints it, from "
        "stdin into the files the box's services read.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=f"by default {ROOT}")
    parser.add_argument(
        "--take", type=Path, help="a file of the registration to read, not stdin, then delete"
    )
    args = parser.parse_args(argv)
    try:
        box = json.loads(args.take.read_text()) if args.take else json.load(sys.stdin)
        provision(box, args.root)
    except (OSError, ValueError) as e:
        raise SystemExit(f"cannot provision: {e}") from e
    if args.take:
        args.take.unlink()
    print(f"provisioned {box['name']} in {args.root}")


if __name__ == "__main__":
    main()
