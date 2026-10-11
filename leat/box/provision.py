"""Provisioning: the files a box keeps of its registration, written when we build it.

    ssh leatbox leatbox-registry register | python -m leat.box.provision [--root /etc/leat]

Of the registration's JSON, it writes into the root:

    box.env          LEAT_ID, LEAT_NAME and LEAT_REGISTRY, which the box's services read
    registry-secret  the box's secret, by which leat.box.address sets its name's address
    acme-dns.json    its acme-dns account, by which Caddy has its certificate

Each secret is root's alone to read; the services are given them by systemd, as credentials.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

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
    _write(root / "box.env", f"LEAT_ID={id}\nLEAT_NAME={name}\nLEAT_REGISTRY={registry}\n", 0o644)
    _write(root / "registry-secret", secret, 0o600)
    _write(root / "acme-dns.json", json.dumps(account, indent=2) + "\n", 0o600)


def _write(path: Path, text: str, mode: int) -> None:
    # a file whole, of its mode from the start, so that a secret is never another's to read
    temporary = path.with_name(f".{path.name}.new")
    temporary.unlink(missing_ok=True)
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "w") as f:
        f.write(text)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m leat.box.provision",
        description="Write a box's registration, as leatbox-registry register prints it, from "
        "stdin into the files the box's services read.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=f"by default {ROOT}")
    args = parser.parse_args(argv)
    try:
        box = json.load(sys.stdin)
        provision(box, args.root)
    except ValueError as e:
        raise SystemExit(f"cannot provision: {e}") from e
    print(f"provisioned {box['name']} in {args.root}")


if __name__ == "__main__":
    main()
