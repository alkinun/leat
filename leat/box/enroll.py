"""Enrollment: a box registers itself, of its TPM, on its first boot, as it has no identity yet.

    python -m leat.box.enroll                  # by leat-enroll.service
    python -m leat.box.enroll --fingerprint    # its TPM's, which its maker sends us to allow

The registry allows the boxes made, by their TPMs' endorsement keys' fingerprints. The box gives it
its endorsement key and the name of an attestation key it makes in its TPM; the registry answers a
credential of a secret, which only that TPM takes out, of tpm2_activatecredential; the box gives
the secret back and is answered its registration, which leat.box.provision writes into /etc/leat,
as of a file a stick brings. Then the services of its identity start: its name, its front door and
leat agent. A box whose disk was erased registers itself again of its own name.
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from leat.box.provision import ROOT, provision

REGISTRY = "https://api.leatbox.net"
# the services of a box's identity, skipped at boot while it had none
STARTED = ["leat-address.service", "leat-proxy.service", "leat-agent.service"]
TIMEOUT = 30  # seconds the registry is waited for at most


class Refused(Exception):
    """The registry refuses the box: its TPM is not one of the boxes made, or its answer wrong."""


def fingerprint(pem: str) -> str:
    """An endorsement key's fingerprint, as the registry knows it: the SHA-256 of its public key's
    DER, in hex, of its PEM."""
    body = "".join(line for line in pem.splitlines() if line and not line.startswith("-----"))
    return hashlib.sha256(base64.b64decode(body)).hexdigest()


def enroll(registry: str = REGISTRY, root: Path = ROOT) -> dict[str, Any]:
    """Registers the box of its TPM, its registration written into `root`; returns it. Raises
    Refused, OSError of a registry out of reach, and subprocess.CalledProcessError of the TPM."""
    with tempfile.TemporaryDirectory(dir=os.environ.get("RUNTIME_DIRECTORY")) as work:
        files = Path(work)
        pem, name = keys(files)
        asked = _post(f"{registry}/enroll", {"ek": pem, "ak_name": name.hex()})
        (files / "credential").write_bytes(base64.b64decode(asked["credential"]))
        secret = activate(files)
        box = _post(
            f"{registry}/enroll/confirm", {"challenge": asked["challenge"], "secret": secret.hex()}
        )
    provision(box, root)
    return box


def keys(files: Path) -> tuple[str, bytes]:
    """The TPM's endorsement key, of the TCG's default template, as PEM, and the name of an
    attestation key made under it; their contexts kept in `files`, for activate()."""
    _tpm("tpm2_createek", "-c", files / "ek.ctx", "-G", "rsa", "-u", files / "ek.pem", "-f", "pem")
    _tpm("tpm2_createak", "-C", files / "ek.ctx", "-c", files / "ak.ctx", "-G", "rsa", "-g",
         "sha256", "-s", "rsassa", "-u", files / "ak.pub", "-n", files / "ak.name")  # fmt: skip
    return (files / "ek.pem").read_text(), (files / "ak.name").read_bytes()


def activate(files: Path) -> bytes:
    """The secret of the credential in `files`, as only this TPM takes it out, of its keys there:
    the endorsement key's use allowed by a policy session of the endorsement hierarchy's."""
    session = files / "session.ctx"
    _tpm("tpm2_startauthsession", "--policy-session", "-S", session)
    try:
        _tpm("tpm2_policysecret", "-S", session, "-c", "e")
        _tpm("tpm2_activatecredential", "-c", files / "ak.ctx", "-C", files / "ek.ctx", "-i",
             files / "credential", "-o", files / "secret", "-P", f"session:{session}")  # fmt: skip
    finally:
        subprocess.run(["tpm2_flushcontext", session], capture_output=True, check=False)
    return (files / "secret").read_bytes()


def _tpm(*args: Any) -> None:
    # a command of tpm2-tools'. Raises subprocess.CalledProcessError, its error said
    subprocess.run([str(a) for a in args], capture_output=True, check=True, timeout=60)


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    # the registry's answer of JSON. Raises Refused of its refusal, OSError if it is out of reach
    request = urllib.request.Request(url, json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})  # fmt: skip
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        if e.code in (400, 403):
            raise Refused(json.loads(e.read() or b"{}").get("error", str(e))) from None
        raise OSError(f"the registry answered {e.code}") from None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m leat.box.enroll", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--registry", default=REGISTRY, help=f"by default {REGISTRY}")
    parser.add_argument(
        "--fingerprint", action="store_true",
        help="print the TPM's endorsement key's fingerprint, and register nothing",
    )  # fmt: skip
    args = parser.parse_args(argv)
    if args.fingerprint:
        with tempfile.TemporaryDirectory() as work:
            print(fingerprint(keys(Path(work))[0]))
        return
    try:
        box = enroll(args.registry)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"the TPM refused: {e.stderr.decode(errors='replace').strip()}") from None
    except (Refused, OSError) as e:
        raise SystemExit(f"not registered: {e}") from None
    print(f"registered as {box['name']}", file=sys.stderr)
    subprocess.run(["systemctl", "start", "--no-block", *STARTED], check=False)


if __name__ == "__main__":
    main()
