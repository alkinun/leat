import base64
import json
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from leat.box import address, provision

REGISTERED = {
    "id": "k7x2qm",
    "name": "k7x2qm.leatbox.net",
    "secret": "s3cret",
    "registry": "https://api.leatbox.net",
    "acme_dns": {"username": "u", "password": "p", "subdomain": "sub",
                 "server_url": "https://api.leatbox.net"},
}  # fmt: skip


def test_provision(tmp_path):
    root = tmp_path / "leat"
    provision.provision(REGISTERED, root)
    assert (root / "box.env").read_text() == (
        "LEAT_ID=k7x2qm\nLEAT_NAME=k7x2qm.leatbox.net\nLEAT_REGISTRY=https://api.leatbox.net\n"
    )
    assert (root / "registry-secret").read_text() == "s3cret"
    # as Caddy's acmedns module reads it
    assert json.loads((root / "acme-dns.json").read_text()) == REGISTERED["acme_dns"]
    modes = {p.name: p.stat().st_mode & 0o777 for p in root.iterdir()}
    assert modes == {"box.env": 0o644, "registry-secret": 0o600, "acme-dns.json": 0o600}
    provision.provision(REGISTERED | {"secret": "new"}, root)  # again: replaced whole
    assert (root / "registry-secret").read_text() == "new" and len(list(root.iterdir())) == 3
    for wrong in ({}, REGISTERED | {"id": "../etc"}, REGISTERED | {"name": "other.leatbox.net"},
                  REGISTERED | {"acme_dns": {}}):  # fmt: skip
        with pytest.raises(ValueError):
            provision.provision(wrong, tmp_path / "wrong")


class Registry(BaseHTTPRequestHandler):
    """The registry's PUT /address, which answers `answer`, and keeps what it was told."""

    told: list[tuple[str, dict]] = []
    answer: tuple[int, dict] = (200, {"name": "k7x2qm.leatbox.net"})

    def do_PUT(self) -> None:
        auth = base64.b64decode(self.headers["Authorization"].split()[1]).decode()
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Registry.told.append((auth, body))
        status, said = self.answer
        data = json.dumps(said).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def registry() -> Iterator[str]:
    Registry.told, Registry.answer = [], (200, {"name": "k7x2qm.leatbox.net"})
    with ThreadingHTTPServer(("127.0.0.1", 0), Registry) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{server.server_port}"
        server.shutdown()


def test_update(registry, tmp_path):
    # told once, then again only as the addresses change, or a day has gone by
    found = {"ipv4": "192.168.1.20", "ipv6": None}
    assert address.update(registry, "k7x2qm", "s3cret", tmp_path, found, now=1000)
    assert Registry.told == [("k7x2qm:s3cret", {"ipv4": "192.168.1.20"})]
    assert not address.update(registry, "k7x2qm", "s3cret", tmp_path, found, now=2000)
    moved = {"ipv4": "192.168.1.21", "ipv6": "fd00::21"}
    assert address.update(registry, "k7x2qm", "s3cret", tmp_path, moved, now=3000)
    assert Registry.told[-1][1] == {"ipv4": "192.168.1.21", "ipv6": "fd00::21"}
    assert address.update(registry, "k7x2qm", "s3cret", tmp_path, moved, now=3000 + address.AGAIN)
    assert len(Registry.told) == 3
    # refused, with the registry's reason, and told again at the next run
    Registry.answer = (400, {"error": "8.8.8.8 is not a private address of an office's network"})
    with pytest.raises(OSError, match=r"\(400\): 8.8.8.8 is not a private"):
        address.update(registry, "k7x2qm", "s3cret", tmp_path, {"ipv4": "8.8.8.8", "ipv6": None})
    with pytest.raises(OSError):
        address.tell("http://127.0.0.1:9", "k7x2qm", "s3cret", found)


def test_main(registry, tmp_path, monkeypatch, capsys):
    root = tmp_path / "leat"
    provision.provision(REGISTERED | {"registry": registry}, root)
    monkeypatch.setattr(address, "addresses", lambda: {"ipv4": "10.0.0.7", "ipv6": None})
    args = ["--root", str(root), "--state", str(tmp_path / "state")]
    address.main(args)
    assert capsys.readouterr().out == "k7x2qm's name points at 10.0.0.7\n"
    address.main(args)  # nothing to tell
    assert capsys.readouterr().out == "" and len(Registry.told) == 1
    # the secret systemd gives as a credential, over the root's
    credentials = tmp_path / "credentials"
    credentials.mkdir()
    (credentials / "registry-secret").write_text("given\n")
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(credentials))
    assert address.settings(root) == ("k7x2qm", registry, "given")
    Registry.answer = (401, {"error": "no box has that ID and secret"})
    monkeypatch.setattr(address, "addresses", lambda: {"ipv4": "10.0.0.8", "ipv6": None})
    with pytest.raises(SystemExit):
        address.main(args)
    assert "no box has that ID and secret" in capsys.readouterr().err


def test_ula(monkeypatch):
    # the first stable unique local IPv6 address of the IPv4 address's link: not a temporary one, a
    # global one, or a VPN's, as Tailscale's
    links = [
        {"ifname": "tailscale0", "addr_info": [
            {"family": "inet", "local": "100.101.1.2"},
            {"family": "inet6", "local": "fd7a:115c:a1e0::5"},
        ]},
        {"ifname": "eth0", "addr_info": [
            {"family": "inet", "local": "192.168.1.20"},
            {"family": "inet6", "local": "2a01:4f8::5"},
            {"family": "inet6", "local": "fd12::99", "temporary": True},
            {"family": "inet6", "local": "fd12::5"},
            {"family": "inet6", "local": "fd12::6"},
        ]},
    ]  # fmt: skip

    def run(*args, **kwargs):
        return SimpleNamespace(stdout=json.dumps(links))

    monkeypatch.setattr(subprocess, "run", run)
    assert address._ula("192.168.1.20") == "fd12::5"
    assert address._ula("10.0.0.1") is None  # no link of it
    links[1]["addr_info"] = links[1]["addr_info"][:3]
    assert address._ula("192.168.1.20") is None

    def missing(*args, **kwargs):
        raise FileNotFoundError("ip")

    monkeypatch.setattr(subprocess, "run", missing)
    assert address._ula("192.168.1.20") is None


def test_units():
    # the box's units run what this package has, as it reads its files
    box = Path(__file__).parent.parent / "box"
    proxy = (box / "systemd/leat-proxy.service").read_text()
    assert "LoadCredential=acme-dns.json:/etc/leat/acme-dns.json" in proxy
    assert "{$CREDENTIALS_DIRECTORY}/acme-dns.json" in (box / "Caddyfile").read_text()
    updater = (box / "systemd/leat-address.service").read_text()
    assert "LoadCredential=registry-secret:/etc/leat/registry-secret" in updater
    assert "-m leat.box.address" in updater
