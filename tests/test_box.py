import base64
import datetime
import io
import json
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageColor

from leat.box import address, provision, screen

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


def test_take(tmp_path, capsys):
    # a registration beside the image, taken in once and deleted; one that is not, left, and said
    taken = tmp_path / "leat-box.json"
    taken.write_text(json.dumps(REGISTERED))
    provision.main(["--root", str(tmp_path / "leat"), "--take", str(taken)])
    assert not taken.exists() and (tmp_path / "leat" / "registry-secret").read_text() == "s3cret"
    assert capsys.readouterr().out.startswith("provisioned k7x2qm.leatbox.net")
    taken.write_text("{}")
    with pytest.raises(SystemExit, match="not a box's registration"):
        provision.main(["--root", str(tmp_path / "other"), "--take", str(taken)])
    assert taken.exists() and not (tmp_path / "other").exists()
    with pytest.raises(SystemExit, match="cannot provision"):
        provision.main(["--take", str(tmp_path / "none.json")])


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


NAME = "k7x2qm.leatbox.net"
READY = screen.State(NAME, "192.168.1.20", certified=True, answering=True, link="Wired",
                     model=("Qwen_Qwen3.6-35B-A3B-Q4_K_M", "loaded"), free=212e9,
                     version="2026.10.11")  # fmt: skip


def test_said():
    # what the box's screen says of each state: ready once its certificate holds and the app
    # answers, the address shown as its owner chose, and always before it is set up
    said = {
        "switched on": screen.State(NAME, None, booting=True),
        "no network": screen.State(NAME, None),
        "not provisioned": screen.State(None, "192.168.1.20"),
        "starting": screen.State(NAME, "192.168.1.20", answering=True),
        "set up": replace(READY, empty=True, show_address=False),
        "ready": READY,
        "hidden": replace(READY, show_address=False),
    }
    words = {k: screen.said(state) for k, state in said.items()}
    assert words["switched on"][0] == "Starting" and words["starting"][0] == "Starting"
    assert words["no network"][:2] == ("No network", "Plug the network cable into the box.")
    assert words["not provisioned"][0] == "Not set up"
    scan = "Scan the code, or open this address in a browser."
    assert words["set up"] == ("Set up your Leat", scan, f"https://{NAME}")
    assert words["ready"][0] == "Ready" and words["ready"][2] == f"https://{NAME}"
    assert words["hidden"][2] is None
    assert all(w[2] is None for k, w in words.items() if k not in ("set up", "ready"))


def test_footer():
    assert screen.footer(READY) == [
        "Qwen3.6 35B A3B · ready",
        "Wired · 192.168.1.20",
        "Internet on",
        "212 GB free",
    ]
    loading = replace(READY, model=("gpt-oss-20b-MXFP4", "loading"), offline=True, link="Wi-Fi")
    assert screen.footer(loading)[:3] == ["gpt oss 20b · loading", "Wi-Fi · 192.168.1.20",
                                          "Internet off"]  # fmt: skip
    assert screen.footer(replace(READY, model=None))[0] == "No model"
    assert screen.footer(screen.State(None, None)) == []  # nothing known, nothing said
    for id, name in (("google_gemma-4-26B-A4B-it-Q4_K_M", "gemma 4 26B A4B"),
                     ("Meta-Llama-3.1-8B-Instruct-Q4_K_M", "Meta Llama 3.1 8B"),
                     ("occamy-1.0-Q4_K_M", "occamy 1.0")):  # fmt: skip
        assert screen._model(id) == name


def test_paint(monkeypatch):
    # the screen painted in the app's colours: Leat's mark at the top left, the QR code's white
    # card at the right of a box ready, none of one whose address its owner hid
    card = Image.new("RGB", (300, 300), screen.CARD)
    monkeypatch.setattr(screen, "qr_card", lambda url, side: card)
    now = datetime.datetime(2026, 10, 11, 14, 32)
    painted = screen.paint(replace(READY, office="Kanzlei Weber"), (1280, 800), now)
    assert painted.size == (1280, 800)
    page, white = ImageColor.getrgb(screen.PAGE), ImageColor.getrgb(screen.CARD)
    assert painted.getpixel((0, 0)) == page
    u, margin = 8.0, 64
    mark = round(4.4 * u)
    top = round(7 * u) + round(2.2 * u) - mark // 2
    assert painted.getpixel((margin + mark - 4, top + 4)) != page  # the mark's top right quarter
    assert painted.getpixel((1280 - margin - 150, 400)) == white  # the card, at the middle's right
    hidden = screen.paint(replace(READY, show_address=False), (1280, 800), now)
    assert hidden.getpixel((1280 - margin - 150, 400)) == page
    assert screen.paint(READY, (3840, 2160), now).size == (3840, 2160)  # any display's size


def test_logo():
    # Leat's mark, a pinwheel of four quarter circles, each about a corner, the middle empty
    mask = screen.logo(96)
    assert mask.size == (96, 96)
    for x, y in ((10, 30), (66, 10), (86, 66), (30, 86)):  # within each quarter circle
        assert mask.getpixel((x, y)) > 200
    for x, y in ((48, 48), (48, 20), (20, 48), (76, 48)):  # the channels between them
        assert mask.getpixel((x, y)) < 64


def test_written():
    # the screen written as text, of a display without a framebuffer, centred
    lines = screen.written(replace(READY, office="Kanzlei Weber"), "##\n##")
    plain = [line.replace(screen.BOLD, "").replace(screen.PLAIN, "") for line in lines]
    assert plain[0] == "Leat · Kanzlei Weber" and plain[2] == "Ready"
    assert (
        f"https://{NAME}" in plain and plain.count("##") == 2 and plain[-1] == "or type leat.local"
    )
    drawn = screen.draw([f"{screen.BOLD}Leat{screen.PLAIN}", "", "Ready"], 20, 9)
    assert drawn.startswith("\033[2J\033[H\033[?25l")
    assert (
        drawn.split("\033[?25l", 1)[1]
        == "\n" * 3 + f"        {screen.BOLD}Leat{screen.PLAIN}\n\n       Ready"
    )


def test_screen_take(monkeypatch, tmp_path):
    # the screen's console shown and the kernel's messages sent to the first; with a framebuffer,
    # the console's text kept off the display, and without one, the text's font set
    ran, ioctls = [], []
    monkeypatch.setattr(screen.subprocess, "run", lambda args, **kw: ran.append(args))
    monkeypatch.setattr(screen.fcntl, "ioctl", lambda fd, op, arg: ioctls.append((op, arg)))
    monkeypatch.setattr(screen, "open", lambda path, mode: io.StringIO(), raising=False)
    framebuffer = tmp_path / "fb0"
    framebuffer.touch()
    screen.take(framebuffer=str(framebuffer))
    assert ran == [["chvt", "2"]] and ioctls == [(0x541C, bytes([11, 1])), (0x4B3A, 1)]
    ran.clear(), ioctls.clear()
    screen.take(framebuffer=str(tmp_path / "none"))
    assert ran == [["chvt", "2"], ["setfont", "-C", "/dev/tty2", "ter-i28b"]]
    assert ioctls == [(0x541C, bytes([11, 1]))]


def test_screen_name(tmp_path):
    assert screen._name(tmp_path) is None
    provision.provision(REGISTERED, tmp_path)
    assert screen._name(tmp_path) == "k7x2qm.leatbox.net"
    assert screen.look(tmp_path / "none").name is None


def test_units():
    # the box's units run what this package has, as it reads its files
    box = Path(__file__).parent.parent / "box"
    proxy = (box / "systemd/leat-proxy.service").read_text()
    assert "LoadCredential=acme-dns.json:/etc/leat/acme-dns.json" in proxy
    assert "{$CREDENTIALS_DIRECTORY}/acme-dns.json" in (box / "Caddyfile").read_text()
    updater = (box / "systemd/leat-address.service").read_text()
    assert "LoadCredential=registry-secret:/etc/leat/registry-secret" in updater
    assert "-m leat.box.address" in updater
    shown = (box / "systemd/leat-screen.service").read_text()
    assert "TTYPath=/dev/tty2" in shown and "-m leat.box.screen --take" in shown
