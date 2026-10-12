import base64
import contextlib
import datetime
import hashlib
import io
import json
import os
import shutil
import socket
import sqlite3
import ssl
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageColor

from leat.box import address, backup, enroll, health, provision, screen, system, watch

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
READY = screen.State(NAME, "192.168.1.20", certified=True, answering=True,
                     model=("Qwen_Qwen3.6-35B-A3B-Q4_K_M", "loaded"), free=212e9,
                     version="2026.10.11")  # fmt: skip


def test_said():
    # what the box's screen says of each state: ready once its certificate holds and the app
    # answers, the address shown as its admins chose, and always before it is set up
    said = {
        "switched on": screen.State(NAME, None, booting=True),
        "no network": screen.State(NAME, None),
        "not provisioned": screen.State(None, "192.168.1.20"),
        "starting": screen.State(NAME, "192.168.1.20", answering=True),
        "set up": replace(READY, empty=True, show_address=False),
        "by code": replace(READY, empty=True, setup="K7X2QM"),
        "ready": READY,
        "hidden": replace(READY, show_address=False),
    }
    words = {k: screen.said(state) for k, state in said.items()}
    assert words["switched on"][0] == "Starting" and words["starting"][0] == "Starting"
    assert words["no network"][:2] == ("No network", "Plug the network cable into the box.")
    assert words["not provisioned"][0] == "Not set up"
    scan = "Scan the code, or open this address in a browser."
    assert words["set up"] == ("Set up your Leat", scan, f"https://{NAME}")
    typed = "Scan the code with a phone, or open this address and type the setup code."
    assert words["by code"] == ("Set up your Leat", typed, f"https://{NAME}")
    assert words["ready"][0] == "Ready" and words["ready"][2] == f"https://{NAME}"
    assert words["hidden"][2] is None
    assert all(w[2] is None for k, w in words.items() if k not in ("set up", "by code", "ready"))
    # the QR code holds the setup code while the box is to be set up, which the app takes
    url = f"https://{NAME}"
    assert screen.scanned(said["by code"], url) == f"{url}/#setup=K7X2QM"
    assert screen.scanned(replace(READY, setup="K7X2QM"), url) == url  # set up: none
    lines = screen.written(said["by code"], "##")
    assert lines[-1] == f"Setup code: {screen.BOLD}K7X 2QM{screen.PLAIN}"
    # the power button pressed: a recovery code, of any box set up, whatever its admins chose; or,
    # of one with none, whether to switch off
    recovering = replace(READY, show_address=False, recovery="P4WR8N", recovery_left=241)
    headline, sentence, shown = screen.said(recovering)
    assert headline == "Recovery" and "recovery code" in sentence and shown == url
    assert screen.scanned(recovering, url) == f"{url}/#recover=P4WR8N"
    assert screen.written(recovering, "##")[-3:] == [
        f"Recovery code, for 5 more minutes: {screen.BOLD}P4W R8N{screen.PLAIN}",
        "", "Press the power button again to switch the box off."]  # fmt: skip
    # what needs attention, written below what the box says
    warned = replace(READY, warnings=(("The box is nearly full", "5 GB are left."),))
    assert f"{screen.BOLD}The box is nearly full{screen.PLAIN}: 5 GB are left." in screen.written(
        warned, "##")  # fmt: skip
    asking = screen.said(screen.State(None, None, switching=True))
    assert asking == ("Switch off?", "Press the power button again to switch the box off.", None)


def test_power_buttons(tmp_path):
    # the power buttons' event devices, of the kernel's list of its inputs
    devices = tmp_path / "devices"
    devices.write_text(  # as a Dell laptop lists them: its power key on two
        'N: Name="Lid Switch"\nH: Handlers=event0 \n\n'
        'N: Name="Power Button"\nH: Handlers=kbd event1 \nB: KEY=8000 10000000000000 0\n\n'
        'N: Name="Sleep Button"\nH: Handlers=kbd event2 \nB: KEY=4000 0 0\n\n'
        'N: Name="Intel HID events"\nH: Handlers=kbd event4 rfkill \n'
        "B: KEY=81000300000000 5000004000 1e294000000020 0\n\n"
        'N: Name="PS/2 Generic Mouse"\nH: Handlers=event10 mouse0 \nB: KEY=70000 0 0 0 0\n'
    )
    assert screen.power_buttons(devices) == ["/dev/input/event1", "/dev/input/event4"]


def test_power_presses(tmp_path, monkeypatch):
    # a press asks the agent for a recovery code; of a box with none to give, it arms the button;
    # a press again then, or while the screen shows a code, switches the box off
    asked, off = [], []
    answers = iter([OSError("no agent"), None])

    def urlopen(request, timeout):
        asked.append(request.full_url)
        if (answer := next(answers)) is not None:
            raise answer
        return io.BytesIO(b"{}")

    monkeypatch.setattr(screen.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(screen.system, "ask", lambda request: off.append(request) or {})
    (tmp_path / "button").write_bytes(b"")
    power = screen.Power([str(tmp_path / "button")], threading.Event())
    power.press()  # no agent: armed
    assert power.switching() and power.woken.is_set() and off == []
    power.press()  # again: off
    assert off == [{"power": "off"}]
    power.armed, off[:] = 0.0, []
    power.press()  # the agent's recovery code asked for
    assert asked[-1] == f"{screen.AGENT}/api/box/recovery" and not power.switching()
    power.recovering = True  # as the screen shows it
    power.press()
    assert off == [{"power": "off"}]


def test_power_listens():
    # a press is the power key going down, of the kernel's input events; not its going up, nor
    # another key's
    read, write = os.pipe()
    pressed = []
    power = screen.Power.__new__(screen.Power)
    power._buttons = [open(read, "rb", buffering=0)]  # noqa: SIM115
    power.press = lambda: pressed.append(True)  # type: ignore[method-assign]
    listening = threading.Thread(target=power._listen, daemon=True)
    listening.start()
    for code, value in ((116, 0), (30, 1), (116, 1)):
        os.write(write, screen._EVENT.pack(0, 0, screen._EV_KEY, code, value))
    os.close(write)  # the button gone: no more listened to
    listening.join(timeout=5)
    assert not listening.is_alive() and pressed == [True]


def test_dimmed():
    # dimmed at night, of a box ready and quiet; never of one that has something to say
    night, day = datetime.datetime(2026, 10, 12, 23, 30), datetime.datetime(2026, 10, 12, 9, 0)
    assert screen.dimmed(READY, night) and screen.dimmed(READY, night.replace(hour=5))
    assert not screen.dimmed(READY, day) and not screen.dimmed(READY, night.replace(hour=6))
    for awake in (replace(READY, warnings=(("The box is nearly full", "5 GB are left."),)),
                  replace(READY, recovery="P4WR8N"), replace(READY, switching=True),
                  replace(READY, empty=True), screen.State(NAME, None)):  # fmt: skip
        assert not screen.dimmed(awake, night)


def test_footer():
    assert screen.footer(READY) == [
        "Qwen3.6 35B A3B · ready",
        "Network · 192.168.1.20",
        "Internet on",
        "212 GB free",
    ]
    loading = replace(READY, model=("gpt-oss-20b-MXFP4", "loading"), offline=True)
    assert screen.footer(loading)[:3] == ["gpt oss 20b · loading", "Network · 192.168.1.20",
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
    u = 8.0
    margin, mark = round(7 * u), round(3.4 * u)
    top = round(7 * u) + round(1.7 * u) - mark // 2
    assert painted.getpixel((margin + mark - 4, top + 4)) != page  # the mark's top right quarter
    assert painted.getpixel((1280 - margin - 150, 400)) == white  # the card, at the middle's right
    hidden = screen.paint(replace(READY, show_address=False), (1280, 800), now)
    assert hidden.getpixel((1280 - margin - 150, 400)) == page
    assert screen.paint(READY, (3840, 2160), now).size == (3840, 2160)  # any display's size


@pytest.mark.skipif(shutil.which("qrencode") is None, reason="needs qrencode")
def test_qr_card():
    # the QR code on its white card, a quiet zone of 2 modules about it, its corners rounded
    card = screen.qr_card("https://7j324w.leatbox.net", 400)
    assert card is not None and card.width == card.height <= 400
    page, white = ImageColor.getrgb(screen.PAGE), ImageColor.getrgb(screen.CARD)
    k = card.width // 29  # a version 3 code, 25 modules, and 2 of quiet zone each side
    assert card.getpixel((0, 0)) == page and card.getpixel((k, k)) == white
    assert card.getpixel((2 * k, 2 * k)) == (0, 0, 0)  # the finder pattern's corner


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


def test_system_refuses():
    # the helper does a few things alone, each checked
    for request in ([], {}, {"shell": "ls"}, {"timezone": "Mars/Olympus"}, {"timezone": 3},
                    {"power": "now"}, {"power": True}, {"reset": "yes"}):  # fmt: skip
        assert "error" in system.handle(request), request


def test_system_timezone(tmp_path, monkeypatch):
    # the time zone set, and kept on the state's partition; put back at boot once an update
    # replaced the system's, of a zone the system has not, and not of one it has
    ran = []
    monkeypatch.setattr(system.subprocess, "run", lambda args, **kw: ran.append(args))
    monkeypatch.setattr(system, "KEPT", tmp_path / "kept")
    assert system.handle({"timezone": "Europe/Istanbul"}) == {}
    assert ran == [["timedatectl", "set-timezone", "Europe/Istanbul"]]
    assert (tmp_path / "kept/timezone").read_text() == "Europe/Istanbul\n"
    ran.clear()
    monkeypatch.setattr(system, "zone", lambda: "Europe/Berlin")  # the new system's
    system.restore()
    assert ran == [["timedatectl", "set-timezone", "Europe/Istanbul"]]
    ran.clear()
    monkeypatch.setattr(system, "zone", lambda: "Europe/Istanbul")
    system.restore()
    assert ran == []
    (tmp_path / "kept/timezone").write_text("")  # as a power cut left one not yet on the disk
    system.restore()
    (tmp_path / "kept/timezone").unlink()
    system.restore()  # none kept: none put back
    assert ran == [] and not list((tmp_path / "kept").iterdir())


def test_system_power(tmp_path, monkeypatch):
    # the box switched off, or restarted, by systemd, the answer given first; or reset, of a flag on
    # the ESP that systemd-repart reads, and restarted
    ran = []
    monkeypatch.setattr(system.subprocess, "run", lambda args, **kw: ran.append(args))
    monkeypatch.setattr(system, "RESET", tmp_path / "leat-reset")
    assert system.handle({"power": "off"}) == {} and system.handle({"power": "restart"}) == {}
    assert ran == [["systemctl", "--no-block", "poweroff"], ["systemctl", "--no-block", "reboot"]]
    ran.clear()
    assert "error" in system.handle({"reset": "yes"}) and ran == []
    assert system.handle({"reset": True}) == {}
    assert (tmp_path / "leat-reset").read_text() == ""
    assert ran == [["systemctl", "--no-block", "reboot"]]


def test_system_update_status(tmp_path, monkeypatch):
    # the box's version, and the newest UKI an update installed past it, whatever its tries, or one
    # that failed, its tries spent; whether it looks for one now, and when it last did
    monkeypatch.setattr(system, "OS_RELEASE", tmp_path / "os-release")
    monkeypatch.setattr(system, "UKIS", tmp_path)
    (tmp_path / "os-release").write_text('ID=arch\nIMAGE_VERSION="2026.10.11.2241-fe75bd0"\n')
    (tmp_path / "leat_2026.10.11.2241-fe75bd0.efi").write_bytes(b"")
    shown = {"ActiveState": "inactive", "ExecMainExitTimestamp": "@1791760000"}
    monkeypatch.setattr(system, "_shown", lambda unit, name: shown[name])
    assert system.handle({"update": "status"}) == {
        "version": "2026.10.11.2241-fe75bd0", "ready": None, "checking": False,
        "checked": 1791760000, "failed": None,
    }  # fmt: skip
    (tmp_path / "leat_2026.10.12.0300-abc1234+3-0.efi").write_bytes(b"")
    (tmp_path / "leat_2026.10.11.1900-0000000.efi").write_bytes(b"")  # older: none to boot into
    shown.update(ActiveState="activating", ExecMainExitTimestamp="")
    status = system.update_status()
    assert (status["ready"], status["checking"], status["checked"], status["failed"]) == (
        "2026.10.12.0300-abc1234", True, None, None)  # fmt: skip
    (tmp_path / "leat_2026.10.12.0300-abc1234+3-0.efi").rename(
        tmp_path / "leat_2026.10.12.0300-abc1234+0-3.efi")  # fmt: skip
    status = system.update_status()  # its tries spent: failed, and none ready
    assert (status["ready"], status["failed"]) == (None, "2026.10.12.0300-abc1234")
    ran = []
    monkeypatch.setattr(system.subprocess, "run", lambda args, **kw: ran.append(args))
    assert system.handle({"update": "check"}) == {}
    assert ran == [["systemctl", "--no-block", "start", "systemd-sysupdate-update.service"]]


def test_system_diagnostics(monkeypatch):
    # the box's system, as text, each part under its title, a command that fails said so; the
    # logs of the system's services alone, never leat agent's, nor Caddy's of its requests
    ran = []

    def run(args, **kw):
        ran.append(args)
        if args[0] == "lsblk":
            raise FileNotFoundError("no lsblk")
        return SimpleNamespace(stdout=f"of {args[0]}", stderr="")

    monkeypatch.setattr(system.subprocess, "run", run)
    text = system.handle({"diagnostics": True})["text"]
    assert "## Version\n\nof cat" in text and "## Disks\n\n(no lsblk)" in text
    logs = next(a for a in ran if "--lines=400" in a)
    assert "--unit=leat-address" in logs and not any("leat-agent" in a or "leat-proxy" in a
                                                     for a in logs)  # fmt: skip
    certificate = next(a for a in ran if "--unit=leat-proxy" in a)
    assert certificate[-2:] == ["--grep", '"logger":"tls']


def test_system_ask(tmp_path):
    # the agent's question over the socket, sent whole, and the helper's answer, or its refusal
    # raised; a box without the helper, as off a box, an OSError
    path = str(tmp_path / "system.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen()
    asked = []

    def read(connection: socket.socket) -> bytes:
        return b"".join(iter(lambda: connection.recv(4096), b""))

    def answer():
        for said in ({"error": "there is no time zone 'Mars/Olympus'"}, {"fixed": None}):
            connection, _ = server.accept()
            with connection:
                asked.append(json.loads(read(connection)))
                connection.sendall(json.dumps(said).encode())

    threading.Thread(target=answer, daemon=True).start()
    with pytest.raises(ValueError, match="no time zone"):
        system.ask({"timezone": "Mars/Olympus"}, path)
    assert system.ask({"network": "status"}, path) == {"fixed": None}
    assert asked == [{"timezone": "Mars/Olympus"}, {"network": "status"}]
    server.close()
    with pytest.raises(OSError):
        system.ask({"network": "status"}, str(tmp_path / "none.sock"))


def test_health(tmp_path, monkeypatch):
    # healthy once the box's partitions are mounted and Leat answers, of a box provisioned, or of
    # one not; a trial boot of an update whose Leat never answers is restarted, and the check fails
    assert not health.healthy(tmp_path, wait=0, mounts=(tmp_path,))  # not a partition mounted
    assert health.healthy(tmp_path, wait=0, mounts=())  # not provisioned: no Leat to run
    (tmp_path / "box.env").write_text("LEAT_NAME=k7x2qm.leatbox.net\n")
    answers = iter([None, None, {"empty": False}])
    monkeypatch.setattr(screen, "box", lambda: next(answers))
    assert health.healthy(tmp_path, wait=5, every=0, mounts=())
    monkeypatch.setattr(screen, "box", lambda: None)
    assert not health.healthy(tmp_path, wait=0.05, every=0.01, mounts=())
    ran = []
    monkeypatch.setattr(health.subprocess, "run", lambda args, **kw: ran.append(args))
    monkeypatch.setattr(health, "healthy", lambda: False)
    monkeypatch.setattr(health, "TRIAL", tmp_path / "LoaderBootCountPath")
    with pytest.raises(SystemExit):
        health.main()
    assert ran == []  # a boot marked good: no restart
    (tmp_path / "LoaderBootCountPath").write_bytes(b"")
    with pytest.raises(SystemExit):
        health.main()
    assert ran == [["systemctl", "--no-block", "reboot"]]


def test_watch(tmp_path, monkeypatch):
    # what needs attention: the name unfound, said once it has gone so for ten minutes, as a new
    # address takes some, and never while the DNS finds no name at all, as without internet; the
    # internet unreached, as long; the certificate expiring; the disk nearly full; the model out of
    # reach; an update failed
    now = [0.0]
    dns = {"k7x2qm.leatbox.net": [], "leatbox.net": ["2.28.106.154"]}
    found = {"free": 50e9, "failed": None, "model": None, "reached": True, "left": 60}

    def getaddrinfo(name, port, family):
        if not dns[name]:
            raise socket.gaierror("not found")
        return [(2, 1, 6, "", (a, port)) for a in dns[name]]

    monkeypatch.setattr(watch.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(watch, "ipv4", lambda: "192.168.1.20")
    monkeypatch.setattr(
        watch.shutil, "disk_usage", lambda path: SimpleNamespace(total=100e9, free=found["free"])
    )
    monkeypatch.setattr(watch.system, "ask", lambda r: {"failed": found["failed"]})
    monkeypatch.setattr(watch, "_reached", lambda: found["reached"])
    monkeypatch.setattr(watch, "days_left", lambda name: found["left"])
    seen = watch.Watch("k7x2qm.leatbox.net", tmp_path, lambda: found["model"],
                       clock=lambda: now[0])  # fmt: skip
    assert seen.look() == [] and seen.warnings == []  # unfound just now
    now[0] = 599
    assert seen.look() == []
    now[0] = 600
    assert [w["key"] for w in seen.look()] == ["name"] and "FRITZ!Box" in seen.warnings[0]["said"]
    dns["k7x2qm.leatbox.net"] = ["192.168.1.20"]
    found.update(free=5e9, failed="2026.10.12.0300-abc1234", model="the engine is not reachable")
    warned = seen.look()
    assert [w["key"] for w in warned] == ["disk", "model", "update"]
    assert "5 GB" in warned[0]["said"] and "2026.10.12.0300-abc1234" in warned[2]["said"]
    found.update(free=50e9, failed=None, model=None)
    dns.update({"k7x2qm.leatbox.net": [], "leatbox.net": []})  # no internet: said of it alone
    found["reached"] = False
    now[0] = 2000
    assert seen.look() == []
    now[0] = 2600
    assert [w["key"] for w in seen.look()] == ["internet"]
    found["left"] = 14
    warned = seen.look()
    assert [w["key"] for w in warned] == ["internet", "certificate"]
    assert warned[1]["title"] == "Leat's certificate expires in 14 days"
    found["left"] = 1
    assert seen.look()[1]["title"] == "Leat's certificate expires within a day"
    found["left"] = 0
    assert seen.look()[1]["title"] == "Leat's certificate has expired"
    found.update(reached=True, left=89)  # back, and the name still unfound: from now
    dns["leatbox.net"] = ["2.28.106.154"]
    assert seen.look() == []
    now[0] = 3200
    assert [w["key"] for w in seen.look()] == ["name"]


def test_watchdays_left(tmp_path, monkeypatch):
    # the days left of the certificate the box serves its name with, as a browser trusting its CA
    # sees it: 0 once expired; none of one it does not trust, nor while it serves none
    name = "k7x2qm.leatbox.net"

    def serving(*dates: str) -> int | None:
        key, cert = tmp_path / "key.pem", tmp_path / "cert.pem"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt",
                        "ec_paramgen_curve:P-256", "-nodes", "-keyout", key, "-out", cert,
                        "-subj", f"/CN={name}", "-addext", f"subjectAltName=DNS:{name}",
                        *dates], check=True, capture_output=True)  # fmt: skip
        server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.load_cert_chain(cert, key)
        listening = socket.create_server(("127.0.0.1", 0))
        monkeypatch.setattr(watch, "HTTPS", listening.getsockname())

        def serve() -> None:
            connection = listening.accept()[0]
            with contextlib.suppress(OSError), server.wrap_socket(connection, server_side=True):
                pass

        threading.Thread(target=serve, daemon=True).start()
        try:
            return watch.days_left(name)
        finally:
            listening.close()

    trusted, default = tmp_path / "cert.pem", ssl.create_default_context
    monkeypatch.setattr(watch.ssl, "create_default_context", lambda: default(cafile=trusted))
    assert serving("-days", "10") == 9
    expires = ssl.cert_time_to_seconds(ssl._ssl._test_decode_cert(str(trusted))["notAfter"])
    with monkeypatch.context() as clock:  # its last day, three hours before it expires: valid still
        clock.setattr(watch.time, "time", lambda: expires - 3 * 3600)
        assert serving("-days", "10") == 1
    helped = subprocess.run(["openssl", "req", "-help"], capture_output=True, text=True).stderr
    if "-not_after" in helped:  # of OpenSSL 3.4 on, which sets a certificate's dates
        day = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)
        assert serving("-not_before", "20260101000000Z", "-not_after",
                       day.strftime("%Y%m%d%H%M%SZ")) == 0  # expired  # fmt: skip
    monkeypatch.setattr(watch.ssl, "create_default_context", default)
    assert serving("-days", "10") is None  # of a CA no browser trusts
    monkeypatch.setattr(watch, "HTTPS", ("127.0.0.1", 9))
    assert watch.days_left(name) is None  # none served


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
    # and each unit is enabled by the image's preset, as each the preset enables is one of these,
    # and systemd's own
    preset = box / "image/mkosi.extra/usr/lib/systemd/system-preset/10-leat.preset"
    enabled = {line.split()[1] for line in preset.read_text().splitlines()
               if line.startswith("enable leat-")}  # fmt: skip
    # but the helpers', of their sockets, and a backup's and a restore's, of a timer's or a request
    started = {"leat-system@.service", "leat-backup@.service", "leat-backup.service",
               "leat-restore.service"}  # fmt: skip
    units = {p.name for p in (box / "systemd").iterdir()} - started
    assert enabled == units
    # a dev image's updates of its own channel, as a box's never take its SSH
    postinst = (box / "image/mkosi.postinst.chroot").read_text()
    assert "https://leatbox.com/updates/dev/" in postinst and "/usr/lib/leat/channel" in postinst
    # updates looked for by the timer's own name, as a preset enables none of an alias's, as
    # systemd-sysupdate.timer is now
    assert "enable systemd-sysupdate-update.timer" in preset.read_text()
    # an update cut off removed before the next is tried, or none would be again
    dropped = box / "image/mkosi.extra/usr/lib/systemd/system/systemd-sysupdate-update.service.d"
    assert (
        "ExecStartPre=-/usr/lib/systemd/systemd-sysupdate vacuum"
        in (dropped / "leat.conf").read_text()
    )


def test_screen_look(tmp_path, monkeypatch):
    # the box as its screen looks at it: the agent's warnings in the screen's language, German of
    # a box of none yet
    monkeypatch.setattr(screen, "_name", lambda root: NAME)
    monkeypatch.setattr(screen, "_certified", lambda name: True)
    monkeypatch.setattr(screen, "_free", lambda: None)
    monkeypatch.setattr(system, "image_version", lambda: None)
    warned = {"key": "model", "title": "Leat can't reach its model",
              "said": "Chats get no answer until it is back. Restart the box."}  # fmt: skip
    monkeypatch.setattr(screen, "box", lambda: {"empty": True, "warnings": [warned]})
    monkeypatch.setattr(screen, "_address", lambda: "192.168.1.20")
    looked = screen.look()
    said = "Chats bekommen keine Antwort, bis es wieder da ist. Starten Sie die Box neu."
    assert looked.language == "de"
    assert looked.warnings == (("Leat erreicht sein Modell nicht", said),)


def test_screen_local():
    # the box's name on the office's network, as resolved publishes it: leat2 of a second box
    state = replace(READY, local="leat2")
    assert screen.written(state, "##")[-1] == "or type leat2.local"
    assert (
        screen.written(replace(state, language="de"), "##")[-1] == "oder geben Sie leat2.local ein"
    )


def test_system_network(tmp_path, monkeypatch):
    # a fixed address of the wired link's, checked, written as networkd reads it, a check of its
    # gateway asked for a while after; DHCP's again once the gateway answers not, said so; kept,
    # and put back at boot
    ran = []
    monkeypatch.setattr(system, "KEPT", tmp_path / "kept")
    monkeypatch.setattr(system, "FIXED", tmp_path / "run" / "10-leat-fixed.network")
    monkeypatch.setattr(system.subprocess, "run", lambda args, **kw: ran.append(args))
    for wrong in (
        {"address": "192.168.1.50", "gateway": "192.168.1.1"},  # of no prefix: a /32
        {"address": "192.168.1.50/24", "gateway": "10.0.0.1"},
        {"address": "192.168.1.255/24", "gateway": "192.168.1.1"},
        "static",
        5,
    ):
        assert "error" in system.handle({"network": wrong})
    fixed = {"address": "192.168.1.50/24", "gateway": "192.168.1.1", "dns": ["9.9.9.9"]}
    assert system.handle({"network": fixed}) == {}
    written = system.FIXED.read_text()
    assert "Address=192.168.1.50/24\nGateway=192.168.1.1\nDNS=9.9.9.9\n" in written
    assert ["networkctl", "reload"] in ran
    assert any(a[0] == "systemd-run" and "--check-network" in a for a in ran)
    assert system.handle({"network": "status"}) == {"fixed": fixed, "fell_back": False}
    monkeypatch.setattr(system, "_answers", lambda gateway: True)
    system.check_network()
    assert system.FIXED.exists()
    system.FIXED.unlink()  # as a boot clears /run
    system.restore()
    assert system.FIXED.exists()
    monkeypatch.setattr(system, "_answers", lambda gateway: False)
    system.check_network()
    assert not system.FIXED.exists()
    assert system.handle({"network": "status"}) == {"fixed": None, "fell_back": True}
    assert system.handle({"network": fixed}) == {}  # set again: no fall back said
    assert system.handle({"network": "dhcp"}) == {} and not system.FIXED.exists()
    assert system.handle({"network": "status"}) == {"fixed": None, "fell_back": False}


def test_enroll(tmp_path, monkeypatch):
    # the box registers itself of its TPM: its keys given, its credential's secret given back, and
    # its registration written as a stick's is; a refusal said, nothing written
    asked = []
    box = {"id": "k7x2qm", "name": "k7x2qm.leatbox.net", "secret": "s3cret",
           "registry": "https://api.leatbox.net",
           "acme_dns": {"username": "u", "password": "p", "subdomain": "d",
                        "server_url": "https://api.leatbox.net"}}  # fmt: skip

    def post(url, body):
        asked.append((url, body))
        if url.endswith("/enroll"):
            return {"challenge": "c1", "credential": base64.b64encode(b"credential").decode()}
        return box

    def activate(files):
        assert (files / "credential").read_bytes() == b"credential"
        return b"\x01\x02"

    monkeypatch.setattr(enroll, "keys", lambda files: ("PEM", b"\x00\x0bname"))
    monkeypatch.setattr(enroll, "activate", activate)
    monkeypatch.setattr(enroll, "_post", post)
    assert enroll.enroll("https://registry", tmp_path / "etc") == box
    assert asked == [
        ("https://registry/enroll", {"ek": "PEM", "ak_name": "000b6e616d65"}),
        ("https://registry/enroll/confirm", {"challenge": "c1", "secret": "0102"}),
    ]
    assert "LEAT_ID=k7x2qm" in (tmp_path / "etc/box.env").read_text()
    assert (tmp_path / "etc/registry-secret").read_text() == "s3cret"

    def refused(url, body):
        raise enroll.Refused("this box's TPM is not one of the boxes made")

    monkeypatch.setattr(enroll, "_post", refused)
    with pytest.raises(enroll.Refused):
        enroll.enroll("https://registry", tmp_path / "none")
    assert not (tmp_path / "none").exists()


def test_fingerprint():
    # an endorsement key's fingerprint, as the registry knows it: the SHA-256 of its DER
    der = bytes(range(200))
    pem = (
        "-----BEGIN PUBLIC KEY-----\n"
        + base64.encodebytes(der).decode()
        + "-----END PUBLIC KEY-----\n"
    )
    assert enroll.fingerprint(pem) == hashlib.sha256(der).hexdigest()


def test_backup_key():
    # a recovery key of 6 groups of 4, of letters none mistaken for another; one typed taken as it
    # was made, its case, spaces and dashes anyhow
    key = backup.new_key()
    assert len(key) == 29 and key.count("-") == 5 and set(key) <= set(backup.LETTERS + "-")
    assert backup.normal(key.lower().replace("-", " ")) == key
    assert backup.new_key() != key
    for wrong in ("", "ABCD", key + "A", key.replace(key[0], "0")):
        with pytest.raises(system.Refused):
            backup.normal(wrong)


LSBLK = {"blockdevices": [
    {"name": "nvme0n1", "tran": "nvme", "model": "Samsung", "children": [
        {"name": "nvme0n1p1", "uuid": "AAAA-AAAA", "fstype": "vfat", "label": None, "size": 1,
         "fsavail": 1}]},
    {"name": "sda", "tran": "usb", "model": "SanDisk Extreme ", "uuid": None, "fstype": None,
     "children": [
        {"name": "sda1", "uuid": "1234-ABCD", "fstype": "exfat", "label": "BACKUP",
         "size": 64e9, "fsavail": 60e9},
        {"name": "sda2", "uuid": "beef", "fstype": "swap", "label": None, "size": 1e9,
         "fsavail": None}]},
    {"name": "sdb", "tran": "usb", "model": "Stick", "uuid": "5678", "fstype": "ntfs",
     "label": "", "size": 32e9, "fsavail": None},
]}  # fmt: skip


def test_backup_disks(monkeypatch):
    # the USB disks' file systems the box mounts, of a disk partitioned or not; never its own
    # disk's, nor swap
    monkeypatch.setattr(backup.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout=json.dumps(LSBLK)))  # fmt: skip
    assert backup.disks() == [
        {"uuid": "1234-ABCD", "label": "BACKUP", "fstype": "exfat", "model": "SanDisk Extreme",
         "size": 64e9, "free": 60e9},
        {"uuid": "5678", "label": "", "fstype": "ntfs", "model": "Stick", "size": 32e9,
         "free": None},
    ]  # fmt: skip
    assert backup._checked({"disk": "5678"}) == {"disk": "5678", "label": "Stick"}
    with pytest.raises(system.Refused, match="not plugged"):
        backup._checked({"disk": "AAAA-AAAA"})


def test_backup_share():
    # a share's address as Windows writes it, or as Linux; its user and password as given
    assert backup._checked({"share": r"\\nas\backups\leat", "user": "office", "password": "p,w"}
                           ) == {"share": "//nas/backups/leat", "user": "office",
                                 "password": "p,w"}  # fmt: skip
    assert backup._checked({"share": "smb://192.168.1.5/b", "user": "", "password": ""})["share"] \
        == "//192.168.1.5/b"  # fmt: skip
    for wrong in ({"share": "nas", "user": "u", "password": "p"},
                  {"share": "//nas", "user": "u", "password": "p"},
                  {"share": "//nas/b", "user": "u", "password": "p\nusername=root"},
                  {"share": "//nas/b"}, {}):  # fmt: skip
        with pytest.raises(system.Refused):
            backup._checked(wrong)


def test_backup_refuses(tmp_path, monkeypatch):
    # its requests, each checked; none restores of a box that backs up nowhere
    monkeypatch.setattr(backup, "STATE", tmp_path)
    for request in ([], {}, {"backup": "everything"}, {"backup": {"disk": 3}},
                    {"restore": "latest"}, {"restore": {"snapshot": "../../etc"}},
                    {"restore": {"snapshot": "latest"}}):  # fmt: skip
        assert "error" in backup.handle(request), request


@pytest.fixture
def backups(tmp_path, monkeypatch):
    # the box's backups, of a target that is a folder, restic's commands noted and answered as a
    # test has them, and systemd's
    data, state, target = tmp_path / "data", tmp_path / "state", tmp_path / "target"
    for folder in (data / "workspace/people/1", state, target):
        folder.mkdir(parents=True)
    monkeypatch.setattr(backup, "DATA", data)
    monkeypatch.setattr(backup, "STATE", state)
    monkeypatch.setattr(backup, "STATUS", state / "status.json")
    monkeypatch.setattr(backup, "mounted", lambda t: contextlib.nullcontext(target))
    monkeypatch.setattr(backup, "_name", lambda: "k7x2qm.leatbox.net")
    told = SimpleNamespace(restic=[], ran=[], answers={}, data=data, state=state, target=target)

    def restic(repository, key, *args):
        told.restic.append((repository, key, args))
        answer = told.answers.get(args[0], "")
        return answer(args) if callable(answer) else answer

    monkeypatch.setattr(backup, "_restic", restic)
    monkeypatch.setattr(backup.subprocess, "run",
                        lambda args, **kw: told.ran.append(args) or SimpleNamespace(
                            returncode=0, stdout="inactive"))  # fmt: skip
    plugged = {"uuid": "1234-ABCD", "label": "BACKUP", "model": "SanDisk", "fstype": "exfat"}
    monkeypatch.setattr(backup, "disks", lambda: [plugged])
    return told


def test_backup_set_up(backups):
    # a target of no repository: one made, of a new key, answered once, kept with the target, and
    # backed up to at once; of one there, its key asked for, then kept, and nothing backed up
    # before tonight, as a new box restores first; of a wrong key, refused
    made = backup.handle({"backup": {"disk": "1234-ABCD"}})
    assert backup.normal(made["key"]) == made["key"]
    assert backups.restic == [(backups.target / "Leat backup", made["key"], ("init",))]
    assert (backups.state / "key").read_text() == made["key"]
    kept = json.loads((backups.state / "target.json").read_text())
    assert kept == {"disk": "1234-ABCD", "label": "BACKUP"}
    assert backups.ran[-1] == ["systemctl", "--no-block", "start", "leat-backup.service"]
    assert oct((backups.state / "key").stat().st_mode & 0o777) == "0o600"
    (backups.target / "Leat backup").mkdir()
    (backups.target / "Leat backup/config").write_text("")
    backups.ran.clear()
    assert backup.handle({"backup": {"disk": "1234-ABCD"}}) == {"found": True}
    assert backup.handle({"backup": {"disk": "1234-ABCD", "key": made["key"].lower()}}) == {}
    assert backups.restic[-1][1:] == (made["key"], ("cat", "config")) and backups.ran == []

    def wrong(args):
        raise system.Refused("that recovery key is not this backup's")

    backups.answers["cat"] = wrong
    assert backup.handle({"backup": {"disk": "1234-ABCD", "key": backup.new_key()}}) == {
        "error": "that recovery key is not this backup's"}  # fmt: skip
    status = backup.handle({"backup": "status"})
    assert status["target"] == {"disk": "1234-ABCD", "label": "BACKUP"} and not status["running"]
    assert backup.handle({"backup": "off"}) == {}
    assert backup.handle({"backup": "status"})["target"] is None
    assert not (backups.state / "key").exists()


def test_backup_share_password(backups):
    # a share's password kept for the backups, never shown
    backup.handle({"backup": {"share": r"\\nas\b", "user": "office", "password": "secret"}})
    assert backup.handle({"backup": "status"})["target"] == {"share": "//nas/b", "user": "office"}


def test_backup_run(backups):
    # the databases copied consistently, as they are written to, and backed up beside the files,
    # tagged with the box's name; backups past keeping forgotten; its status kept, anyone's to
    # read; one that failed said, and of a box that backs up nowhere, none tried
    backup.run()
    assert backups.restic == [] and not backup.STATUS.exists()
    backup.handle({"backup": {"disk": "1234-ABCD"}})
    backups.restic.clear()
    db = sqlite3.connect(backups.data / "leat.db")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE t (x)")
    db.execute("INSERT INTO t VALUES (42)")
    db.commit()  # in its write-ahead log, not yet in the database's file
    (backups.data / "workspace/people/1/notes.txt").write_text("hello")
    copied = []

    def backed_up(args):
        made = backups.data / ".backup/leat.db"
        copied.append(sqlite3.connect(made).execute("SELECT x FROM t").fetchall())
        return '{"message_type":"status"}\n{"message_type":"summary","total_bytes_processed":9,' \
               '"data_added":3}\n'  # fmt: skip

    backups.answers["backup"] = backed_up
    backup.run()
    assert copied == [[(42,)]] and not (backups.data / ".backup").exists()
    (_, key, args), (_, _, forgot) = backups.restic
    assert args[:6] == ("backup", "--json", "--host", "leat-box", "--tag",
                        "box=k7x2qm.leatbox.net") and args[-1] == str(backups.data)  # fmt: skip
    assert "--exclude" in args and str(backups.data / "*.db") in args
    assert forgot[0] == "forget" and "--prune" in forgot
    status = json.loads(backup.STATUS.read_text())
    assert status["size"] == 9 and status["added"] == 3 and "error" not in status
    assert oct(backup.STATUS.stat().st_mode & 0o777) == "0o644"

    def failed(args):
        raise system.Refused("the backup's target is out of reach: no such device")

    backups.answers["backup"] = failed
    with pytest.raises(SystemExit):
        backup.run()
    assert json.loads(backup.STATUS.read_text())["error"].startswith("the backup's target")


def test_backup_restore(backups):
    # a backup restored apart, then swapped for the data, leat agent stopped meanwhile, and the
    # device that asked kept beside it; the data before deleted; a backup of no conversations
    # refused, the data as it was
    backup.handle({"backup": {"disk": "1234-ABCD"}})
    (backups.data / "leat.db").write_text("now")
    snapshot = {"id": "ab" * 32, "time": "2026-10-12T01:40:00Z", "tags": ["box=old.leatbox.net"]}
    backups.answers["snapshots"] = json.dumps([snapshot])

    def restored(args):
        made = Path(args[args.index("--target") + 1]) / backups.data.relative_to("/")
        (made / ".backup").mkdir(parents=True)
        (made / ".backup/leat.db").write_text("then")
        (made / "workspace").mkdir()

    backups.answers["restore"] = restored
    device = {"token": "f" * 64, "name": "Firefox", "person": "Alkın"}
    assert backup.handle({"restore": {"snapshot": "latest", "device": device}}) == {}
    assert backups.ran[-1] == ["systemctl", "--no-block", "start", "leat-restore.service"]
    backups.ran.clear()
    backup.restore()
    assert (backups.data / "leat.db").read_text() == "then"
    assert not (backups.data / ".backup").exists()
    assert json.loads((backups.data / "restored.json").read_text()) == {
        "box": "old.leatbox.net", "device": device}  # fmt: skip
    stopped = backups.ran.index(["systemctl", "stop", "leat-agent.service"])
    assert backups.ran[stopped + 2] == ["systemctl", "start", "leat-agent.service"]
    assert not (backups.state / "before").exists() and not (backups.state / "asked.json").exists()
    assert "ok" in json.loads((backups.state / "restore.json").read_text())
    backups.answers["restore"] = lambda args: None  # nothing of Leat's
    backup.handle({"restore": {"snapshot": "latest", "device": device}})
    with pytest.raises(SystemExit):
        backup.restore()
    assert (backups.data / "leat.db").read_text() == "then"
    assert "conversations" in backup.handle({"backup": "status"})["restore"]["error"]


def test_watch_backups(tmp_path, monkeypatch):
    # a box that backs up: told of a backup that failed, or of none for three days
    monkeypatch.setattr(watch.backup, "STATUS", status := tmp_path / "status.json")
    assert watch._unsaved() is None
    status.write_text(json.dumps({"time": 1, "error": "out of reach"}))
    assert watch._unsaved() == "The last backup failed"
    status.write_text(json.dumps({"time": 1, "ok": time.time() - 86400}))
    assert watch._unsaved() is None
    status.write_text(json.dumps({"time": 1, "ok": time.time() - 3 * 86400}))
    assert watch._unsaved() == "No backup for 3 days"
