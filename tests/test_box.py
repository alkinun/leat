import base64
import datetime
import io
import json
import os
import shutil
import socket
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageColor

from leat.box import address, health, provision, screen, system

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
    for request in ([], {}, {"wifi": "reboot"}, {"timezone": "Mars/Olympus"}, {"timezone": 3},
                    {"wifi": "join", "name": ""}, {"wifi": "join", "name": "x" * 33},
                    {"wifi": "join", "name": "Office", "passphrase": "short"},
                    {"wifi": "join", "name": "Office", "passphrase": "x" * 64},
                    {"wifi": "join", "name": "Office", "passphrase": "pass\nword"},
                    {"power": "now"}, {"power": True}):  # fmt: skip
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
    assert (tmp_path / "leat-reset").read_text() == "LEAT_FACTORY_RESET=--factory-reset=yes\n"
    assert ran == [["systemctl", "--no-block", "reboot"]]


def test_system_update_status(tmp_path, monkeypatch):
    # the box's version, and the newest UKI an update installed past it, whatever its tries, or one
    # that failed, its tries spent; whether it looks for one now, and when it last did
    monkeypatch.setattr(system, "OS_RELEASE", tmp_path / "os-release")
    monkeypatch.setattr(system, "UKIS", tmp_path)
    (tmp_path / "os-release").write_text('ID=arch\nIMAGE_VERSION="2026.10.11.2241-fe75bd0"\n')
    (tmp_path / "leat_2026.10.11.2241-fe75bd0.efi").write_bytes(b"")
    shown = {"ActiveState": "inactive", "LastTriggerUSecRealtime": "@1791760000"}
    monkeypatch.setattr(system, "_shown", lambda unit, name: shown[name])
    assert system.handle({"update": "status"}) == {
        "version": "2026.10.11.2241-fe75bd0", "ready": None, "checking": False,
        "checked": 1791760000, "failed": None,
    }  # fmt: skip
    (tmp_path / "leat_2026.10.12.0300-abc1234+3-0.efi").write_bytes(b"")
    (tmp_path / "leat_2026.10.11.1900-0000000.efi").write_bytes(b"")  # older: none to boot into
    shown.update(ActiveState="activating", LastTriggerUSecRealtime="")
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
    assert ran == [["systemctl", "--no-block", "start", "systemd-sysupdate.service"]]


def test_system_wifi_files(tmp_path, monkeypatch):
    # a network joined is iwd's known network: a file of its name, or of its name's bytes in hex,
    # root's alone to read; open of no passphrase; forgotten, gone
    monkeypatch.setattr(system, "IWD", tmp_path / "iwd")
    assert system.handle({"wifi": "join", "name": "Office", "passphrase": "s3cret pass"}) == {}
    known = tmp_path / "iwd" / "Office.psk"
    assert known.read_text() == "[Security]\nPassphrase=s3cret pass\n"
    assert (
        known.stat().st_mode & 0o777 == 0o600 and (tmp_path / "iwd").stat().st_mode & 0o777 == 0o700
    )
    assert system.handle({"wifi": "join", "name": "Café Ü"}) == {}  # open, of a name in hex
    assert (tmp_path / "iwd" / f"={'Café Ü'.encode().hex()}.open").exists()
    system.handle({"wifi": "join", "name": "Office"})  # open now: the passphrase's file gone
    assert sorted(p.name for p in (tmp_path / "iwd").iterdir()) == sorted(
        ["Office.open", f"={'Café Ü'.encode().hex()}.open"])  # fmt: skip
    assert system.handle({"wifi": "forget", "name": "Office"}) == {}
    assert not (tmp_path / "iwd" / "Office.open").exists()


def test_system_scan(monkeypatch):
    # the networks in reach, the strongest first, as iwd orders them, of its objects over D-Bus
    def prop(kind, value):
        return {"type": kind, "data": value}

    objects = {
        "/net/connman/iwd/0/4": {"net.connman.iwd.Station": {"Scanning": prop("b", False)}},
        "/net/connman/iwd/0/4/a": {"net.connman.iwd.Network": {
            "Name": prop("s", "Office"), "Type": prop("s", "psk"), "Connected": prop("b", True)}},
        "/net/connman/iwd/0/4/b": {"net.connman.iwd.Network": {
            "Name": prop("s", "Guests"), "Type": prop("s", "open"), "Connected": prop("b", False)}},
    }  # fmt: skip

    def bus(*args, check=True):
        if args[-1] == "GetManagedObjects":
            return {"type": "a{oa{sa{sv}}}", "data": [objects]}
        if args[-1] == "GetOrderedNetworks":
            return {"type": "a(on)", "data": [[["/net/connman/iwd/0/4/a", -4500],
                                               ["/net/connman/iwd/0/4/b", -7600]]]}  # fmt: skip
        if args[-1] == "Scanning":
            return prop("b", False)
        return {}

    monkeypatch.setattr(system, "_bus", bus)
    assert system.handle({"wifi": "scan"}) == {
        "radio": True, "connected": "Office", "networks": [
            {"name": "Office", "secure": True, "strength": 4, "connected": True},
            {"name": "Guests", "secure": False, "strength": 1, "connected": False},
        ],
    }  # fmt: skip
    monkeypatch.setattr(system, "_objects", lambda: {})
    assert system.handle({"wifi": "scan"}) == {"radio": False, "networks": [], "connected": None}


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
        for said in ({"error": "there is no time zone 'Mars/Olympus'"}, {"networks": []}):
            connection, _ = server.accept()
            with connection:
                asked.append(json.loads(read(connection)))
                connection.sendall(json.dumps(said).encode())

    threading.Thread(target=answer, daemon=True).start()
    with pytest.raises(ValueError, match="no time zone"):
        system.ask({"timezone": "Mars/Olympus"}, path)
    assert system.ask({"wifi": "scan"}, path) == {"networks": []}
    assert asked == [{"timezone": "Mars/Olympus"}, {"wifi": "scan"}]
    server.close()
    with pytest.raises(OSError):
        system.ask({"wifi": "scan"}, str(tmp_path / "none.sock"))


def test_health(tmp_path, monkeypatch):
    # healthy once Leat answers, of a box provisioned, or of one not; a trial boot of an update
    # whose Leat never answers is restarted, and the check fails whatever
    assert health.healthy(tmp_path, wait=0)  # not provisioned: no Leat to run
    (tmp_path / "box.env").write_text("LEAT_NAME=k7x2qm.leatbox.net\n")
    answers = iter([None, None, {"empty": False}])
    monkeypatch.setattr(screen, "box", lambda: next(answers))
    assert health.healthy(tmp_path, wait=5, every=0)
    monkeypatch.setattr(screen, "box", lambda: None)
    assert not health.healthy(tmp_path, wait=0.05, every=0.01)
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
    # address takes some; the disk nearly full; the model out of reach; an update failed. Looked at
    # once a minute at most
    now = [0.0]
    found = {"addresses": [], "ipv4": "192.168.1.20", "free": 50e9, "failed": None}
    monkeypatch.setattr(health.socket, "getaddrinfo", lambda name, port, family: [
        (2, 1, 6, "", (a, port)) for a in found["addresses"]])  # fmt: skip
    monkeypatch.setattr(health, "ipv4", lambda: found["ipv4"])
    monkeypatch.setattr(
        health.shutil, "disk_usage", lambda path: SimpleNamespace(total=100e9, free=found["free"])
    )
    monkeypatch.setattr(health.system, "ask", lambda r: {"failed": found["failed"]})
    watch = health.Watch("k7x2qm.leatbox.net", tmp_path, clock=lambda: now[0])
    assert watch.warnings() == []  # unfound just now
    now[0] = 599
    assert watch.warnings() == []
    now[0] = 660
    assert [w["key"] for w in watch.warnings()] == ["name"]
    found.update(addresses=["192.168.1.20"], free=5e9, failed="2026.10.12.0300-abc1234")
    assert [w["key"] for w in watch.warnings()] == ["name"]  # looked at a minute ago: as then
    now[0] = 720
    warned = watch.warnings(lambda: "the engine is not reachable")
    assert [w["key"] for w in warned] == ["disk", "model", "update"]
    assert "5 GB" in warned[0]["said"] and "2026.10.12.0300-abc1234" in warned[2]["said"]
    found.update(addresses=["192.168.1.30"], free=50e9, failed=None)  # the name, of another
    now[0] = 780
    assert watch.warnings() == []  # unfound again, from now
    now[0] = 1380
    assert [w["key"] for w in watch.warnings()] == ["name"]


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
