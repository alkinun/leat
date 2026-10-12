"""The box's screen: what a display plugged into it shows, in place of a login, which no box has.

    python -m leat.box.screen          # by leat-screen.service, on tty2
    python -m leat.box.screen --take   # as root, before: the display the screen's alone

It says, in a few large words, whether the box is ready, and the address to open, with a QR code of
it for a phone: starting, without a network, not provisioned, ready for its first person to set it
up, or ready. Ready is said once a browser would find it so: the box's certificate, checked as a
browser checks it, of its name, and the app answering behind it. Above, Leat's mark, the office's
name and the time; below, the box's model, network, internet, room left and version; nothing of
anyone's work.

It is painted, in the app's colours and Inter, onto the kernel's framebuffer, or, where there is
none, written as text. It is drawn again as it changes, looked at every few seconds, and every
minute whatever, as the time does, should anything have written over it.

The box's power button is the screen's, logind leaving it be: pressed, the screen shows a recovery
code, of the agent's, which makes whoever gives it an admin, as an office whose every admin's device
is gone gets back in; or, of a box that has none to give, asks whether to switch off. Pressed again
then, the box switches off, cleanly.
"""

import argparse
import contextlib
import dataclasses
import datetime
import fcntl
import functools
import io
import json
import re
import select
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from leat.box import system
from leat.box.address import ipv4
from leat.box.provision import ROOT

EVERY = 3  # seconds between looks
AGAIN = 60  # seconds after which the screen is drawn again, unchanged
# seconds after the box is switched on in which it is starting, its network not said to be missing:
# a laptop's screen said "No network" for the 2 s before DHCP gave it an address
BOOTING = 45
AGENT = "http://127.0.0.1:8000"  # leat agent, behind Caddy
TTY = "/dev/tty2"  # the screen's virtual console: the boot's messages are on the first
FRAMEBUFFER = "/dev/fb0"
DATA = "/var/lib/leat"  # the office's data, of whose partition the room left is said
CONSOLE_FONT = "ter-i28b"  # of the text the screen is written as, without a framebuffer
FONTS = Path("/usr/share/fonts/inter")  # Inter, the app's own font but for Geist, which Arch lacks
# the app's colours, dark: its page, text, muted text and lines; and the QR code's card, white, as
# phones read dark codes on light best
PAGE, TEXT, MUTED, LINE, CARD = "#0a0a0a", "#ededed", "#8f8f8f", "#262626", "#ffffff"
WARNING = "#f5a524"  # of what needs attention
# the hours, of the box's time zone, the screen is dimmed, once nothing needs seeing
NIGHT = (22, 6)
DIM = 0.2  # its brightness then
BOLD, PLAIN = "\033[1;97m", "\033[0m"  # of the words that matter, written as text
_TIOCLINUX, _KMSG_REDIRECT = 0x541C, 11  # the ioctl that sends the kernel's messages to a console
_KDSETMODE, _KD_GRAPHICS = 0x4B3A, 1  # the ioctl that keeps the console's text off the display
_FBIOGET_VSCREENINFO = 0x4600
_EVENT = struct.Struct("llHHi")  # an input event, of a 64-bit kernel: its time, type, code, value
_EV_KEY, _KEY_POWER = 1, 116
ARMED = 10  # seconds a press of the power button waits for another, which switches the box off


@dataclass(frozen=True)
class State:
    """What the screen says: the box's name, if provisioned; its IPv4 address, if it has one;
    whether its certificate holds for its name; whether the app answers; whether the box is yet to
    be set up by its first person; whether it was switched on just now; and, of the agent, the
    office's name, whether to show the address, the model, its id and whether it is "loaded" or
    "loading", whether the box is kept from the internet, the code that sets it up, while it is
    to be, and a recovery code, while one holds; whether the power button was pressed just now,
    once, so that a press again switches the box off; and its link, "Wired" or "Wi-Fi", the bytes
    left on its data's partition, and its image's version."""

    name: str | None
    address: str | None
    certified: bool = False
    answering: bool = False
    empty: bool = False
    booting: bool = False
    office: str | None = None
    show_address: bool = True
    model: tuple[str, str] | None = None
    offline: bool = False
    link: str | None = None
    free: int | None = None
    version: str | None = None
    setup: str | None = None
    recovery: str | None = None
    recovery_left: int = 0  # seconds the recovery code holds for yet
    switching: bool = False
    # what needs attention: each its title, and what to do
    warnings: tuple[tuple[str, str], ...] = ()


def look(root: Path = ROOT) -> State:
    """The box's state, as it is now."""
    name, address = _name(root), _address()
    booting = time.clock_gettime(time.CLOCK_BOOTTIME) < BOOTING
    state = State(name, address, booting=booting, link=_link(), free=_free(), version=_version())
    if name is None or address is None:
        return state
    told = box()
    if told is None:
        return dataclasses.replace(state, certified=_certified(name))
    model = told.get("model")
    return dataclasses.replace(
        state, certified=_certified(name), answering=True, empty=bool(told.get("empty")),
        office=told.get("office"), show_address=bool(told.get("address", True)),
        model=(model["id"], model["status"]) if model else None, offline=bool(told.get("offline")),
        setup=told.get("setup"), recovery=told.get("recovery"),
        recovery_left=int(told.get("recovery_left") or 0),
        warnings=tuple((w["title"], w["said"]) for w in told.get("warnings", [])),
    )  # fmt: skip


def said(state: State) -> tuple[str, str, str | None]:
    """What the screen says of a state: a few words, a sentence, and the address to open, if it is
    to be shown."""
    off = "Press the power button again to switch the box off."
    if state.switching and not state.recovery:
        return "Switch off?", off, None
    if state.address is None and state.booting:
        return "Starting", "This takes a minute.", None
    if state.address is None:
        return "No network", "Plug the network cable into the box.", None
    if state.name is None:
        return "Not set up", "This box has no identity yet. Call Leat.", None
    if not (state.certified and state.answering):
        return "Starting", "This takes a minute.", None
    url, scan = f"https://{state.name}", "Scan the code, or open this address in a browser."
    if state.recovery:  # whatever its admins chose, as none may be there
        return (
            "Recovery",
            "Scan the code, or open this address and choose Use a recovery code: "
            "that device becomes an admin.",
            url,
        )
    if state.empty and state.setup:  # whatever its admins choose, as there are none yet
        typed = "Scan the code with a phone, or open this address and type the setup code."
        return "Set up your Leat", typed, url
    if state.empty:
        return "Set up your Leat", scan, url
    if not state.show_address:
        return "Ready", "Open Leat in a browser in the office.", None
    return "Ready", scan, url


def scanned(state: State, url: str) -> str:
    """What the QR code of an address holds: the address, and the setup code too while the box is
    to be set up, which the app takes, so that whoever scans it types none."""
    if state.recovery:
        return f"{url}/#recover={state.recovery}"
    return f"{url}/#setup={state.setup}" if state.empty and state.setup else url


def holds(state: State) -> str:
    """What the recovery code is said to be, and for how long it holds: "Recovery code, for 4 more
    minutes", by the minute begun."""
    minutes = max(1, -(-state.recovery_left // 60))
    return f"Recovery code, for {minutes} more minute{'s' if minutes > 1 else ''}"


def spaced(code: str) -> str:
    """A setup code as the screen shows it, in two halves: "K7X 2QM"."""
    return f"{code[:3]} {code[3:]}"


def dimmed(state: State, now: datetime.datetime) -> bool:
    """Whether the screen is dimmed: at night, when the box is ready and nothing of it needs
    attention, as an empty office's display need not light it, nor wear; anything else wakes it."""
    night = now.hour >= NIGHT[0] or now.hour < NIGHT[1]
    quiet = said(state)[0] == "Ready" and not state.warnings and not state.switching
    return night and quiet


def footer(state: State) -> list[str]:
    """The screen's last line, of the box: its model, its network, whether it reaches the internet,
    and the room left on its data's partition; each once it is known."""
    items = []
    if state.answering:
        if state.model is None:
            items.append("No model")
        else:
            model, status = state.model
            items.append(f"{_model(model)} · {'ready' if status == 'loaded' else status}")
    if state.address:
        items.append(f"{state.link or 'Network'} · {state.address}")
    if state.answering:
        items.append("Internet off" if state.offline else "Internet on")
    if state.free is not None:
        items.append(f"{state.free / 1e9:.0f} GB free")
    return items


def paint(state: State, size: tuple[int, int], now: datetime.datetime) -> Image.Image:
    """The screen, as a picture of `size`: the top, the middle, and the last line."""
    width, height = size
    u = min(height / 100, width / 160)  # the screen's measure: a hundredth of its height
    image = Image.new("RGB", size, PAGE)
    pen = ImageDraw.Draw(image)
    margin, words = round(7 * u), font(round(2.6 * u))

    # the top: Leat's mark and name, the office's, and the time
    mark, middle = round(3.4 * u), round(7 * u) + round(1.7 * u)
    image.paste(ImageColor.getrgb(TEXT), (margin, middle - mark // 2), logo(mark))
    x = margin + mark + round(1.4 * u)
    pen.text((x, middle), "Leat", font=font(round(2.6 * u), "SemiBold"), fill=TEXT, anchor="lm")
    if state.office:
        x += round(pen.textlength("Leat", font=font(round(2.6 * u), "SemiBold")))
        pen.text((x, middle), f"   {state.office}", font=words, fill=MUTED, anchor="lm")
    pen.text((width - margin, middle), now.strftime("%H:%M"), font=words, fill=MUTED, anchor="rm")

    # the middle: what the box says, and the address to open, with its QR code on a card
    headline, sentence, url = said(state)
    card = qr_card(scanned(state, url), round(min(40 * u, width * 0.28))) if url else None
    right = width - margin - (card.width + round(8 * u) if card else 0)
    lines: list[tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, str, float]] = [
        (headline, font(round(6.4 * u), "SemiBold"), TEXT, 2.4 * u),
        *((line, font(round(2.8 * u)), MUTED, 1.0 * u)
          for line in _wrapped(sentence, font(round(2.8 * u)), right - margin, pen)),
    ]  # fmt: skip
    advice = font(round(2.4 * u))
    for title, what in state.warnings[:2]:  # what needs attention, below what the box says
        lines[-1] = (*lines[-1][:3], 3.0 * u)
        lines += [(title, font(round(2.8 * u), "SemiBold"), WARNING, 0.6 * u),
                  *((line, advice, WARNING, 0.6 * u)
                    for line in _wrapped(what, advice, right - margin, pen))]  # fmt: skip
    if url:
        lines += [(url.removeprefix("https://"), font(round(3.2 * u), "Medium"), TEXT, 1.1 * u),
                  ("or type leat.local", font(round(2.4 * u)), MUTED, 0.0)]  # fmt: skip
        lines[-3] = (*lines[-3][:3], 4.0 * u)  # room before the address
        # and the code that recovers it, or sets it up, below
        shown = (holds(state), state.recovery) if state.recovery else ("Setup code", state.setup)
        if shown[1] and (state.recovery or state.empty):
            lines[-1] = (*lines[-1][:3], 3.0 * u)
            code = font(round(4.4 * u), "SemiBold")
            lines += [(shown[0], font(round(2.4 * u)), MUTED, 0.6 * u),
                      (spaced(shown[1]), code, TEXT, 0.0)]  # fmt: skip
        if state.recovery:  # and that a press again switches the box off
            lines[-1] = (*lines[-1][:3], 3.0 * u)
            lines.append(("Press the power button again to switch the box off.",
                          font(round(2.4 * u)), MUTED, 0.0))  # fmt: skip
    tall = sum(_height(pen, f) + gap for _, f, _, gap in lines)
    y = height / 2 - tall / 2
    for line, f, colour, gap in lines:
        pen.text((margin, y), line, font=f, fill=colour, anchor="lt")
        y += _height(pen, f) + gap
    if card:
        image.paste(card, (width - margin - card.width, round(height / 2 - card.height / 2)))

    # the last line: the box's model, network, internet and room; its version, right
    below = height - round(7 * u)
    pen.line((margin, below - round(4 * u), width - margin, below - round(4 * u)), fill=LINE)
    small, x = font(round(2.1 * u)), margin
    for item in footer(state):
        pen.text((x, below), item, font=small, fill=MUTED, anchor="lm")
        x += round(pen.textlength(item, font=small) + 3.4 * u)
    if state.version:
        pen.text((width - margin, below), state.version, font=small, fill=MUTED, anchor="rm")
    return image


def written(state: State, qr: str = "") -> list[str]:
    """The screen's lines, written as text, as on a display without a framebuffer, of a state and
    the QR code of its address, of block characters, if any."""
    headline, sentence, url = said(state)
    title = f"Leat · {state.office}" if state.office else "Leat"
    lines = [f"{BOLD}{title}{PLAIN}", "", f"{BOLD}{headline}{PLAIN}", "", sentence]
    for warned, what in state.warnings[:2]:
        lines += ["", f"{BOLD}{warned}{PLAIN}: {what}"]
    if url:
        lines += ["", url, "", *qr.splitlines(), "", "or type leat.local"]
    if url and state.recovery:
        lines += ["", f"{holds(state)}: {BOLD}{spaced(state.recovery)}{PLAIN}", "",
                  "Press the power button again to switch the box off."]  # fmt: skip
    elif url and state.empty and state.setup:
        lines += ["", f"Setup code: {BOLD}{spaced(state.setup)}{PLAIN}"]
    return lines


def draw(lines: list[str], width: int, height: int) -> str:
    """The lines, each centred, the block of them in the middle of the screen, which is cleared."""
    top = max(0, (height - len(lines)) // 2)
    longest = max((_shown(line) for line in lines), default=0)
    left = max(0, (width - longest) // 2)
    body = "\n".join(
        " " * max(left, (width - _shown(line)) // 2) + line if line else "" for line in lines
    )
    return "\033[2J\033[H\033[?25l" + "\n" * top + body


@functools.lru_cache(maxsize=8)
def logo(size: int) -> Image.Image:
    """Leat's mark, as the app's logo.svg draws it, of `size` pixels: a mask of a pinwheel of four
    quarter circles, each of a radius of 11 of the mark's 24, drawn large and made small, smooth."""
    big, k = size * 4, size * 4 / 24
    mask = Image.new("L", (big, big), 0)
    pen = ImageDraw.Draw(mask)
    for (x, y), start in (((0, 11), 270), ((13, 0), 0), ((24, 13), 90), ((11, 24), 180)):
        box = ((x - 11) * k, (y - 11) * k, (x + 11) * k, (y + 11) * k)
        pen.pieslice(box, start, start + 90, fill=255)
    return mask.resize((size, size), Image.Resampling.LANCZOS)


@functools.lru_cache(maxsize=4)
def qr_card(url: str, side: int) -> Image.Image | None:
    """A QR code of the address, its modules sharp, on a white card of about `side` pixels, its
    corners rounded, with a quiet zone of 2 modules around it: enough on a dark screen, where 4
    looked a thick border; or None without qrencode."""
    if shutil.which("qrencode") is None:
        return None
    try:
        png = subprocess.run(["qrencode", "-t", "PNG", "-s", "1", "-m", "0", "-o", "-", url],
                             capture_output=True, timeout=5, check=True).stdout  # fmt: skip
        code = Image.open(io.BytesIO(png)).convert("RGBA").convert("L")
    except (OSError, subprocess.SubprocessError):
        return None
    n, quiet = code.width, 2
    k = max(1, side // (n + 2 * quiet))  # each module's pixels
    whole = (n + 2 * quiet) * k
    card = Image.new("RGB", (whole, whole), PAGE)
    ImageDraw.Draw(card).rounded_rectangle((0, 0, whole - 1, whole - 1), radius=k, fill=CARD)
    card.paste(code.resize((n * k, n * k), Image.Resampling.NEAREST), (quiet * k, quiet * k))
    return card


@functools.lru_cache(maxsize=32)
def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Inter of a size and a weight, as "SemiBold"; Pillow's own font where there is no Inter."""
    for name in (f"Inter-{weight}.ttf", f"Inter-{weight}.otf", "InterVariable.ttf"):
        with contextlib.suppress(OSError):
            chosen = ImageFont.truetype(str(FONTS / name), size)
            if name.startswith("InterVariable"):
                with contextlib.suppress(OSError, ValueError):
                    chosen.set_variation_by_name(weight)
            return chosen
    return ImageFont.load_default(size)


class Framebuffer:
    """The display, of the kernel's framebuffer: its size, and pictures written to it whole."""

    def __init__(self, device: str = FRAMEBUFFER):
        self.device = device
        with open(device, "rb") as fb:  # the screen's: its width, height and bits a pixel
            info = fcntl.ioctl(fb, _FBIOGET_VSCREENINFO, bytes(160))
        width, height, _, _, _, _, bits = struct.unpack_from("7I", info)
        name = Path(device).name
        self.stride = int(Path(f"/sys/class/graphics/{name}/stride").read_text())
        if bits != 32:
            raise OSError(f"{device} has {bits} bits a pixel, not 32")
        self.size = (width, height)

    def show(self, image: Image.Image) -> None:
        """Writes a picture of the framebuffer's size to it, row by row if its rows are padded."""
        data, row = image.convert("RGB").tobytes("raw", "BGRX"), image.width * 4
        with open(self.device, "r+b", buffering=0) as fb:
            if self.stride == row:
                fb.write(data)
                return
            for y in range(image.height):
                fb.seek(y * self.stride)
                fb.write(data[y * row : (y + 1) * row])


def take(tty: str = TTY, framebuffer: str = FRAMEBUFFER, console_font: str = CONSOLE_FONT) -> None:
    """Takes the display, as root: shows the screen's console; keeps the console's text off it, as
    the screen is painted onto the framebuffer, or, without one, sets the text's font; and sends
    the kernel's messages to the first console, as the kernel writes them over whatever console
    the display shows, as it did over the QR code."""
    subprocess.run(["chvt", tty.removeprefix("/dev/tty")], check=False)
    graphical = Path(framebuffer).exists()
    if not graphical:
        subprocess.run(["setfont", "-C", tty, console_font], check=False, capture_output=True)
    with open(tty, "w") as console:
        fcntl.ioctl(console, _TIOCLINUX, bytes([_KMSG_REDIRECT, 1]))
        if graphical:
            fcntl.ioctl(console, _KDSETMODE, _KD_GRAPHICS)


def _wrapped(sentence: str, f: ImageFont.FreeTypeFont | ImageFont.ImageFont, room: float,
             pen: ImageDraw.ImageDraw) -> list[str]:  # fmt: skip
    # a sentence in lines no wider than `room`, broken between words
    lines, line = [], ""
    for word in sentence.split():
        tried = f"{line} {word}".strip()
        if line and pen.textlength(tried, font=f) > room:
            lines.append(line)
            tried = word
        line = tried
    return [*lines, line] if line else lines


def _height(pen: ImageDraw.ImageDraw, f: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> float:
    # the height a line of a font takes: its ascent and descent, whatever its letters
    left, top, right, bottom = pen.textbbox((0, 0), "Ag", font=f, anchor="lt")
    return bottom - top


def _model(id: str) -> str:
    # a model's id as people read it: "Qwen_Qwen3.6-35B-A3B-Q4_K_M" as "Qwen3.6 35B A3B"
    name = id.split("_", 1)[1] if "_" in id.split("-")[0] else id
    parts = name.split("-")
    while parts and (parts[-1][:1] in ("Q", "F") and any(c.isdigit() for c in parts[-1])
                     or parts[-1] in ("MXFP4", "BF16", "it", "Instruct")):  # fmt: skip
        parts.pop()
    return " ".join(parts) or id


def _shown(line: str) -> int:
    # the characters a line takes on the screen, its styles' codes but none
    return len(line.replace(BOLD, "").replace(PLAIN, ""))


def _name(root: Path) -> str | None:
    # the box's name, of its provisioned box.env
    with contextlib.suppress(OSError):
        for line in (root / "box.env").read_text().splitlines():
            if line.startswith("LEAT_NAME="):
                return line.split("=", 1)[1].strip() or None
    return None


def _address() -> str | None:
    # the IPv4 address the box reaches others from, or None without a network
    with contextlib.suppress(OSError):
        return ipv4()
    return None


def _link() -> str | None:
    # how the box is on its network: "Wi-Fi" or "Wired", of its default route's interface
    with contextlib.suppress(OSError, IndexError):
        for line in Path("/proc/net/route").read_text().splitlines()[1:]:
            fields = line.split()
            if fields[1] == "00000000":
                return "Wi-Fi" if Path(f"/sys/class/net/{fields[0]}/wireless").exists() else "Wired"
    return None


def _free() -> int | None:
    # the bytes left on the data's partition
    with contextlib.suppress(OSError):
        return shutil.disk_usage(DATA).free
    return None


def _version() -> str | None:
    # the box's image's version, as os-release says it
    with contextlib.suppress(OSError):
        for line in Path("/usr/lib/os-release").read_text().splitlines():
            if line.startswith("IMAGE_VERSION="):
                return line.split("=", 1)[1].strip('"') or None
    return None


def _certified(name: str) -> bool:
    # whether the box's own HTTPS, on this machine, has a certificate a browser takes for its name
    context = ssl.create_default_context()
    with (
        contextlib.suppress(OSError),
        socket.create_connection(("127.0.0.1", 443), timeout=3) as raw,
        context.wrap_socket(raw, server_hostname=name),
    ):
        return True
    return False


def box() -> dict | None:
    """The box as the agent tells its screen, or None if the agent does not answer."""
    with (
        contextlib.suppress(OSError, ValueError),
        urllib.request.urlopen(f"{AGENT}/api/box", timeout=3) as response,
    ):
        return json.loads(response.read())
    return None


def power_buttons(devices: Path = Path("/proc/bus/input/devices")) -> list[str]:
    """The event devices of the box's power buttons, as the kernel lists its inputs: each that has
    the power key, as logind takes them, the ACPI's "Power Button" or another, as a Dell laptop's
    "Intel HID events"."""
    found = []
    for block in devices.read_text().split("\n\n"):
        handlers, keys = re.search(r"H: Handlers=(.*)", block), re.search(r"B: KEY=(.*)", block)
        if handlers and keys and _has_key(keys[1], _KEY_POWER):
            found += [f"/dev/input/{h}" for h in handlers[1].split() if h.startswith("event")]
    return found


def _has_key(bitmap: str, key: int) -> bool:
    # whether a device's keys, as the kernel writes their bitmap, in words of 64 bits, the highest
    # first, have a key
    words = bitmap.split()[::-1]
    word, bit = divmod(key, 64)
    return word < len(words) and bool(int(words[word], 16) >> bit & 1)


class Power:
    """The box's power button, its presses heard: the first asks the agent for a recovery code,
    or, of a box that has none to give, arms the button; a press again, while the screen shows the
    code or the button is armed, switches the box off."""

    def __init__(self, buttons: list[str], woken: threading.Event):
        self.woken, self.armed, self.recovering = woken, 0.0, False
        self._buttons = [open(path, "rb", buffering=0) for path in buttons]  # noqa: SIM115
        threading.Thread(target=self._listen, daemon=True).start()

    def switching(self) -> bool:
        """Whether a press now switches the box off: of one just now, without a recovery code."""
        return time.monotonic() < self.armed

    def press(self) -> None:
        """A press of the button."""
        if self.recovering or self.switching():
            with contextlib.suppress(OSError, ValueError):
                system.ask({"power": "off"})
            return
        try:
            request = urllib.request.Request(f"{AGENT}/api/box/recovery", b"{}", method="POST")
            urllib.request.urlopen(request, timeout=3).close()
        except OSError:  # no agent, or a box not set up: none to recover, but the box to switch off
            self.armed = time.monotonic() + ARMED
        self.woken.set()

    def _listen(self) -> None:
        # each press, as the kernel tells it: a key's going down; a button that tells nothing more,
        # as one gone, no more listened to
        while self._buttons:
            ready, _, _ = select.select(self._buttons, [], [])
            for button in ready:
                if not (data := button.read(_EVENT.size * 64)):
                    self._buttons.remove(button)
                    button.close()
                    continue
                for at in range(0, len(data), _EVENT.size):
                    _, _, kind, code, value = _EVENT.unpack_from(data, at)
                    if kind == _EV_KEY and code == _KEY_POWER and value == 1:
                        self.press()


def main() -> None:
    try:
        display: Framebuffer | None = Framebuffer()
    except OSError:  # written as text, then
        display = None
    woken = threading.Event()  # by the power button, to look again at once
    power = None
    with contextlib.suppress(OSError):
        power = Power(power_buttons(), woken)
    shown: tuple | None = None
    drawn = 0.0
    while True:
        time.tzset()  # the box's time zone, as an admin may have changed it
        state, now = look(), datetime.datetime.now()
        if power is not None:
            power.recovering = state.recovery is not None
            state = dataclasses.replace(state, switching=power.switching())
        seen = (state, now.strftime("%H:%M"))
        if seen != shown or time.monotonic() - drawn > AGAIN:
            if display is not None:
                painted = paint(state, display.size, now)
                display.show(painted.point(lambda v: v * DIM) if dimmed(state, now) else painted)
            else:
                url = said(state)[2]
                lines = written(state, _text_qr(scanned(state, url)) if url else "")
                sys.stdout.write(draw(lines, *shutil.get_terminal_size((80, 25))))
                sys.stdout.flush()
            shown, drawn = seen, time.monotonic()
        woken.wait(EVERY)
        woken.clear()


@functools.lru_cache(maxsize=4)
def _text_qr(url: str) -> str:
    # a QR code of the address, of block characters, as qrencode writes one; or "" without it
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        drawn = subprocess.run(["qrencode", "-t", "UTF8", "-m", "2", url], capture_output=True,
                               text=True, timeout=5, check=True).stdout  # fmt: skip
        return drawn.rstrip("\n")
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m leat.box.screen", description=__doc__)
    parser.add_argument("--take", action="store_true", help="take the display, as root, and end")
    if parser.parse_args().take:
        take()
    else:
        with contextlib.suppress(KeyboardInterrupt):
            main()
