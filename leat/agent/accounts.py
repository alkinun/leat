"""The accounts: the box's people, the devices each uses, and the pairing of a new device.

The first to open the app on a new box names themselves and is its first admin, and all the box held
before is theirs. On a box, setting it up takes the code its screen shows, which its QR code holds,
so that whoever sets it up stands before it, not anyone on the office's network who opens it first;
a few wrong codes, and the screen shows another.

Any device after asks to join, and shows a code, which every device paired shows too, as a Bluetooth
pairing has it: a person lets in their own new device, from one of theirs; an admin lets in someone
new, or someone whose every device is gone, as one lost is unpaired. So no admin joins as another
who has a device, and sees what is theirs.

An office whose every admin's device is gone gets back in at the box: its power button pressed, its
screen shows a recovery code for five minutes, which its QR code holds too, and makes whoever gives
it, on a device, a new admin; a few wrong, and it holds no more. The admins' apps tell them, each
of whom may cancel it.

An admin looks after the box and its people: lets them in, renames and removes them, makes them
admins or not; there may be several, and there is one always. No admin sees another's own
conversations or files. A device keeps a secret, sent as a cookie, of which the box keeps the hash
alone, so that a copy of its state lets no one in. A computer's network drive is a device of its
person's too, of a password made in Settings and shown once, which opens the drive alone.
"""

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from leat.agent.agent import Refused

if TYPE_CHECKING:
    from leat.agent.agent import Agent

WAIT = 600  # seconds a request to join waits to be let in
ASKING = 5  # requests waiting at most, so that no one fills the apps' pages with them
NAME = 40  # characters of a person's name at most
SEEN = 300  # seconds between the times a device is noted as seen
CODE = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # a setup code's letters, none mistaken for another
TRIES = 5  # wrong setup or recovery codes before the screen shows another, or none
RECOVERING = 300  # seconds a recovery code holds


@dataclass
class _Request:
    # a device's request to join: who it says it is, what it is, and the code it shows; once let
    # in, the secret it is to take
    id: str
    name: str
    device: str
    code: str
    asked: float = field(default_factory=time.time)
    secret: str | None = None


class Accounts:
    """The accounts of an agent's state: its people, their devices, and the requests to join."""

    def __init__(self, agent: "Agent", locked: bool = False):
        self.agent, self.store = agent, agent.store
        self.locked = locked  # whether setting up takes the code the box's screen shows
        self.code = _setup_code()
        self._wrong = 0  # wrong codes given for this one
        self._recovery: tuple[str, float] | None = None  # a recovery code, and until when it holds
        self._unrecovered = 0  # wrong recovery codes given for it
        self._requests: dict[str, _Request] = {}
        self._lock = threading.Lock()

    def empty(self) -> bool:
        """Whether the box has no one yet, before its first person names themselves."""
        return not self.store.people()

    def setup(self, name: str, device: str, code: str | None = None) -> str:
        """Makes its first person, its first admin, of a name, on a device, of the code the box's
        screen shows if it is locked; returns the device's secret. Raises ValueError if the box
        has people already, or the name is no name; Refused of a wrong code."""
        name = _checked(name)
        with self._lock:
            if not self.empty():
                raise ValueError("this Leat is set up already: ask to join")
            given = _normal(code or "")
            if self.locked and not secrets.compare_digest(given, self.code):
                self._wrong += 1
                if self._wrong < TRIES:
                    raise Refused("that is not the setup code the box's screen shows")
                self.code, self._wrong = _setup_code(), 0
                raise Refused("that is not the setup code: the box's screen shows a new one now")
            person = self.store.add_person(name)
            self.agent.settle()  # the files that were no one's, now theirs
            return self._pair(person["id"], device)

    def begin_recovery(self) -> str:
        """A new recovery code, as the box's power button asks, which holds for five minutes."""
        with self._lock:
            self._recovery, self._unrecovered = (_setup_code(), time.time() + RECOVERING), 0
            code = self._recovery[0]
        self._publish()
        return code

    def recovery(self) -> str | None:
        """The recovery code that holds now, if one does."""
        return self.recovering()[0]

    def recovering(self) -> tuple[str | None, int]:
        """The recovery code that holds now, if one does, and the seconds it holds for yet."""
        with self._lock:
            if self._recovery and (left := self._recovery[1] - time.time()) > 0:
                return self._recovery[0], int(left)
            self._recovery = None
            return None, 0

    def cancel_recovery(self) -> bool:
        """Voids the recovery code, as an admin may; whether one held."""
        with self._lock:
            held, self._recovery = self._recovery is not None, None
        self._publish()
        return held

    def recover(self, code: str, name: str, device: str) -> tuple[dict[str, Any], str]:
        """Makes a new admin of a name, on a device, of the recovery code the box's screen
        shows, which then holds no more; returns them and the device's secret. Raises Refused of a
        wrong code, or none holding, ValueError of no name."""
        name = _checked(name)
        holding = self.recovery()
        with self._lock:
            given = _normal(code)
            if holding is None:
                raise Refused("no recovery code holds now: press the box's power button for one")
            if not secrets.compare_digest(given, holding):
                self._unrecovered += 1
                if self._unrecovered >= TRIES:
                    self._recovery = None
                    raise Refused("that is not the recovery code, and it holds no more: press the "
                                  "box's power button for another")  # fmt: skip
                raise Refused("that is not the recovery code the box's screen shows")
            self._recovery = None
            person = self.store.add_person(name, admin=True)
            secret = self._pair(person["id"], device)
        self._publish()
        return person, secret

    def device(self, secret: str | None, kind: str = "app") -> dict[str, Any] | None:
        """The device that keeps a secret, of a kind, an app's or a network drive's, with its
        person's name and whether they are an admin, if it is paired."""
        if not secret or (device := self.store.device(_hash(secret))) is None:
            return None
        if device["kind"] != kind:  # a drive's password no app's cookie, nor the other way
            return None
        if time.time() - device["seen"] > SEEN:
            self.store.seen(device["id"])
        person = self.store.person(device["person"]) or {}
        return device | {"name": person.get("name"), "admin": bool(person.get("admin"))}

    def ask(self, name: str, device: str) -> dict[str, Any]:
        """A device's request to join, as who it says it is: its id, to ask after, and the code
        it shows, which the devices paired show too. Raises ValueError if too many wait."""
        name = _checked(name)
        with self._lock:
            self._expire()
            if len(self._requests) >= ASKING:
                raise ValueError("too many devices ask to join now: try again in a few minutes")
            request = _Request(
                secrets.token_hex(8), name, device[:60], f"{secrets.randbelow(10**6):06}"
            )
            self._requests[request.id] = request
        self._publish()
        return {"id": request.id, "code": request.code}

    def answer(self, id: str) -> str | None:
        """The secret a request to join takes once it was let in, which it takes once; None
        while it waits. Raises LookupError if it was turned down, or waited too long."""
        with self._lock:
            self._expire()
            if (request := self._requests.get(id)) is None:
                raise LookupError("the request was turned down, or waited too long: ask again")
            if request.secret is None:
                return None
            del self._requests[id]
        self._publish()
        return request.secret

    def allow(self, id: str, by: dict[str, Any], person: int | None = None,
              name: str | None = None) -> dict[str, Any]:  # fmt: skip
        """Lets a request's device in, as a device's person allows it: as their own; or, of an
        admin, as someone new of a name, theirs by default, or as someone whose every device is
        gone. Returns the person, and the device, as it calls itself. Raises LookupError if there
        is no such request or person, Refused if it is not the device's to allow."""
        with self._lock:
            self._expire()
            if (request := self._requests.get(id)) is None or request.secret is not None:
                raise LookupError("no such device asks to join")
            if person is None:
                if not by["admin"]:
                    raise Refused("only an admin may let someone new in")
                person = self.store.add_person(_checked(name or request.name))["id"]
            elif person != by["person"]:
                if self.store.person(person) is None:
                    raise LookupError(f"there is no person {person}")
                if not by["admin"]:
                    raise Refused("a device joins as you alone, of yours")
                if any(d["person"] == person and d["kind"] == "app" for d in self.store.devices()):
                    raise Refused("a device joins as someone with a device of theirs from one of "
                                  "theirs alone")  # fmt: skip
            request.secret = self._pair(person, request.device)
        self._publish()
        return (self.store.person(person) or {}) | {"device": request.device}

    def refuse(self, id: str) -> None:
        """Turns down a request to join."""
        with self._lock:
            self._requests.pop(id, None)
        self._publish()

    def unpair(self, id: int) -> None:
        """Unpairs a device, which must ask to join again. Raises LookupError if there is none."""
        if not self.store.remove_device(id):
            raise LookupError(f"there is no device {id}")
        self._publish()

    def rename(self, id: int, name: str) -> None:
        """Renames a person. Raises LookupError if there is none, ValueError of no name."""
        if not self.store.rename(id, _checked(name)):
            raise LookupError(f"there is no person {id}")
        self._publish()

    def set_admin(self, id: int, admin: bool) -> None:
        """Makes a person an admin, or no longer one. Raises LookupError if there is no such
        person, ValueError of the last admin."""
        self.store.set_admin(id, admin)
        self._publish()

    def remove(self, id: int, heir: int) -> None:
        """Removes a person, with all that is theirs, as store.remove_person: the projects they
        shared with chosen people becoming their first member's, those shared with everyone their
        heir's. Raises LookupError if there is no such person, ValueError of the heir themselves
        or of the last admin."""
        if self.store.person(id) is None:
            raise LookupError(f"there is no person {id}")
        if id == heir:
            raise ValueError("you cannot remove yourself: ask another admin")
        for c in self.store.conversations(id):
            self.agent.delete(c["id"], id)
        for project in self.store.projects(id):  # their own, with any others' conversations in them
            if project["person"] == id and not project["shared"] and not project["members"]:
                self.agent.delete_project(project["id"], id)
        self.store.remove_person(id, heir)
        self.agent.clear(id)  # and their own files
        self.agent.projects_changed()  # those they shared, now another's
        self._publish()

    @staticmethod
    def me(device: dict[str, Any]) -> dict[str, Any]:
        """A device's person, as the app is told who it is: their id, name, whether they are an
        admin, and the device's id."""
        return {k: device[k] for k in ("person", "name", "admin")} | {"device": device["id"]}

    def state(self, device: dict[str, Any]) -> dict[str, Any]:
        """What a device's app shows of the accounts: its person, and the requests to join, which
        any of the box's people may let in as their own; and the people, each whether an admin,
        each with their devices, of an admin's, of anyone's but their own of the others'."""
        with self._lock:
            self._expire()
            keys = ("id", "name", "device", "code", "asked")
            waiting = [r for r in self._requests.values() if r.secret is None]
            requests = [{k: getattr(r, k) for k in keys} for r in waiting]
        devices = self.store.devices()
        people = [
            {"id": p["id"], "name": p["name"], "admin": bool(p["admin"]),
             "devices": [d for d in devices if d["person"] == p["id"]
                         and (device["admin"] or p["id"] == device["person"])]}
            for p in self.store.people()
        ]  # fmt: skip
        state: dict[str, Any] = {"type": "accounts", "me": self.me(device), "people": people,
                                 "requests": requests}  # fmt: skip
        if device["admin"]:  # and the seconds a recovery code holds for yet, if one does
            code, left = self.recovering()
            state["recovering"] = left if code else None
        return state

    def drive(self, person: int) -> str:
        """A password of a person's own, by which a computer opens their files and projects as a
        network drive, which their devices list as one of theirs, to unpair; returns it, which
        the box keeps the hash of alone, as it is shown once."""
        words = "abcdefghijkmnpqrstuvwxyz23456789"  # none mistaken for another
        secret = "-".join("".join(secrets.choice(words) for _ in range(4)) for _ in range(4))
        self.store.add_device(person, "Network drive", _hash(secret), kind="drive")
        self._publish()
        return secret

    def _pair(self, person: int, device: str) -> str:
        # pairs a device to a person; returns the secret it keeps
        secret = secrets.token_urlsafe(32)
        self.store.add_device(person, device[:60] or "A device", _hash(secret))
        return secret

    def _expire(self) -> None:
        # forgets the requests that waited too long
        old = [id for id, r in self._requests.items() if time.time() - r.asked > WAIT]
        for id in old:
            del self._requests[id]

    def _publish(self) -> None:
        # that the accounts changed, to every device, which each stream tells as its device sees
        self.agent.events.publish({"type": "accounts"})


def _normal(code: str) -> str:
    # a code as it was given, of its letters and digits alone, in capitals: "k7x-2qm" as K7X2QM
    return "".join(c for c in code.upper() if c.isalnum())


def _checked(name: str) -> str:
    # a person's name, its spaces made one; raises ValueError if it is none, or too long
    if not (name := " ".join(name.split())):
        raise ValueError("a person needs a name")
    if len(name) > NAME:
        raise ValueError(f"a name is {NAME} characters at most")
    return name


def _setup_code() -> str:
    # a new setup code, of 6 letters and digits: 32^6, a billion, a few tries each
    return "".join(secrets.choice(CODE) for _ in range(6))


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()
