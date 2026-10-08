"""The household: its people, the devices each uses, and the pairing of a new device.

The first to open the app on a new box names themselves and is the household's owner, and all the
box held before is theirs. Any device after asks to join, and shows a code; the owner lets it in,
as a person of the household, known or new, once their own device shows the same code, as a
Bluetooth pairing has it. A device keeps a secret, sent as a cookie, of which the box keeps the
hash alone, so that a copy of its state lets no one in.
"""

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from leat.agent.agent import Agent

WAIT = 600  # seconds a request to join waits for the owner
ASKING = 5  # requests waiting at most, so that no one fills the owner's page with them
NAME = 40  # characters of a person's name at most
SEEN = 300  # seconds between the times a device is noted as seen


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


class Household:
    """The household of an agent's state: its people, their devices, and the requests to join."""

    def __init__(self, agent: "Agent"):
        self.agent, self.store = agent, agent.store
        self._requests: dict[str, _Request] = {}
        self._lock = threading.Lock()

    def empty(self) -> bool:
        """Whether no one is the household's yet, before its owner names themselves."""
        return not self.store.people()

    def setup(self, name: str, device: str) -> str:
        """Makes its first person, the owner, of a name, on a device; returns the device's secret.
        Raises ValueError if the household has an owner already, or the name is no name."""
        name = _checked(name)
        with self._lock:
            if not self.empty():
                raise ValueError("this home has its people already: ask to join")
            person = self.store.add_person(name)
            self._link_telegram(person["id"])
            return self._pair(person["id"], device)

    def device(self, secret: str | None) -> dict[str, Any] | None:
        """The device that keeps a secret, with its person's name and whether they are the owner,
        if it is paired."""
        if not secret or (device := self.store.device(_hash(secret))) is None:
            return None
        if time.time() - device["seen"] > SEEN:
            self.store.seen(device["id"])
        person = self.store.person(device["person"]) or {}
        return device | {"name": person.get("name"), "owner": bool(person.get("owner"))}

    def ask(self, name: str, device: str) -> dict[str, Any]:
        """A device's request to join, as who it says it is: its id, to ask after, and the code
        it shows, which the owner's device shows too. Raises ValueError if too many wait."""
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
        """The secret a request to join takes once the owner let it in, which it takes once; None
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

    def allow(self, id: str, person: int | None = None, name: str | None = None) -> dict:
        """Lets a request's device in, as a person of the household, or as a new one of a name,
        theirs by default. Raises LookupError if there is no such request or person."""
        with self._lock:
            self._expire()
            if (request := self._requests.get(id)) is None or request.secret is not None:
                raise LookupError("no such device asks to join")
            if person is None:
                person = self.store.add_person(_checked(name or request.name))["id"]
            elif self.store.person(person) is None:
                raise LookupError(f"there is no person {person}")
            request.secret = self._pair(person, request.device)
        self._publish()
        return self.store.person(person) or {}

    def refuse(self, id: str) -> None:
        """Turns down a request to join."""
        with self._lock:
            self._requests.pop(id, None)
        self._publish()

    def set_child(self, id: int, child: bool) -> None:
        """Says whether a person is a child, whose conversations begun after keep to a child's
        rules. Raises LookupError if there is no such person but the owner."""
        if not self.store.set_child(id, child):
            raise LookupError(f"there is no person {id} but the owner")
        self._publish()

    def unpair(self, id: int) -> None:
        """Unpairs a device, which must ask to join again. Raises LookupError if there is none."""
        if not self.store.remove_device(id):
            raise LookupError(f"there is no device {id}")
        self._publish()

    def remove(self, id: int) -> None:
        """Removes a person who is not the owner, with all that is theirs. Raises LookupError if
        there is no such person, ValueError of the owner."""
        if (person := self.store.person(id)) is None:
            raise LookupError(f"there is no person {id}")
        if person["owner"]:
            raise ValueError("the owner cannot be removed")
        for c in self.store.conversations(id):
            self.agent.delete(c["id"], id)
        self.store.remove_person(id)
        self._unlink_telegram(id)
        self._publish()

    def state(self) -> dict[str, Any]:
        """What the owner's app shows: the people, their devices, and the requests to join."""
        with self._lock:
            self._expire()
            keys = ("id", "name", "device", "code", "asked")
            waiting = [r for r in self._requests.values() if r.secret is None]
            requests = [{k: getattr(r, k) for k in keys} for r in waiting]
        devices = self.store.devices()
        people = [p | {"devices": [d for d in devices if d["person"] == p["id"]]}
                  for p in self.store.people()]  # fmt: skip
        return {"type": "household", "people": people, "requests": requests}

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

    def _link_telegram(self, owner: int) -> None:
        # the people Telegram let in before the household had any are its owner, as was all else
        from leat.agent.channels import telegram  # which imports the agent

        if settings := self.store.setting(telegram.KEY):
            allowed = {k: v | {"person": v.get("person") or owner}
                       for k, v in settings.get("allowed", {}).items()}  # fmt: skip
            self.store.set_setting(telegram.KEY, settings | {"allowed": allowed})

    def _unlink_telegram(self, person: int) -> None:
        # turns out the people Telegram let in as a person removed, who must ask again
        from leat.agent.channels import telegram  # which imports the agent

        if settings := self.store.setting(telegram.KEY):
            allowed = {k: v for k, v in settings.get("allowed", {}).items()
                       if v.get("person") != person}  # fmt: skip
            self.store.set_setting(telegram.KEY, settings | {"allowed": allowed})

    def _publish(self) -> None:
        self.agent.events.publish(self.state() | {"to": "owner"})


def _checked(name: str) -> str:
    # a person's name, its spaces made one; raises ValueError if it is none, or too long
    if not (name := " ".join(name.split())):
        raise ValueError("a person needs a name")
    if len(name) > NAME:
        raise ValueError(f"a name is {NAME} characters at most")
    return name


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()
