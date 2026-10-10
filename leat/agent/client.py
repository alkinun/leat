"""The engine as the agent reaches it: leat serve's OpenAI chat API, over HTTP."""

import contextlib
import http.client
import json
import socket
import urllib.parse
from collections.abc import Iterator
from typing import Any


class EngineError(Exception):
    """The engine refused a request, failed it, or could not be reached: `status` is the HTTP
    status it answered with, if it answered."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class Completion:
    """A chat completion streaming: iterating it gives its chunks. close() ends it from any thread,
    as a client hanging up, which frees its slot in the engine."""

    def __init__(self, sock: socket.socket, response: http.client.HTTPResponse):
        self._sock, self._response = sock, response
        self._closed = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            for line in self._response:  # the events' lines, to the last or a close()
                if line.strip() == b"data: [DONE]":
                    return
                if line.startswith(b"data: "):
                    chunk = json.loads(line[6:])
                    if "error" in chunk:
                        raise EngineError(chunk["error"]["message"])
                    yield chunk
        except (OSError, http.client.HTTPException) as e:
            if not self._closed:
                raise EngineError(f"the engine's reply broke off: {e}") from e
            return
        finally:
            _close(self._sock, self._response)
        if not self._closed:
            raise EngineError("the engine's reply broke off")

    def close(self) -> None:
        # the socket's shutdown wakes the thread reading it, which may wait for a long prefill
        self._closed = True
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)


# seconds the engine may take to list its models: one that takes longer is as good as away, whereas
# a completion, which may wait for others, or a load, may take any time
LIST = 10


class Client:
    """leat serve at `url`, as http://127.0.0.1:8080, with its API key if it asks for one."""

    def __init__(self, url: str, key: str | None = None):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(f"the engine's URL must be http://host:port, not {url!r}")
        self.url, self._host, self._port = url.rstrip("/"), parts.hostname, parts.port or 80
        self._headers = {"Content-Type": "application/json"}
        if key:
            self._headers["Authorization"] = f"Bearer {key}"

    def models(self) -> list[dict[str, Any]]:
        """The models it has, each with its status: "loaded", "loading" or "unloaded"."""
        return self._json("GET", "/v1/models", timeout=LIST).get("data", [])

    def load(self, model: str) -> None:
        """Loads a model in place of the loaded one; returns once it is ready."""
        self._json("POST", "/v1/models/load", {"model": model})

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Each text's embedding, by the embedding model the engine has beside its model. Raises
        EngineError, of status 404 if it has none."""
        data = self._json("POST", "/v1/embeddings", {"input": texts}).get("data", [])
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]

    def complete(self, body: dict[str, Any], person: int | None = None) -> Completion:
        """A chat completion of `body`, streamed, for a person of the box's, by their id, if it is
        one's: the engine keeps the prefixes it caches of a person's prompts to theirs alone, so
        that no one can tell from how fast a reply starts what another asked."""
        user = {"user": f"person-{person}"} if person is not None else {}
        return Completion(
            *self._open("POST", "/v1/chat/completions", body | user | {"stream": True})
        )

    def reply(self, body: dict[str, Any], person: int | None = None) -> dict[str, Any]:
        """A chat completion's reply to `body`, whole, for a person as complete() is: its
        content, and its tool calls if any."""
        return whole(self.complete(body, person))

    def _json(
        self, method: str, path: str, body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        sock, response = self._open(method, path, body, timeout)
        try:
            return json.loads(response.read())
        except (OSError, ValueError) as e:
            raise EngineError(f"the engine's answer is unreadable: {e}") from e
        finally:
            _close(sock, response)

    def _open(
        self, method: str, path: str, body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[socket.socket, http.client.HTTPResponse]:  # fmt: skip
        # a request's socket and its response, a success's: any other raises EngineError. The
        # response keeps the socket once the connection lets it go, as it does when the engine
        # closes it after the response: leat serve's streams do.
        connection = http.client.HTTPConnection(self._host, self._port, timeout=timeout)
        try:
            data = None if body is None else json.dumps(body).encode()
            connection.request(method, path, data, self._headers)
            sock, response = connection.sock, connection.getresponse()
        except OSError as e:
            connection.close()
            raise EngineError(f"the engine at {self.url} is not reachable: {e}") from e
        if response.status == 200:
            return sock, response
        try:
            message = json.loads(response.read())["error"]["message"]
        except (OSError, ValueError, LookupError, TypeError):
            message = f"the engine answered HTTP {response.status}"
        finally:
            _close(sock, response)
        raise EngineError(message, response.status)


def whole(completion: Completion) -> dict[str, Any]:
    """A completion's reply, whole, to its end or a close(): its content, and its tool calls if
    any."""
    reply: dict[str, Any] = {"role": "assistant", "content": ""}
    for chunk in completion:
        delta = chunk["choices"][0]["delta"] if chunk["choices"] else {}
        reply["content"] += delta.get("content") or ""
        if calls := delta.get("tool_calls"):
            reply["tool_calls"] = [{k: c[k] for k in ("id", "type", "function")} for c in calls]
    return reply


def _close(sock: socket.socket, response: http.client.HTTPResponse) -> None:
    response.close()
    sock.close()
