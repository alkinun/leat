"""Reading what an image shows: a scanned page, or a photo of a receipt, transcribed by the model
leat serve has loaded, if it sees images, as text the index keeps and searches.
"""

import base64
import threading
import time

from leat.agent.client import Client, EngineError

CHECK = 10  # seconds the engine's word on whether its model sees images holds
WORDS = 2000  # tokens of an image's text at most
# what the model is asked of an image
TRANSCRIBE = """\
Transcribe every word of this image as it is written, in reading order, as markdown: a heading as \
a heading, a table as a markdown table, numbers and dates exactly as they are. If it holds no \
text, say in a line what it shows. Write nothing else."""


class Transcriber:
    """Transcribes images by the model `engine` serves, if it sees images."""

    def __init__(self, engine: Client):
        self.engine = engine
        self._checked, self._sees = 0.0, False
        self._lock = threading.Lock()

    def available(self) -> bool:
        """Whether the engine's loaded model sees images, as it said CHECK seconds ago at most."""
        with self._lock:
            if time.monotonic() - self._checked > CHECK:
                try:
                    models = self.engine.models()
                except EngineError:
                    models = []
                self._sees = any(m.get("status") == "loaded" and m.get("vision") for m in models)
                self._checked = time.monotonic()
            return self._sees

    def recheck(self) -> None:
        """Has the engine asked again, at the next available(), as a model was loaded."""
        with self._lock:
            self._checked = 0.0

    def __call__(self, image: bytes, kind: str) -> str:
        """An image's text, of its bytes and their type, as "image/png". Raises EngineError."""
        url = f"data:{kind};base64,{base64.b64encode(image).decode()}"
        parts = [{"type": "image_url", "image_url": {"url": url}},
                 {"type": "text", "text": TRANSCRIBE}]  # fmt: skip
        body = {
            "messages": [{"role": "user", "content": parts}],
            "max_tokens": WORDS,
            "temperature": 0,
            "reasoning_effort": "none",
        }
        return self.engine.reply(body)["content"].strip()
