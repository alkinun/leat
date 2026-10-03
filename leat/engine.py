"""Generation: chunked prefill and token-by-token decode, each replayed from one compiled graph."""

import itertools
from collections.abc import Iterator
from pathlib import Path

from tinygrad import Tensor, TinyJit, UOp, dtypes

from leat.gguf import GGUF
from leat.model import Config, Transformer
from leat.sampler import sample
from leat.tokenizer import Tokenizer


class Engine:
    """A loaded model, ready to generate.

    The KV cache holds one sequence. A prompt that extends the previous one only prefills the new
    tokens, so multi-turn chat costs only the latest turn.
    """

    def __init__(self, path: str | Path, max_context: int = 4096, prefill_chunk: int = 512):
        self.gguf = gguf = GGUF.open(path)
        self.tokenizer = Tokenizer(gguf.metadata)
        self.config = Config.from_gguf(gguf.metadata)
        self.model = Transformer(self.config, gguf.load(), max_context)
        self.max_context, self.prefill_chunk = max_context, prefill_chunk
        self._pos = UOp.variable("start_pos", 0, max_context - 1)
        self._len = UOp.variable("chunk_len", 1, prefill_chunk)
        # TinyJit runs a function once as is, then captures it on the second call: capture on the
        # first, a fresh process's slow call for each graph. The device's random state must exist
        # by then, or the graph would create it anew every call.
        Tensor.rand(1).realize()  # on the default device, which holds the model
        self._prefill, self._decode = TinyJit(self._step), TinyJit(self._step)
        self._prefill.cnt = self._decode.cnt = 1
        self._cached: list[int] = []  # tokens whose keys and values are in the cache

    def generate(
        self, prompt: list[int], max_tokens: int, temperature: float = 0.0, ignore_eog: bool = False
    ) -> Iterator[int]:
        """Yields up to `max_tokens` ids; stops early at end of generation or the context limit."""
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
        if not 0 < len(prompt) < self.max_context:
            raise ValueError(
                f"prompt must have 1 to {self.max_context - 1} tokens, got {len(prompt)}"
            )
        temp = Tensor([temperature], dtype=dtypes.float32)
        # reuse the cached prefix, but always run at least the last prompt token to get its logits
        pairs = zip(prompt[:-1], self._cached, strict=False)
        pos = sum(1 for _ in itertools.takewhile(lambda p: p[0] == p[1], pairs))
        padded = Tensor([prompt + [0] * (self.max_context - len(prompt))], dtype=dtypes.int32)
        self._cached = prompt[:pos]  # what stays valid if prefill is interrupted
        while pos < len(prompt):
            n = min(self.prefill_chunk, len(prompt) - pos)
            start, length = self._pos.bind(pos), self._len.bind(n)
            token = self._prefill(padded[:, start : start + length], start, temp)
            pos += n
        self._cached = list(prompt)
        for remaining in reversed(range(max_tokens)):
            yield (t := int(token.item()))
            eog = t in self.tokenizer.eog_ids and not ignore_eog
            if not remaining or eog or pos >= self.max_context:
                return
            token = self._decode(token, self._pos.bind(pos), temp)
            self._cached.append(t)
            pos += 1

    def reset(self) -> None:
        """Forgets the cached prefix, so the next prompt is prefilled from scratch."""
        self._cached = []

    def _step(self, tokens: Tensor, start_pos: UOp, temperature: Tensor) -> Tensor:
        hidden = self.model(tokens, start_pos)
        return sample(self.model.logits(hidden[:, -1, :]), temperature).realize()
