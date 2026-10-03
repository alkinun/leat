# leat

A minimal, fast LLM inference engine built on [tinygrad](https://github.com/tinygrad/tinygrad).

leat runs GGUF models with their weights kept in the quantized storage format. The goal is single-stream decode limited by memory bandwidth, not by the engine. It targets NVIDIA RTX 30-series GPUs first, then AMD Strix Halo.

> Status: on NVIDIA, hand-written kernels run decoding faster than llama.cpp and prompt processing at 0.8x its speed, and `leat serve` serves the OpenAI chat completions API. Every kernel is tested against the reference ops, plain tinygrad code that runs on any device; `LEAT_KERNELS=ref` runs everything that way.

## Quickstart

```bash
uv sync
DEV=NV uv run leat run Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

As a server, for any OpenAI client:

```bash
DEV=NV uv run leat serve Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
messages = [{"role": "user", "content": "Why is the sky blue?"}]
for chunk in client.chat.completions.create(model="llama", messages=messages, stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

From Python:

```python
from leat import ChatTemplate, Engine

engine = Engine("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", max_context=4096)
chat = ChatTemplate(engine.gguf.metadata, engine.tokenizer)
prompt = chat.encode([{"role": "user", "content": "Why is the sky blue?"}])
print(engine.tokenizer.decode(list(engine.generate(prompt, max_tokens=256))))
```

`leat bench` measures speed and `leat perplexity` measures quality, either on a text file or against logits saved by llama.cpp's `llama-perplexity --kl-divergence-base`.

## Supported

- Architecture: `llama` with the `llama-bpe` tokenizer (Llama 3.x)
- Storage types: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K, which covers Q4_K_M, Q5_K_M, Q6_K and Q8_0 files
- Devices: any tinygrad backend; developed on NVIDIA with `DEV=NV`
- Server: `/v1/chat/completions`, whole or streamed, with stop strings, seeds and tool calls, and `/v1/models`. Sampling is greedy or by temperature; requests for `top_p`, penalties, `logprobs` or several choices are refused. Completions run one at a time.
- Prefix caching: the KV cache keeps `--slots` sequences. A conversation continues in its slot, and a prompt that shares a prefix with any slot, such as a system prompt, starts from a copy of it.

## Measurements

RTX 3090, Meta-Llama-3.1-8B-Instruct Q4_K_M, one sequence.

| | pp512 (tok/s) | tg128 (tok/s) | tg128 after 8192 tokens |
|---|---:|---:|---:|
| llama.cpp b11372, CUDA | 5417 | 147.6 | 125.0 |
| leat | 4384 | 151.0 | 122.3 |

leat's numbers include sampling on the device and reading the token back: after the prompt for pp512, after every token for tg128.

Through `leat serve`, the first token of a 2141-token prompt arrives after 549 ms, or after 55 ms when another conversation has cached its 2130-token system prompt.

Quality against llama.cpp on the same file (wikitext-2, chunks of 512 tokens with the second half of each scored). leat's prompt path runs each chunk as one prompt; its decode path runs the first half as a prompt and the second one token at a time. Both quantize activations to int8, as llama.cpp does:

| | perplexity, 20 chunks | mean KL divergence | same top token |
|---|---:|---:|---:|
| llama.cpp | 8.3870 | | |
| leat, prompt path | 8.3757 | 0.0012 | 98.3% |
| leat, decode path (2 chunks) | | 0.0013 | 98.4% |

For scale, ignoring Llama 3.1's RoPE frequency factors, a subtle bug, raises the KL from 0.0010 to 0.0026 on the first 5 chunks with the reference ops.

## Requirements

- Linux, Python 3.12+, [uv](https://docs.astral.sh/uv/)
- `clang`: tinygrad compiles its GPU command submission with it
- NVIDIA: the open kernel module and NVRTC (from the CUDA toolkit)

## Development

```bash
uv sync
uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest
```

The default run is hermetic and needs only a CPU. GPU and real-model tests are opt-in. `LLAMA_CPP` and `WIKITEXT` add the comparisons against llama.cpp: token ids, and KL divergence of the output distribution.

```bash
DEV=NV LEAT_MODEL=model.gguf LLAMA_CPP=llama.cpp/build/bin WIKITEXT=wiki.test.raw uv run pytest
```

## License

MIT
