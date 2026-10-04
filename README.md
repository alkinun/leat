# leat

A minimal, fast LLM inference engine built on [tinygrad](https://github.com/tinygrad/tinygrad).

leat runs GGUF models with their weights kept in the quantized storage format. The goal is single-stream decode limited by memory bandwidth, not by the engine. It targets NVIDIA RTX 30-series GPUs first, then AMD Strix Halo.

> Status: on NVIDIA, hand-written kernels decode Llama 3.1, Qwen3, Qwen3 MoE and Gemma 4 faster than llama.cpp, and process prompts at 0.8 to 0.9x its speed. `leat serve` serves the OpenAI chat completions API. Every kernel is tested against the reference ops, plain tinygrad code that runs on any device; `LEAT_KERNELS=ref` runs everything that way.

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

- Architectures: `llama` (Llama 3.x), `qwen3` and `qwen3moe` (Qwen3 and its mixtures of experts) and `gemma4` (Gemma 4, text only), with their tokenizers
- Storage types: F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, which covers Q4_K_M, Q5_K_M, Q6_K and Q8_0 files
- Devices: any tinygrad backend; developed on NVIDIA with `DEV=NV`
- Server: `/v1/chat/completions`, whole or streamed, with stop strings, seeds, tool calls in Llama 3's, Qwen3's and Gemma 4's syntax and `chat_template_kwargs` such as `{"enable_thinking": false}`, and `/v1/models`. Sampling is greedy or by temperature; requests for `top_p`, penalties, `logprobs` or several choices are refused. Completions run one at a time.
- Prefix caching: the KV cache keeps `--slots` sequences. A conversation continues in its slot, and a prompt that shares a prefix with any slot, such as a system prompt, starts from a copy of it.

## Measurements

RTX 3090, Q4_K_M files, one sequence, in tokens per second. llama.cpp is b11372 with CUDA. leat's numbers include sampling on the device and reading the token back: after the prompt for pp512, after every token for tg128.

| | llama.cpp pp512 | leat pp512 | llama.cpp tg128 | leat tg128 |
|---|---:|---:|---:|---:|
| Llama 3.1 8B Instruct | 5417 | 4330 | 147.6 | 151.1 |
| Qwen3 8B | 5244 | 4171 | 141.2 | 147.4 |
| Qwen3 30B A3B | 4681 | 4169 | 211.1 | 224.5 |
| Gemma 4 26B A4B it | 4844 | 4206 | 157.6 | 185.5 |

After 8192 tokens of context, Llama 3.1 8B decodes at 122.3 tok/s against llama.cpp's 125.0.

Through `leat serve`, the first token of a 2141-token prompt to Llama 3.1 8B arrives after 549 ms, or after 55 ms when another conversation has cached its 2130-token system prompt.

Quality against llama.cpp on the same file: wikitext-2, chunks of 512 tokens with the second half of each scored, run as one prompt each. Both quantize activations to int8; the mixture's choice of experts amplifies that noise.

| | llama.cpp perplexity, 20 chunks | leat perplexity | mean KL divergence | same top token |
|---|---:|---:|---:|---:|
| Llama 3.1 8B Instruct | 8.3870 | 8.3757 | 0.0012 | 98.3% |
| Qwen3 8B | 11.0321 | 11.0198 | 0.0030 | 97.5% |
| Qwen3 30B A3B | 9.4920 | 9.4946 | 0.0045 | 97.4% |

leat's decode path, which runs the second half of each chunk one token at a time, scores 0.0013 for Llama 3.1 8B and 0.0034 for Qwen3 30B A3B on 2 chunks. For scale, ignoring Llama 3.1's RoPE frequency factors, a subtle bug, raises the KL from 0.0010 to 0.0026 on the first 5 chunks with the reference ops; Qwen3 8B scores 0.0025 against llama.cpp on the reference ops alone.

Gemma 4's instruction-tuned model does not model raw text: both engines score wikitext in the tens of thousands. On its chat format, over the 542 positions of six answers to chat prompts, leat's next-token distributions differ from llama.cpp's by a mean KL of 0.0066, with the same top token at 97.8%.

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
