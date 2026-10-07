# leat

A minimal, fast LLM inference engine built on [tinygrad](https://github.com/tinygrad/tinygrad).

leat runs GGUF models with their weights kept in the quantized storage format. The goal is single-stream decode limited by memory bandwidth, not by the engine. It was developed on an NVIDIA RTX 3090 and is moving to AMD's Strix Halo.

> Status: on NVIDIA, hand-written kernels decode every model below faster than llama.cpp, and process prompts at 0.79 to 1.07x its speed. On AMD's RDNA GPUs the decode kernels run, and on RDNA 3, as Strix Halo's, the prompt kernels too, on its WMMA matrix cores: so far tested on tinygrad's emulated GPU only. `leat serve` serves the OpenAI chat completions API to several clients at once, decoding their replies in batched steps that read each weight once for all of them. Every kernel is tested against an independent NumPy reference, and so are the plain tinygrad ops they replace, which run on any device; `LEAT_KERNELS=ref` runs everything that way.

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

leat agent, the assistant, runs beside the server, whose models it uses, and serves its app at http://127.0.0.1:8000. It searches the web through a [SearXNG](https://github.com/searxng/searxng) on the machine, and runs Python in [bubblewrap](https://github.com/containers/bubblewrap)'s sandbox, with the libraries of an environment of its own. [docs/agent-plan.md](docs/agent-plan.md) says what it is to become.

```bash
docker run -d --name searxng -p 127.0.0.1:8888:8080 -e SEARXNG_SECRET=$(openssl rand -hex 32) \
    -v $PWD/examples/searxng.yml:/etc/searxng/settings.yml:ro searxng/searxng
uv venv --python /usr/bin/python3 ~/.local/share/leat/sandbox
uv pip install --python ~/.local/share/leat/sandbox -r leat/agent/sandbox.txt
DEV=NV uv run leat serve Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf --max-context 16384
uv run leat agent
```

`leat bench` measures speed, of random prompts as llama-bench does or with `--chat` of replies to chat prompts, which a drafter guesses as it would in use, and `leat perplexity` measures quality, either on a text file or against logits saved by llama.cpp's `llama-perplexity --kl-divergence-base`. [scripts/validate.py](scripts/validate.py) checks a machine end to end in one command: the GPU tests, speed against llama.cpp on the same files, speculative decoding, and the server, into a Markdown report. [scripts/evaluate.py](scripts/evaluate.py) measures the agent against leat serve, on what people ask of it: whether it calls the tools it should and says what it should, each case several times.

## Supported

Architectures, text only, with their tokenizers and chat templates:

| `general.architecture` | Models |
|---|---|
| `llama` | Llama 3.x, Mistral 7B, Mistral Small 3.x |
| `qwen2` | Qwen2.5 |
| `qwen3`, `qwen3moe` | Qwen3 and its mixtures of experts |
| `qwen35moe` | Qwen3.5 and Qwen3.6's mixtures of experts |
| `gemma3` | Gemma 3 |
| `gemma4` | Gemma 4 26B A4B and 31B, not E2B and E4B, whose per-layer embeddings and shared KV layers it lacks |
| `gpt-oss` | gpt-oss |
| `phi3` | Phi-4-mini, Phi-3 mini |

Their parts: grouped-query attention, sliding windows, attention sinks, QK norms, biases, partial RoPE, RoPE scaled as Llama 3, YaRN and LongRoPE, SwiGLU, GELU and gpt-oss's clamped SwiGLU, mixtures of experts, beside a shared expert or one with a gate, logit soft-capping, and Qwen3.5's Gated DeltaNet linear attention, its recurrent state kept for each sequence, beside attention gated per head.

Tokenizers: SentencePiece, as Mistral 7B's and Gemma 3's; byte-level BPE with llama.cpp's Llama 3, Qwen2, Qwen3.5, GPT-2, StarCoder, GPT-4o and Tekken pre-tokenizers and the families that share them; and Gemma 4's.

Storage types, and the kernels that take them on the GPU; the reference ops take every type:

| | up to 8 tokens, matrix-vector | more tokens, int8 tensor cores |
|---|---|---|
| Q4_K, Q5_K, Q6_K, Q8_0, Q5_0 | NVIDIA, RDNA | NVIDIA, RDNA 3 |
| Q4_0, IQ4_NL, IQ4_XS, MXFP4 | NVIDIA, RDNA | NVIDIA, RDNA 3 |
| Q4_1, Q5_1 | NVIDIA, RDNA | |
| Q2_K, Q3_K, F32, F16, BF16 | | |

Devices: any tinygrad backend runs the reference ops. NVIDIA GPUs (`DEV=NV` or `CUDA`) run every kernel; AMD's RDNA 3 and 4 GPUs (`DEV=AMD`), as Strix Halo's, run the warp-level ones: matrix-vector products, norms, quantization, RoPE, decode attention, the mixtures' routing and their few-token path, Gated DeltaNet's convolution and recurrence, and sampling: every kernel of a decode step, batched or not. RDNA 3's also run the tensor-core ones, int8 matrix products and FlashAttention, on WMMA; prompts on RDNA 4 take the reference ops for now, but for Gated DeltaNet's.

Server: `/v1/chat/completions`, whole or streamed, `/v1/models` and `/v1/models/load`; a POST from another site's page in a browser is refused. Replies split into `reasoning_content`, as Qwen3's `<think>` blocks, Gemma 4's thought channel and gpt-oss's analysis channel, text, and tool calls in Llama 3's, Qwen's, Qwen3.5's, Gemma 4's and gpt-oss's syntax. Requests take stop strings, seeds and `chat_template_kwargs` such as `{"enable_thinking": false}`. Sampling, on the device, is greedy or by temperature, with `top_k`, `top_p` and `min_p`, which the kernels cut within a hundredth of a nat, and `presence_penalty` on the tokens a reply has generated; requests for other penalties, `logprobs` or several choices are refused. Each reply ends with `timings`, as llama.cpp's server sends them: the prompt's tokens past those cached and the reply's, each timed, from when the completion gets a slot, and their rates.

Concurrent requests: completions run together, one in each of `--slots` slots of the KV cache, 4 by default; more wait their turn. Each step prefills a chunk of one prompt, of 256 tokens at most while others decode, then decodes a token of every running completion in one batch, of up to 8, whose matrices read each weight once for all of them. A client that hangs up frees its slot at the next step. A slot past the others holds the padding of batches of 3, 5, 6 or 7, which run in the graphs of 4 and 8.

Prefix caching: a conversation continues in its slot, and a prompt that shares a prefix with any slot, such as a system prompt, starts from a copy of it. Qwen3.5's recurrent state holds all a slot ran, so there a prompt shares a slot's tokens only when it shares all of them, or all those before the state the slot kept 16 tokens before its last prompt's end: where a chat's next turn, which renders the last turn anew, and an agent's next step go on.

Speculative decoding: given a drafter with `--draft`, sequences decoding without `presence_penalty` have it guess ahead, as many at once as the drafter takes, each 3 tokens alone or with one other and 1 with two others, 8 tokens in all at most, as the kernels for more cost more than the guesses save; a step runs each sequence's guesses with its last token, reading the weights once for them all. It keeps each sequence's guesses up to the first the model would not have generated, and the token it generated there: as each draw depends only on the seed and the position, the reply is the one plain decoding gives, but where the kernels for several tokens round a near tie the other way. Drafters: Gemma 4's assistant, as `mtp-gemma-4-26B-A4B-it.gguf`, for Gemma 4 26B A4B, whose attention reads the model's own keys and values; and Qwen3.5's and Qwen3.6's MTP layer, in the model's own file, which `--draft` takes too: a layer with keys and values of its own, which every run of the model fills. Both draft for up to 3 sequences at once, past which plain steps were as fast. Gated DeltaNet's state goes back to the last token a step keeps, from the states the step saved after each.

Models: `leat serve` takes GGUF files and directories of them, and holds one model at a time, which answers every request whatever model it names. The first file loads at start. `POST /v1/models/load` with `{"model": id}`, an id that `/v1/models` lists, loads another once the completions before it have finished, the last one freed first.

Agent: `leat agent` keeps conversations in one SQLite file, `~/.local/share/leat/leat.db` by default, and runs their turns itself, not in the browser: a reply goes on when its page closes, and every page open on the app sees it stream and can stop it; a turn that fails is taken back, its message returned to send again. The model replies without thinking unless a message asks it to, by the app's Think toggle, sampled as Qwen3.6 recommends either way. It searches the web, through the SearXNG at `--search`, http://127.0.0.1:8888 by default, and reads pages, which the app shows as lines of its work and the pages as the answer's sources; a reply's calls run at once, up to 12 replies a turn, the last offered no tools so that it answers. Pages are the web's alone: an address on the machine's own network is refused, at every redirect too. The weather, now and for a week, comes of [Open-Meteo](https://open-meteo.com), which weather sites' pages, drawn in the browser, cannot. It remembers what the user tells of themselves, a fact to a memory, which every conversation begun after knows and the app's Memory page lists, to add to or delete; it forgets what it is asked to, and recalls earlier conversations, whose messages SQLite's full-text search indexes.

Files: the user's are in a workspace, `~/.local/share/leat/workspace` by default, which they upload to and attach to messages in the app, and whose Files page lists them. The model reads them, PDFs and Word, Excel and PowerPoint files too, writes and edits them, and runs Python among them, in bubblewrap's sandbox: it sees `/usr`, the workspace and the environment of `leat/agent/sandbox.txt`'s libraries, and no network, in 4 GB and 120 s. Documents are parsed there too, so that a file made to attack a parser attacks the sandbox. To make a document it first reads a skill, a guide to its kind with a template, of `leat/agent/skills`. The app shows the files a message attached and those its answer made, served sandboxed, and downloaded but for images, PDFs and plain text, which no page can act in. The app is a page with no dependencies, its Markdown and math rendered by `leat/agent/app/markdown.mjs`. The agent answers its own machine and network alone: a request must name it by an address or a `.local` name, which a site's page cannot by DNS rebinding, and a write must come from its own page.

## Measurements

RTX 3090, one sequence, in tokens per second; Q4_K_M files but for gpt-oss's, MXFP4. llama.cpp is b11372 with CUDA, and b9691 for Qwen3.6. leat's numbers include sampling on the device and reading the token back: after the prompt for pp512, after every token for tg128.

| | llama.cpp pp512 | leat pp512 | llama.cpp tg128 | leat tg128 |
|---|---:|---:|---:|---:|
| Llama 3.2 3B Instruct | 10907 | 10185 | 274.4 | 283.1 |
| Llama 3.1 8B Instruct | 5422 | 4829 | 147.7 | 152.9 |
| Mistral 7B Instruct v0.3 | 5454 | 4883 | 156.8 | 161.9 |
| Mistral Small 3.2 24B Instruct | 1996 | 1636 | 54.9 | 58.0 |
| Qwen2.5 7B Instruct | 5830 | 5263 | 152.6 | 165.7 |
| Qwen3 8B | 5296 | 4745 | 142.0 | 149.2 |
| Qwen3 30B A3B | 4697 | 5036 | 213.1 | 233.1 |
| Qwen3.6 35B A3B | 3516 | 3708 | 168.0 | 192.2 |
| Gemma 3 4B it | 9795 | 8436 | 203.0 | 226.8 |
| Gemma 3 12B it | 3463 | 2749 | 89.0 | 97.6 |
| Gemma 4 26B A4B it | 4823 | 4919 | 158.2 | 190.6 |
| Phi-4-mini Instruct | 10382 | 8593 | 237.4 | 239.6 |
| gpt-oss 20B | 6101 | 5846 | 213.2 | 230.0 |

After 8192 tokens of context, Llama 3.1 8B decodes at 122.5 tok/s against llama.cpp's 125.0.

Several sequences decoding at once, in tokens per second in all: `leat bench -s N` against llama.cpp's `llama-batched-bench -npp 1 -ntg 128 -npl 1,2,4`.

| | llama.cpp 1 | leat 1 | llama.cpp 2 | leat 2 | llama.cpp 4 | leat 4 |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.2 3B Instruct | 271.9 | 281.0 | 508.6 | 509.9 | 741.2 | 865.8 |
| Llama 3.1 8B Instruct | 147.2 | 152.3 | 275.9 | 294.1 | 398.9 | 518.9 |
| Mistral Small 3.2 24B Instruct | 54.6 | 57.7 | 105.6 | 108.8 | 140.7 | 183.5 |
| Qwen3 8B | 140.7 | 148.4 | 263.4 | 286.2 | 383.9 | 506.2 |
| Qwen3 30B A3B | 210.4 | 229.9 | 330.1 | 365.7 | 412.1 | 506.2 |
| Gemma 3 12B it | 88.9 | 97.3 | 166.7 | 185.7 | 237.4 | 312.8 |
| Gemma 4 26B A4B it | 157.9 | 190.1 | 273.3 | 310.3 | 363.0 | 443.4 |
| gpt-oss 20B | 213.3 | 228.3 | 333.4 | 335.6 | 438.0 | 437.5 |

Speculative decoding, over replies to eight chat prompts of 256 tokens each, the best of two runs, as `leat bench --chat` measures greedy ones: Qwen3.6 35B A3B with its MTP layer at 329.9 tok/s greedy against 189.0 without, 1.75x, and 283.8 against 189.2 at temperature 1; Gemma 4 26B A4B with its assistant at 294.0 against 186.5, 1.58x, and 285.9 against 185.9. Several sequences at once, in tokens per second in all, greedy: Qwen3.6 decodes 2 at 398.1 against 319.5, 1.25x, and 3 at 446.5 against 385.9, 1.16x; Gemma 4 2 at 325.1 against 294.0, 1.11x, and 3 at 373.3 against 359.7, 1.04x. A step's tokens often share experts: the mixtures' kernels read each row of an expert for all of them at about the same time, the later reads from cache, which made speculative decoding of Qwen3.6 7.8% faster.

Llama 3.2 3B decodes 1195 tok/s in all for 8 sequences, 149 each. A mixture of experts gains less from a batch, whose tokens read experts of their own.

Through `leat serve`, the first token of a 2141-token prompt to Llama 3.1 8B arrives after 449 ms, or after 22 ms when another conversation has cached its 2130-token system prompt. In the engine, past 2130 cached tokens, the first token after one more arrives in 8.8 ms and after 2 to 16 more in 12.7 ms, against 7.2 ms for a decode step. The server is ready 34 s after it starts, the file in the page cache, most of that spent compiling the graphs it replays: 14 s the graph for longer prompts, 7 s the graph for prompts of up to 16 new tokens, and 6.5 s those for decode steps of 2 and 4 sequences. Under load, three clients streaming replies at once each get about 90 tokens a second, and a fourth's 3,000-token prompt sent meanwhile gets its first token after 0.9 s.

Quality against llama.cpp on the same file: wikitext-2, chunks of 512 tokens with the second half of each scored, run as one prompt each. Both quantize activations to int8; the mixture's choice of experts amplifies that noise.

| | llama.cpp perplexity, 20 chunks | leat perplexity | mean KL divergence | same top token |
|---|---:|---:|---:|---:|
| Llama 3.2 3B Instruct | 11.8783 | 11.8669 | 0.0012 | 98.3% |
| Llama 3.1 8B Instruct | 8.3870 | 8.3740 | 0.0012 | 98.3% |
| Mistral 7B Instruct v0.3 | 7.3712 | 7.3699 | 0.0009 | 98.6% |
| Mistral Small 3.2 24B Instruct | 5.9424 | 5.9364 | 0.0017 | 98.2% |
| Qwen2.5 7B Instruct | 7.4307 | 7.3983 | 0.0034 | 96.7% |
| Qwen3 8B | 11.0321 | 11.0142 | 0.0031 | 97.3% |
| Qwen3 30B A3B | 9.4920 | 9.5012 | 0.0043 | 97.6% |
| Qwen3.6 35B A3B | 6.6609 | 6.6521 | 0.0069 | 96.7% |
| Gemma 3 4B it | 17.9125 | 17.8991 | 0.0093 | 96.2% |
| Gemma 3 12B it | 10.1972 | 10.1917 | 0.0054 | 97.3% |
| Phi-4-mini Instruct | 11.4617 | 11.4474 | 0.0032 | 97.2% |
| gpt-oss 20B | 384.8582 | 390.5311 | 0.0247 | 91.3% |

leat's decode path, which runs the second half of each chunk one token at a time, scores 0.0012 for Llama 3.1 8B, 0.0031 for Qwen3 30B A3B and 0.0038 for Qwen3.6 35B A3B on 2 chunks. For scale, ignoring Llama 3.1's RoPE frequency factors, a subtle bug, raises the KL from 0.0010 to 0.0026 on the first 5 chunks with the reference ops; Qwen3 8B scores 0.0025 against llama.cpp on the reference ops alone.

gpt-oss is trained for its harmony chat format and models raw text poorly: both engines score wikitext near 385, and there its KL, 0.025, is the highest of these models. Comparing it on its chat format, as Gemma 4 is below, is still to do.

Gemma 4's instruction-tuned model does not model raw text: both engines score wikitext in the tens of thousands. On its chat format, over the 542 positions of six answers to chat prompts, leat's next-token distributions differ from llama.cpp's by a mean KL of 0.0023, with the same top token at 98.3%.

## Requirements

- Linux, Python 3.12+, [uv](https://docs.astral.sh/uv/)
- `clang`: tinygrad compiles its GPU command submission with it
- NVIDIA: the open kernel module and NVRTC (from the CUDA toolkit)
- AMD: ROCm's comgr, with which tinygrad compiles the kernels' HIP C

## Development

```bash
uv sync
uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest -n auto
```

The default run is hermetic and needs only a CPU: every architecture's tiny random model against an independent NumPy reference, and generation, batching, sampling, the server and the CLI on the reference ops. GPU and real-model tests are opt-in. `LLAMA_CPP` and `WIKITEXT` add the comparisons against llama.cpp: token ids, and KL divergence of the output distribution.

```bash
DEV=NV LEAT_MODEL=model.gguf LLAMA_CPP=llama.cpp/build/bin WIKITEXT=wiki.test.raw uv run pytest
```

Without an AMD GPU, the kernels run on tinygrad's emulated RDNA 3 GPU, as [tests/hip.py](tests/hip.py) sets up: its HIP C compiled by the system's clang, and tinygrad's source tree, of the commit pyproject.toml pins, on PYTHONPATH for the emulator. The emulator is slow: the GPU tests, the kernels' and every tiny model through them, take about 10 minutes on 8 cores.

```bash
PYTHONPATH=path/to/tinygrad DEV=MOCK+AMD uv run pytest -m gpu -n auto
```

CI runs all but the real-model tests on every push: lint and types, the CPU suite on Python 3.12 and 3.14, and the GPU tests on the emulator. It fails if together they run less than 80% of leat; most of what they leave is NVIDIA's alone, the tensor cores' mma.sync, which the GPU run above covers.

## License

MIT
