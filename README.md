<img src="leat/agent/app/logo.svg" alt="" width="48">

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
from leat.chat import ChatTemplate
from leat.engine import Engine

engine = Engine("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", max_context=4096)
chat = ChatTemplate(engine.gguf.metadata, engine.tokenizer)
prompt = chat.encode([{"role": "user", "content": "Why is the sky blue?"}])
print(engine.tokenizer.decode(list(engine.generate(prompt, max_tokens=256))))
```

leat agent, the assistant, runs beside the server, whose models it uses, and serves its app at http://127.0.0.1:8000. It searches the web through a [SearXNG](https://github.com/searxng/searxng) on the machine, and runs Python in [bubblewrap](https://github.com/containers/bubblewrap)'s sandbox, with the libraries of an environment of its own.

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

Architectures, with their tokenizers and chat templates, and images too for Gemma 3 and 4, Mistral Small 3 and Qwen3.5 and 3.6, as below:

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

Server: `/v1/chat/completions`, whole or streamed, `/v1/models`, which gives the loaded model's `max_context`, and `/v1/models/load`; a POST from a page in a browser is refused, as the server has none, even of a name made its address's. Replies split into `reasoning_content`, as Qwen3's `<think>` blocks, Gemma 4's thought channel and gpt-oss's analysis channel, text, and tool calls in Llama 3's, Qwen's, Qwen3.5's, Gemma 4's and gpt-oss's syntax. Requests take stop strings, 16 of 256 characters at most, seeds and `chat_template_kwargs` such as `{"enable_thinking": false}`. Sampling, on the device, is greedy or by temperature, with `top_k`, `top_p` and `min_p`, which the kernels cut within a hundredth of a nat, and `presence_penalty` on the tokens a reply has generated; requests for other penalties, `logprobs` or several choices are refused. Each reply ends with `timings`, as llama.cpp's server sends them: the prompt's tokens past those cached and the reply's, each timed, from when the completion gets a slot, and their rates.

Concurrent requests: completions run together, one in each of `--slots` slots of the KV cache, 4 by default; more wait their turn. Each step prefills a chunk of one prompt, of 256 tokens at most while others decode, then decodes a token of every running completion in one batch, of up to 8, whose matrices read each weight once for all of them. A client that hangs up frees its slot at the next step. A slot past the others holds the padding of batches of 3, 5, 6 or 7, which run in the graphs of 4 and 8.

Prefix caching: a conversation continues in its slot, and a prompt that shares a prefix with any slot, such as a system prompt, starts from a copy of it. Qwen3.5's recurrent state holds all a slot ran, so there a prompt shares a slot's tokens only when it shares all of them, or all those before the state the slot kept 16 tokens before its last prompt's end: where a chat's next turn, which renders the last turn anew, and an agent's next step go on.

Sliding windows: a layer that sees only the last positions, as most of Gemma 3's, Gemma 4's and gpt-oss's do, keeps a cache of its window and the most tokens a step runs of a sequence, its chunk of prompt, image or guesses, as a ring, rather than of the whole context, as only those are read. A ring holds a power of 2 of positions, so that a position's row is its low bits, which kept decoding as fast as before, where rings of 1536 for Gemma 4 decoded 0.3% slower as their rows were remainders of a division. Gemma 4 26B A4B's 25 sliding-window layers of 8 kv heads of 256 and a window of 1024 hold 2048 positions each, 0.84 GB in 2 slots rather than 6.7 GB at a context of 16384: on the 3090, the model with its vision encoder and its assistant takes 20.6 GB in 2 slots of 16384, and 21.7 GB in 2 of 32768. A slot then shares a prefix only where its rings still hold the window before the prefix's end, as llama.cpp's: a prompt that leaves a slot's tokens more than about 1000 positions before where the slot last ran starts over.

Speculative decoding: given a drafter with `--draft`, sequences decoding without `presence_penalty` have it guess ahead, as many at once as the drafter takes, each 3 tokens alone or with one other and 1 with two others, 8 tokens in all at most, as the kernels for more cost more than the guesses save; a step runs each sequence's guesses with its last token, reading the weights once for them all. It keeps each sequence's guesses up to the first the model would not have generated, and the token it generated there: as each draw depends only on the seed and the position, the reply is the one plain decoding gives, but where the kernels for several tokens round a near tie the other way. Drafters: Gemma 4's assistant, as `mtp-gemma-4-26B-A4B-it.gguf`, for Gemma 4 26B A4B, whose attention reads the model's own keys and values; and Qwen3.5's and Qwen3.6's MTP layer, in the model's own file, which `--draft` takes too: a layer with keys and values of its own, which every run of the model fills. Both draft for up to 3 sequences at once, past which plain steps were as fast. Gated DeltaNet's state goes back to the last token a step keeps, from the states the step saved after each.

Images: given a model's vision encoder, the `mmproj` GGUF of llama.cpp's, prompts take images, each upright as its EXIF has it and transparency over white, as transformers has them: Gemma 3's SigLIP scales an image to 896 by 896 pixels, which it pools into 256 embeddings; Gemma 4's scales it, its aspect kept, to at most 280 embeddings, each of 3 by 3 patches of 16 pixels; Mistral Small 3's Pixtral scales it, its aspect kept, to at most 1540 pixels a side and 1024 embeddings, as llama.cpp, each of 2 by 2 patches of 14 pixels, rows of them ended by [IMG_BREAK]; Qwen3.5's, as transformers' smart_resize, to 64 to 1024 embeddings, each of 2 by 2 patches of 16 pixels, a quarter of llama.cpp's most, which would take seconds to encode and a quarter of a context of 16384. An image's embeddings run in a chunk of their own, and see each other, as Gemma's do, or are read causally, as Mistral's and Qwen's are. Qwen3.5's M-RoPE turns an image's embeddings by their row and column over its frequencies' sections, interleaved, and the text after an image on past its longer side, fewer positions than it takes of the cache: each slot holds tables of RoPE of its own, which a step moves on as it runs past an image, so that every graph runs as it does without. It is encoded when its chunk runs, in one compiled graph whatever its size, on the matrix cores in f16 but its patches' projection, and not again where the cache holds it: its positions hold a key of its bytes, which a prompt shares with a slot as it would a token. A model takes the projector beside it whose name holds its own, as converters name both, unless the GPU has no room for it too, as for Qwen3.6 35B A3B past 12288 tokens of context in 4 slots on the 3090, which `leat serve` then says; or the one `--mmproj` gives. It lists no projector as a model. The API takes `image_url` parts of `data:` URLs, and fetches no other; each stands where it is in its message, whatever the template does with images, as llama.cpp's markers do. `/v1/models` says whether the loaded model takes images, and the tokens a square one takes, by which leat agent estimates its prompts. In `leat run`, `/image PATH` shows an image with the next message.

Models: `leat serve` takes GGUF files and directories of them, and holds one model at a time, which answers every request whatever model it names. The first file loads at start, the API served meanwhile: `/v1/models` says it is loading, and completions wait for it. `POST /v1/models/load` with `{"model": id}`, an id that `/v1/models` lists, loads another once the completions before it have finished, the last one freed first.

Agent: `leat agent` keeps conversations in one SQLite file, `~/.local/share/leat/leat.db` by default, and runs their turns itself, not in the browser: a reply goes on when its page closes, and every page open on the app sees it stream and can stop it; a turn that fails is taken back, its message returned to send again. The agent watches the engine, telling every app within half a minute when it comes up or goes away, as when the machine starts. The model replies without thinking unless a message asks it to, by the app's Think toggle, sampled as Qwen3.6 recommends either way. It searches the web, through the SearXNG at `--search`, http://127.0.0.1:8888 by default, and reads pages as Hermes Agent does: their content alone, as markdown, which [trafilatura](https://github.com/adbar/trafilatura) finds in the sandbox, where a page made to attack a parser attacks nothing else, up to 8,000 characters, the rest saved in the workspace to read on. Each source is numbered across the conversation, and the model cites them, [1], which the app links, listing the sources an answer cites, as ChatGPT and Perplexity do. A reply's calls run at once. Pages are the web's alone: an address on the machine's own network is refused, at every redirect too, checked as the connection is made, so that a site's DNS cannot answer the check with one address and the connection with another. The model names each conversation after its first exchange.

Memory, as ChatGPT, Claude and Hermes Agent keep theirs: facts about the user, each of a category, about them, their preferences, the people in their life, their work, their plans, or the household's, everyone's, which every conversation begun after knows, in a room of 3,000 characters that keeps them few; the instructions put it first, which made the model remember far more often. The model remembers, changes and forgets them as it talks, by rules of what is worth it, and each memory rests on the user's own words, which it quotes and the code finds in what they said: what the model read, a page or a file, never becomes what the user is. Each is dated when it was last made or said again, and a plan has its last day, after which it shows as passed; a memory said again is dated again, and when the room is full the model is told which were least recently confirmed. Passwords, PINs and the numbers of IDs, cards and accounts are never kept. Once a conversation is idle for two minutes the model reviews what was said since it last looked, to remember what it missed, change what changed or passed, and forget what breaks the rules; and each night it tidies each person's memory, merging what says the same and turning plans passed into what happened, adding nothing and forgetting a quarter at most, as ChatGPT's "dreaming" and OpenClaw's bounded consolidation do. Whatever is forgotten or changed is kept as it was, and the Memory page, which shows the very list the model reads, can undo it. recall finds what a person said in their earlier conversations, by their words, which SQLite's full-text search indexes, or by their time, giving each match with its turn's question and answer.

Household: the first to open the app on a new box names themselves and owns it, and all the box held before is theirs. Any other device asks to join, showing a six-digit code, and the owner lets it in from Settings, as a person new or known, once their own device shows the same code; the device then keeps a secret in a cookie, of which the box keeps the hash. Each person has their own conversations and memories, which their devices alone are told of; the household's memories, as its pets and address, and the files are everyone's. The owner alone manages the people and their devices, and the engine's model.

Context: a conversation's prompt is fit to the model's context, which leat serve reports, as Hermes Agent fits one. Past 60% of it, the tools' long answers before the latest messages are cleared, a line said in place of each, and if that is not enough the messages before them are summarized by the model into the system prompt, with any summary before, the summary asked at the end of the conversation as the engine's cache holds it, as Claude Code asks its own, rather than of a transcript read from nothing; the conversation itself is kept whole, and the app marks where the summary ends. A reply the context cuts off is redone once the prompt is smaller, or said to be cut off. Each change rewrites the prompt's start, so it is rare, and between changes every prompt extends the last, which the engine's cache holds: the time a message was sent begins it, rather than the date in the system prompt, the replies of earlier turns are rendered as they were, their reasoning kept, and a turn's last reply keeps the tools declared, told to answer. A page cleared from the prompt says where it is saved, to read again. A turn takes 25 replies at most, the last told to answer.

Files: the user's are in a workspace, `~/.local/share/leat/workspace` by default, which they upload to and attach to messages in the app, and whose Files page lists them. The model reads them, PDFs and Word, Excel and PowerPoint files as markdown that keeps their headings, lists and tables, as markitdown reads them, and sees images, those a message attached and those it reads, where the model takes images, as data: URLs of the workspace's files, writes and edits them, and runs Python among them, in bubblewrap's sandbox: it sees `/usr`, the workspace and the environment of `leat/agent/sandbox.txt`'s libraries, and no network, in 4 GB and 120 s. Documents are parsed there too, so that a file made to attack a parser attacks the sandbox. To make a document it first reads a skill, a guide to its kind with a template, of `leat/agent/skills`. The app shows the files a message attached and those its answer made, images as they are, which a message also attaches pasted, served sandboxed, and downloaded but for images, PDFs and plain text, which no page can act in. The app is a page with no dependencies, its Markdown and math rendered by `leat/agent/app/markdown.mjs`. The agent answers its own machine and network alone: a request must name it by an address or a `.local` name, which a site's page cannot by DNS rebinding, and a write must come from its own page.

Themes: the app's look is a theme's, chosen in Settings on each device, light, dark or as the system is. A theme is the few things that make it: its fonts, of the text, of the greeting and titles, and of code, the text's size and the titles' weight, its corners, of the controls and of the messages and composer, how much room it leaves, and its colors, light and dark, fourteen of each, the composer's shadow one. Leat comes with a few looks of its own: Leat's, black on white and white on black, sharp, with lines rather than shadows; Hermes, a terminal's gold on black after Hermes Agent's; and Newsprint, square and serif, in black rules and red ink. Every other comes of the themes people already have, which Settings opens as files: VS Code's color themes, as their extensions' `themes/*.json` are, of one scheme each, whose colors it takes, its editor's, sidebar's, inputs', buttons' and syntax's, what one does not say mixed of its text and background; and shadcn/ui's, as [tweakcn](https://tweakcn.com) makes them, its CSS or its registry's JSON, whose fonts, radius, spacing and shadows it takes too. Settings saves any theme as Leat's own JSON, to change and open again; whatever a theme leaves out is Leat's, and one of a single scheme is always that one. Fonts are named, not served, so each device uses the fonts it has, falling back to the next named:

```json
{
  "name": "Mint",
  "font": "Inter, system-ui, sans-serif", "displayFont": "Georgia, serif", "displayWeight": 400,
  "codeFont": "ui-monospace, monospace", "size": 16, "radius": 6, "radiusLarge": 18, "density": 1,
  "light": {"page": "#f3fbf8", "accent": "#0f9d76", "shadow": "0 4px 20px rgb(0 0 0 / 0.06)"},
  "dark": {"page": "#0f1a17", "accent": "#3ccf9f"}
}
```

The colors are `page`, `side`, `card`, `bubble`, `line`, `text`, `muted`, `accent`, `onAccent`, `error`, `keyword`, `string` and `number`, with `shadow`; [leat/agent/app/themes.mjs](leat/agent/app/themes.mjs) says what each is, how VS Code's and shadcn/ui's become them, and has the looks Leat comes with.

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

Vision encoders, an image's on the 3090, their attention FlashAttention's kernel, which makes no matrix of scores, padding hidden by a dimension past each head's: Gemma 4's takes 0.24 s for 280 embeddings, against 1.6 s in f32 off the matrix cores, Gemma 3's 0.32 s for 256, and Mistral Small 3's and Qwen3.6's 0.34 s and 0.33 s for 1024, of 4096 patches. Their embeddings in f16 are within a cosine of 0.9986 of transformers' in f32 for Gemma 4's, 0.9992 for Mistral Small 3's, 0.998 for Gemma 3's SigLIP and 0.52 for Qwen3.6's, of large activations, whose own in bf16 are within 0.777 and 0.186, and Qwen's mean cosine 0.997 against their 0.957. With Gemma 4 26B A4B, the first token after an image and 21 tokens arrives in 414 ms, and after the same prompt again in 16 ms.

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
