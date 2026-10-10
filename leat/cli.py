"""Command line: `leat agent`, `leat run`, `leat serve`, `leat keys`, `leat bench` and
`leat perplexity`.

The engine's commands import it as they run, so that leat agent, which reaches the engine over HTTP
alone, never loads it, nor tinygrad.
"""

import argparse
import contextlib
import ipaddress
import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from leat.agent.agent import Agent
from leat.agent.client import Client
from leat.agent.server import Server as AgentServer
from leat.agent.store import Store
from leat.agent.tools import files, web
from leat.agent.workspace import Workspace
from leat.defaults import Overrides
from leat.keys import Keys

if TYPE_CHECKING:
    from leat.chat import ChatTemplate
    from leat.server import Server
    from leat.vision import Image


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="leat", description="A minimal, fast LLM inference engine."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    agent = commands.add_parser(
        "agent", help="run leat agent, the assistant, and serve its app; models come of leat serve"
    )
    agent.add_argument(
        "--engine", default="http://127.0.0.1:8080",
        help="leat serve's address; its API key, if it asks for one, in LEAT_ENGINE_KEY",
    )  # fmt: skip
    agent.add_argument(
        "--search", default="http://127.0.0.1:8888", help="a SearXNG's address, for web search"
    )
    agent.add_argument("--host", default="127.0.0.1", help="0.0.0.0 for the local network too")
    agent.add_argument("--port", type=int, default=8000)
    agent.add_argument(
        "--data", type=Path, default=_data(),
        help="where its state is kept: its conversations, the user's files in "
        "workspace/, and in sandbox/ the environment of libraries the sandbox offers",
    )  # fmt: skip

    run = commands.add_parser("run", help="chat with a model in the terminal")
    run.add_argument("model", type=Path, help="GGUF file")
    run.add_argument("--max-context", type=_positive, default=4096)
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument(
        "--top-k", type=int, default=0, help="keep the k likeliest tokens; 0 keeps all"
    )
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--min-p", type=float, default=0.0)
    run.add_argument("--presence-penalty", type=float, default=0.0)
    run.add_argument("--system", help="system prompt")
    run.add_argument(
        "--mmproj", type=Path,
        help="a vision encoder's GGUF, for images; by default the projector beside the model of "
        "its name, if any. `/image PATH` attaches an image to the next message",
    )  # fmt: skip
    run.add_argument(
        "--draft",
        type=Path,
        help="a drafter's GGUF, for speculative decoding: Gemma 4's assistant for Gemma 4, or "
        "for Qwen3.5's MTP layer the model's own",
    )

    serve = commands.add_parser("serve", help="serve the OpenAI chat completions API")
    serve.add_argument(
        "models", type=Path, nargs="+",
        help="GGUF files, or directories of them; the first file loads at start, others on request",
    )  # fmt: skip
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--max-context", type=_positive, default=4096)
    serve.add_argument(
        "--slots", type=_positive, default=4,
        help="sequences generating at once, each in a slot of the KV cache that keeps its tokens",
    )  # fmt: skip
    serve.add_argument(
        "--draft", type=Path, help="a drafter's GGUF, for speculative decoding, of the one model"
    )
    serve.add_argument(
        "--mmproj", type=Path,
        help="a vision encoder's GGUF, of the one model; by default each model takes the "
        "projector beside it of its name, if any",
    )  # fmt: skip
    serve.add_argument(
        "--keys", type=Path, nargs="?", const=_data() / "keys.json",
        help="a file of API keys, as `leat keys` makes it, one of which every request must hold; "
        f"by default {_data() / 'keys.json'}. Needed to serve beyond this machine",
    )  # fmt: skip

    serve.add_argument(
        "--sampling", type=Path, nargs="?", const=_data() / "sampling.json",
        help="a file of JSON of your own sampling, by model id, or \"*\" for every model, as "
        '{"gpt-oss-20b": {"temperature": 0.8}}, over what each model\'s makers recommend; by '
        f"default {_data() / 'sampling.json'}. Read again as it changes",
    )  # fmt: skip

    keys = commands.add_parser("keys", help="make, list and remove leat serve's API keys")
    keys.add_argument("action", choices=("add", "list", "remove"))
    keys.add_argument("name", nargs="?", help="the key's, a person's or an app's, as alkin")
    keys.add_argument("--file", type=Path, default=_data() / "keys.json", help="the file of keys")

    speed = commands.add_parser("bench", help="measure prefill and decode speed")
    speed.add_argument("model", type=Path, help="GGUF file")
    speed.add_argument("-p", "--prompt", type=_positive, default=512, help="prompt tokens")
    speed.add_argument("-n", "--generate", type=_positive, default=128, help="generated tokens")
    speed.add_argument("-r", "--reps", type=_positive, default=3)
    speed.add_argument(
        "-s", "--sequences", type=_positive, default=1, help="sequences generating at once"
    )
    speed.add_argument(
        "--draft",
        type=Path,
        help="a drafter's GGUF, for speculative decoding: Gemma 4's assistant for Gemma 4, or "
        "for Qwen3.5's MTP layer the model's own",
    )
    speed.add_argument(
        "--chat", action="store_true",
        help="decode replies to chat prompts in the model's template instead, as a drafter "
        "guesses them in use, for -n tokens each",
    )  # fmt: skip
    speed.add_argument("--json", action="store_true", help="print one JSON object")

    quality = commands.add_parser(
        "perplexity", help="measure quality on text, or against llama.cpp"
    )
    quality.add_argument("model", type=Path, help="GGUF file")
    source = quality.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", type=Path, help="text file to score")
    source.add_argument(
        "--kl-base", type=Path, help="logits from llama-perplexity --kl-divergence-base"
    )
    quality.add_argument(
        "--ctx", type=_positive, default=512,
        help="chunk size, as in llama-perplexity -c; a --kl-base file has its own",
    )  # fmt: skip
    quality.add_argument("--chunks", type=_positive, help="score only the first N chunks")
    quality.add_argument(
        "--decode", action="store_true", help="score one token at a time, as generation runs"
    )
    quality.add_argument("--json", action="store_true", help="print one JSON object")

    args = parser.parse_args(argv)
    handlers = {
        "agent": _agent, "run": _run, "serve": _serve, "keys": _keys, "bench": _bench,
        "perplexity": _perplexity,
    }  # fmt: skip
    handlers[args.command](args)


def _data() -> Path:
    # the agent's state, where XDG keeps applications' data
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "leat"


def _positive(text: str) -> int:
    # an argument that counts something, of which there must be one at least
    if (n := int(text)) < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


def _agent(args: argparse.Namespace) -> None:
    args.data.mkdir(parents=True, exist_ok=True)
    environment = args.data / "sandbox"
    workspace = Workspace(args.data / "workspace", environment if environment.exists() else None)
    engine = Client(args.engine, os.environ.get("LEAT_ENGINE_KEY"))
    tools = [*web.tools(args.search, engine), *files.tools()]
    agent = Agent(Store(args.data / "leat.db"), engine, tools, workspace)
    agent.start()
    with AgentServer(agent, args.host, args.port) as server:
        print(f"leat agent at {_url(args.host, server.server_port)}, its models of {args.engine}. "
              "Ctrl-C quits.", flush=True)  # fmt: skip
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()


def _url(host: str, port: int) -> str:
    # an address to browse to: this machine's, where the server takes every one
    return f"http://{'127.0.0.1' if host in ('0.0.0.0', '::', '') else host}:{port}"


def _run(args: argparse.Namespace) -> None:
    from tinygrad import Device

    from leat.chat import ChatTemplate, split_reply
    from leat.engine import Engine
    from leat.sampler import Sampling
    from leat.vision import Image, beside

    vision = args.mmproj or beside(args.model)
    engine = Engine(args.model, max_context=args.max_context, draft=args.draft, vision=vision)
    chat, tok = ChatTemplate(engine.gguf.metadata, engine.tokenizer), engine.tokenizer
    sampling = Sampling(args.temperature, args.top_k, args.top_p, args.min_p, args.presence_penalty)
    first = [{"role": "system", "content": args.system}] if args.system else []
    messages, pictures, attached = list(first), list[Image](), list[Image]()
    print(f"{engine.gguf.path.stem} on {Device.DEFAULT}, compiling...", end="", flush=True)
    engine.warm_up()
    print(f" ready.{' /image PATH attaches an image.' if engine.vision else ''} Ctrl-D quits.")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.startswith("/image "):
            try:  # shown before the next message's text
                path = Path(line[7:].strip().strip("'\"")).expanduser()  # as terminals quote one
                attached.append(engine.image(path.read_bytes()))
            except (OSError, ValueError) as e:  # as of a model without a vision encoder
                print(f"[no image: {e}]")
            continue
        n = len(attached)  # the message's images, before its text
        images = [{"type": "image"}] * n + [{"type": "text", "text": line}]
        messages.append({"role": "user", "content": images if n else line})
        pictures, attached = pictures + attached, []
        prompt, thinking = _prompt(chat, messages, pictures)
        if len(prompt) >= engine.max_context and len(messages) > len(first) + 1:
            print(f"[the conversation is {len(prompt)} tokens, over --max-context; starting over]")
            messages = [*first, messages[-1]]  # from the message just written
            pictures = pictures[len(pictures) - n :]
            prompt, thinking = _prompt(chat, messages, pictures)
        if len(prompt) >= engine.max_context:
            print(f"[the message makes {len(prompt)} tokens, over --max-context]")
            messages.pop()
            pictures = pictures[: len(pictures) - n]
            continue
        reply, step, start = [], tok.stream(), time.perf_counter()
        text, shown = "", ("", "")  # the reply so far, and its reasoning and text printed
        try:
            for t in engine.generate(prompt, engine.max_context - len(prompt), sampling,
                                     images=pictures):  # fmt: skip
                if t in tok.eog_ids:
                    break
                reply.append(t)
                text += step(t)
                shown = _show(split_reply(text, chat.form, thinking), shown)
        except KeyboardInterrupt:
            pass
        text += step(None)
        parts = split_reply(text, chat.form, thinking, done=True)
        _show(parts, shown)
        rate = len(reply) / (time.perf_counter() - start)
        print(f"\n\033[2m[{len(reply)} tokens, {rate:.1f} tok/s]\033[0m")
        messages.append({"role": "assistant", "content": parts.content})


def _prompt(
    chat: "ChatTemplate", messages: list[dict], images: list["Image"]
) -> tuple[list[int], bool]:
    # the chat's prompt, showing its images in turn, and whether it opens a block of reasoning
    import jinja2

    try:
        rendered = chat.render(messages)
    except jinja2.TemplateError as e:  # such as a system prompt the template does not take
        raise SystemExit(f"the model's chat template refuses this chat: {e}") from None
    return chat.tokens(rendered, [image.tokens for image in images]), chat.opens_thinking(rendered)


def _show(parts, shown: tuple[str, str]) -> tuple[str, str]:
    # prints what is new of a reply's reasoning, dimmed, and of its text
    reasoning, content = shown
    if len(parts.reasoning) > len(reasoning):
        print(f"\033[2m{parts.reasoning[len(reasoning) :]}\033[0m", end="", flush=True)
    if len(parts.content) > len(content):
        if not content and parts.reasoning:
            print()
        print(parts.content[len(content) :], end="", flush=True)
    return parts.reasoning, parts.content


def _serve(args: argparse.Namespace) -> None:
    from tinygrad import Device

    from leat.server import Server
    from leat.vision import projector

    models = [f for p in args.models for f in (sorted(p.glob("*.gguf")) if p.is_dir() else [p])]
    models = [f for f in models if not projector(f)]  # each model's vision encoder, not a model
    if not models:
        raise SystemExit("no GGUF files: give some, or directories that hold some")
    for flag in ("draft", "mmproj"):
        if getattr(args, flag) and len(models) > 1:
            raise SystemExit(f"--{flag} is of one model: serve that one alone")
    if args.keys is None and not _loopback(args.host):
        raise SystemExit(f"to serve beyond this machine, at {args.host}, give --keys: make one "
                         "with `leat keys add NAME`")  # fmt: skip
    if args.keys is not None and not args.keys.exists():
        raise SystemExit(f"there are no keys at {args.keys}: make one with "
                         f"`leat keys add NAME --file {args.keys}`")  # fmt: skip
    options = {"max_context": args.max_context, "slots": args.slots, "draft": args.draft,
               "vision": args.mmproj}  # fmt: skip
    keys = Keys(args.keys) if args.keys is not None else None
    sampling = Overrides(args.sampling) if args.sampling is not None else None
    if sampling is not None:
        try:
            sampling.of("")
        except (OSError, ValueError) as e:
            raise SystemExit(f"the sampling at {args.sampling} cannot be read: {e}") from e
    with Server(models, args.host, args.port, keys, sampling, **options) as server:
        # served at once, the model loading meanwhile, so that a client asking while it compiles
        # hears it is loading, its completions waiting for it, rather than no answer
        url, name = _url(args.host, server.server_port), models[0].stem
        failed: list[Exception] = []
        print(f"{name} on {Device.DEFAULT}, the API at {url}/v1, compiling...", end=" ", flush=True)
        threading.Thread(target=_start, args=(server, name, failed), daemon=True).start()
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
        if failed:
            raise failed[0]


def _loopback(host: str) -> bool:
    # whether a host the server binds to is this machine's alone
    with contextlib.suppress(ValueError):
        return ipaddress.ip_address(host).is_loopback
    return host == "localhost"


def _keys(args: argparse.Namespace) -> None:
    keys = Keys(args.file)
    if args.action != "list" and not args.name:
        raise SystemExit(f"leat keys {args.action} needs the key's name")
    try:
        if args.action == "list":
            for listed in keys.listed():
                print(f"{listed['name']}\t{listed['created']}")
        elif args.action == "add":
            key = keys.add(args.name)
            print(f"{key}\n\nThe key of {args.name}, shown this once: keep it where its app reads "
                  f"it, as OPENAI_API_KEY. {args.file} keeps its hash alone.")  # fmt: skip
        else:
            keys.remove(args.name)
            print(f"{args.name}'s key is removed: leat serve refuses it from the next request on.")
    except (ValueError, LookupError) as e:
        raise SystemExit(str(e)) from e


def _start(server: "Server", name: str, failed: list[Exception]) -> None:
    # loads the model a server starts with, or stops the server if it cannot
    try:
        server.load(name)
        print("ready. Ctrl-C quits.", flush=True)
    except Exception as e:
        failed.append(e)
        server.shutdown()


def _bench(args: argparse.Namespace) -> None:
    from tinygrad import Device

    from leat import bench
    from leat.engine import Engine
    from leat.gguf import GGUF
    from leat.model import Config

    n = args.sequences
    context = (bench.CHAT_CONTEXT if args.chat else args.prompt) + args.generate
    if args.chat:  # as long as the model's context allows
        metadata = GGUF.open(args.model).metadata
        context = min(context, Config.from_gguf(metadata).context_length)
    engine = Engine(
        args.model, max_context=context, prefill_chunk=args.prompt, slots=n, draft=args.draft
    )
    name = engine.gguf.path.stem
    if args.chat:
        chat = bench.chat_speed(engine, args.generate, args.reps, n)
        if args.json:
            print(json.dumps({"model": name, "device": Device.DEFAULT} | asdict(chat)))
            return
        each = f", {n} at once, {chat.decode / n:.1f} each" if n > 1 else ""
        print(f"{name} on {Device.DEFAULT}")
        print(f"  chat tg{args.generate:<4} {chat.decode:8.1f} tok/s   "
              f"{chat.per_step:.2f} tokens a step{each}")  # fmt: skip
        return
    result = bench.speed(engine, args.prompt, args.generate, args.reps, n)
    if args.json:
        print(json.dumps({"model": name, "device": Device.DEFAULT} | asdict(result)))
        return
    print(f"{name} on {Device.DEFAULT}")
    print(f"  pp{args.prompt:<6} {result.prefill:10.1f} tok/s")
    gbs = f"{result.weight_gbs:.0f} GB/s of weights"
    each = f", {n} at once, {result.decode / n:.1f} each" if n > 1 else ""
    print(f"  tg{args.generate:<6} {result.decode:10.1f} tok/s   {gbs}{each}")


def _perplexity(args: argparse.Namespace) -> None:
    from leat import bench
    from leat.engine import Engine

    ctx = bench.base_chunk(args.kl_base) if args.kl_base else args.ctx
    engine = Engine(args.model, max_context=ctx, prefill_chunk=ctx)  # a chunk in one run
    if args.kl_base:
        result = bench.kl_divergence(engine, args.kl_base, args.chunks, args.decode)
    else:
        # its bytes as they are, \r\n too, as llama-perplexity reads them
        text = args.text.read_bytes().decode()
        result = bench.perplexity(engine, text, args.ctx, args.chunks, args.decode)
    if args.json:
        print(json.dumps({"model": engine.gguf.path.stem} | asdict(result)))
        return
    print(f"perplexity {result.perplexity:.4f}")
    if result.kl_mean is not None:
        print(f"KL mean {result.kl_mean:.6f}  p99 {result.kl_p99:.6f}  max {result.kl_max:.6f}")
        print(f"same top token {result.top1:.2%}")
