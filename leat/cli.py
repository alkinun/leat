"""Command line: `leat agent`, `leat run`, `leat serve`, `leat bench` and `leat perplexity`."""

import argparse
import contextlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import jinja2
from tinygrad import Device

from leat import bench
from leat.agent.agent import Agent
from leat.agent.client import Client
from leat.agent.server import Server as AgentServer
from leat.agent.store import Store
from leat.agent.tools import files, weather, web
from leat.agent.workspace import Workspace
from leat.chat import ChatTemplate, split_reply
from leat.engine import Engine
from leat.gguf import GGUF
from leat.model import Config
from leat.sampler import Sampling
from leat.server import Server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="leat", description="A minimal, fast LLM inference engine."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    agent = commands.add_parser(
        "agent", help="run leat agent, the assistant, and serve its app; models come of leat serve"
    )
    agent.add_argument("--engine", default="http://127.0.0.1:8080", help="leat serve's address")
    agent.add_argument(
        "--search", default="http://127.0.0.1:8888", help="a SearXNG's address, for web search"
    )
    agent.add_argument("--host", default="127.0.0.1", help="0.0.0.0 for the home network too")
    agent.add_argument("--port", type=int, default=8000)
    agent.add_argument(
        "--data", type=Path, default=_data(),
        help="where its state is kept: its conversations and memories, the user's files in "
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
        "agent": _agent, "run": _run, "serve": _serve, "bench": _bench, "perplexity": _perplexity
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
    tools = [*web.tools(args.search), *weather.tools(), *files.tools(workspace)]
    agent = Agent(Store(args.data / "leat.db"), Client(args.engine), tools, workspace)
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
    engine = Engine(args.model, max_context=args.max_context, draft=args.draft)
    chat, tok = ChatTemplate(engine.gguf.metadata, engine.tokenizer), engine.tokenizer
    sampling = Sampling(args.temperature, args.top_k, args.top_p, args.min_p, args.presence_penalty)
    first = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(first)
    print(f"{engine.gguf.path.stem} on {Device.DEFAULT}, compiling...", end="", flush=True)
    engine.warm_up()
    print(" ready. Ctrl-D quits.")
    while True:
        try:
            messages.append({"role": "user", "content": input("> ")})
        except (EOFError, KeyboardInterrupt):
            print()
            return
        prompt, thinking = _prompt(chat, messages)
        if len(prompt) >= engine.max_context and len(messages) > len(first) + 1:
            print(f"[the conversation is {len(prompt)} tokens, over --max-context; starting over]")
            messages = [*first, messages[-1]]  # from the message just written
            prompt, thinking = _prompt(chat, messages)
        if len(prompt) >= engine.max_context:
            print(f"[the message makes {len(prompt)} tokens, over --max-context]")
            messages.pop()
            continue
        reply, step, start = [], tok.stream(), time.perf_counter()
        text, shown = "", ("", "")  # the reply so far, and its reasoning and text printed
        try:
            for t in engine.generate(prompt, engine.max_context - len(prompt), sampling):
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


def _prompt(chat: ChatTemplate, messages: list[dict]) -> tuple[list[int], bool]:
    # the chat's prompt, and whether it opens a block of reasoning
    try:
        rendered = chat.render(messages)
    except jinja2.TemplateError as e:  # such as a system prompt the template does not take
        raise SystemExit(f"the model's chat template refuses this chat: {e}") from None
    return chat.tokens(rendered), chat.opens_thinking(rendered)


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
    files = [f for p in args.models for f in (sorted(p.glob("*.gguf")) if p.is_dir() else [p])]
    if not files:
        raise SystemExit("no GGUF files: give some, or directories that hold some")
    if args.draft and len(files) > 1:
        raise SystemExit("--draft drafts for one model: serve that one alone")
    options = {"max_context": args.max_context, "slots": args.slots, "draft": args.draft}
    with Server(files, args.host, args.port, **options) as server:
        print(f"{files[0].stem} on {Device.DEFAULT}, compiling...", end=" ", flush=True)
        server.load(files[0].stem)
        print(f"the API at {_url(args.host, server.server_port)}/v1. Ctrl-C quits.", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()


def _bench(args: argparse.Namespace) -> None:
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
    ctx = bench.base_chunk(args.kl_base) if args.kl_base else args.ctx
    engine = Engine(args.model, max_context=ctx)
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
