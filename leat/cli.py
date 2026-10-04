"""Command line: `leat run`, `leat serve`, `leat bench` and `leat perplexity`."""

import argparse
import contextlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import jinja2
from tinygrad import Device

from leat import bench
from leat.chat import ChatTemplate, split_reply
from leat.engine import Engine
from leat.server import Server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="leat", description="A minimal, fast LLM inference engine."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="chat with a model in the terminal")
    run.add_argument("model", type=Path, help="GGUF file")
    run.add_argument("--max-context", type=int, default=4096)
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--system", help="system prompt")

    serve = commands.add_parser("serve", help="serve the OpenAI chat completions API")
    serve.add_argument("model", type=Path, help="GGUF file")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--max-context", type=int, default=4096)
    serve.add_argument("--slots", type=int, default=4, help="sequences the KV cache keeps")

    speed = commands.add_parser("bench", help="measure prefill and decode speed")
    speed.add_argument("model", type=Path, help="GGUF file")
    speed.add_argument("-p", "--prompt", type=int, default=512, help="prompt tokens")
    speed.add_argument("-n", "--generate", type=int, default=128, help="generated tokens")
    speed.add_argument("-r", "--reps", type=int, default=3)
    speed.add_argument(
        "-s", "--sequences", type=int, default=1, help="sequences generating at once"
    )
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
        "--ctx", type=int, default=512,
        help="chunk size, as in llama-perplexity -c; a --kl-base file has its own",
    )  # fmt: skip
    quality.add_argument("--chunks", type=int, help="score only the first N chunks")
    quality.add_argument(
        "--decode", action="store_true", help="score one token at a time, as generation runs"
    )
    quality.add_argument("--json", action="store_true", help="print one JSON object")

    args = parser.parse_args(argv)
    {"run": _run, "serve": _serve, "bench": _bench, "perplexity": _perplexity}[args.command](args)


def _run(args: argparse.Namespace) -> None:
    engine = Engine(args.model, max_context=args.max_context)
    chat, tok = ChatTemplate(engine.gguf.metadata, engine.tokenizer), engine.tokenizer
    messages = [{"role": "system", "content": args.system}] if args.system else []
    print(f"{engine.gguf.path.stem} on {Device.DEFAULT}, compiling...", end="", flush=True)
    engine.warm_up()
    print(" ready. Ctrl-D quits.")
    while True:
        try:
            messages.append({"role": "user", "content": input("> ")})
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            rendered = chat.render(messages)
        except jinja2.TemplateError as e:  # such as a system prompt the template does not take
            raise SystemExit(f"the model's chat template refuses this chat: {e}") from None
        prompt, thinking = chat.tokens(rendered), chat.opens_thinking(rendered)
        if len(prompt) >= engine.max_context:
            print(f"[the conversation is {len(prompt)} tokens, over --max-context; starting over]")
            messages = messages[:1] if args.system else []
            continue
        reply, step, start = [], tok.stream(), time.perf_counter()
        text, shown = "", ("", "")  # the reply so far, and its reasoning and text printed
        try:
            for t in engine.generate(prompt, engine.max_context - len(prompt), args.temperature):
                if t in tok.eog_ids:
                    break
                reply.append(t)
                text += step(t)
                shown = _show(split_reply(text, chat.form, thinking), shown)
        except KeyboardInterrupt:
            pass
        text += step(None)
        parts = split_reply(text, chat.form, thinking)
        _show(parts, shown)
        rate = len(reply) / (time.perf_counter() - start)
        print(f"\n\033[2m[{len(reply)} tokens, {rate:.1f} tok/s]\033[0m")
        messages.append({"role": "assistant", "content": parts.content})


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
    engine = Engine(args.model, max_context=args.max_context, slots=args.slots)
    print(f"{engine.gguf.path.stem} on {Device.DEFAULT}, compiling...", end="", flush=True)
    engine.warm_up()
    with Server(engine, args.host, args.port) as server:
        url = f"http://{args.host}:{server.server_port}/v1"
        print(f" serving at {url}. Ctrl-C quits.", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()


def _bench(args: argparse.Namespace) -> None:
    context, n = args.prompt + args.generate, args.sequences
    engine = Engine(args.model, max_context=context, prefill_chunk=args.prompt, slots=n)
    result = bench.speed(engine, args.prompt, args.generate, args.reps, n)
    name = engine.gguf.path.stem
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
        result = bench.perplexity(engine, args.text.read_text(), args.ctx, args.chunks, args.decode)
    if args.json:
        print(json.dumps({"model": engine.gguf.path.stem} | asdict(result)))
        return
    print(f"perplexity {result.perplexity:.4f}")
    if result.kl_mean is not None:
        print(f"KL mean {result.kl_mean:.6f}  p99 {result.kl_p99:.6f}  max {result.kl_max:.6f}")
        print(f"same top token {result.top1:.2%}")
