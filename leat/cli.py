"""Command line: `leat run`, `leat bench` and `leat perplexity`."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from tinygrad import Device

from leat import bench
from leat.chat import ChatTemplate
from leat.engine import Engine


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

    speed = commands.add_parser("bench", help="measure prefill and decode speed")
    speed.add_argument("model", type=Path, help="GGUF file")
    speed.add_argument("-p", "--prompt", type=int, default=512, help="prompt tokens")
    speed.add_argument("-n", "--generate", type=int, default=128, help="generated tokens")
    speed.add_argument("-r", "--reps", type=int, default=3)
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
        "--ctx", type=int, default=512, help="chunk size, as in llama-perplexity -c"
    )
    quality.add_argument("--chunks", type=int, help="score only the first N chunks")
    quality.add_argument(
        "--decode", action="store_true", help="score one token at a time, as generation runs"
    )
    quality.add_argument("--json", action="store_true", help="print one JSON object")

    args = parser.parse_args(argv)
    {"run": _run, "bench": _bench, "perplexity": _perplexity}[args.command](args)


def _run(args: argparse.Namespace) -> None:
    engine = Engine(args.model, max_context=args.max_context)
    chat, tok = ChatTemplate(engine.gguf.metadata, engine.tokenizer), engine.tokenizer
    messages = [{"role": "system", "content": args.system}] if args.system else []
    name = engine.gguf.metadata.get("general.name", args.model.name)
    print(f"{name} on {Device.DEFAULT}. Ctrl-D quits.")
    while True:
        try:
            messages.append({"role": "user", "content": input("> ")})
        except (EOFError, KeyboardInterrupt):
            print()
            return
        prompt = chat.encode(messages)
        if len(prompt) >= engine.max_context:
            print(f"[the conversation is {len(prompt)} tokens, over --max-context; starting over]")
            messages = messages[:1] if args.system else []
            continue
        reply, step, start = [], tok.stream(), time.perf_counter()
        try:
            for t in engine.generate(prompt, engine.max_context - len(prompt), args.temperature):
                if t in tok.eog_ids:
                    break
                reply.append(t)
                print(step(t), end="", flush=True)
        except KeyboardInterrupt:
            pass
        rate = len(reply) / (time.perf_counter() - start)
        print(f"\n\033[2m[{len(reply)} tokens, {rate:.1f} tok/s]\033[0m")
        messages.append({"role": "assistant", "content": tok.decode(reply)})


def _bench(args: argparse.Namespace) -> None:
    engine = Engine(args.model, max_context=args.prompt + args.generate, prefill_chunk=args.prompt)
    result = bench.speed(engine, args.prompt, args.generate, args.reps)
    if args.json:
        print(json.dumps({"model": args.model.name, "device": Device.DEFAULT} | asdict(result)))
        return
    print(f"{args.model.name} on {Device.DEFAULT}")
    print(f"  pp{args.prompt:<6} {result.prefill:10.1f} tok/s")
    gbs = f"{result.weight_gbs:.0f} GB/s of weights"
    print(f"  tg{args.generate:<6} {result.decode:10.1f} tok/s   {gbs}")


def _perplexity(args: argparse.Namespace) -> None:
    engine = Engine(args.model, max_context=args.ctx)
    if args.kl_base:
        result = bench.kl_divergence(engine, args.kl_base, args.chunks, args.decode)
    else:
        result = bench.perplexity(engine, args.text.read_text(), args.ctx, args.chunks, args.decode)
    if args.json:
        print(json.dumps({"model": args.model.name} | asdict(result)))
        return
    print(f"perplexity {result.perplexity:.4f}")
    if result.kl_mean is not None:
        print(f"KL mean {result.kl_mean:.6f}  p99 {result.kl_p99:.6f}  max {result.kl_max:.6f}")
        print(f"same top token {result.top1:.2%}")
