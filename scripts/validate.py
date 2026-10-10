"""Checks a machine end to end in one command, as on a new box's first day: the GPU tests, speed
against llama.cpp on the same files, speculative decoding over chat prompts, and the server
answering a streamed chat. Writes a Markdown report to reports/ and prints it.

    uv run python scripts/validate.py MODEL.gguf|DIR ... [--llama-cpp llama.cpp/build/bin]

Every model runs in a process of its own, so each has the GPU's memory to itself. DEV picks the
device as for leat itself, AMD by default. A drafter is found for Qwen3.5's and Qwen3.6's mixtures
of experts, whose own files hold their MTP layer, and for Gemma 4, an `mtp-*.gguf` beside it or in
the directories given. --quick runs fewer repetitions and skips the batched and chat runs.
"""

import argparse
import contextlib
import datetime
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leat.draft import mtp, own_drafter  # noqa: E402
from leat.gguf import GGUF  # noqa: E402

# a reply to this, streamed from the server, shows the model loads, generates and detokenizes
SERVER_PROMPT = "Name three primary colors, one per line."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="+", type=Path, help="GGUF files, or directories of them")
    parser.add_argument("--llama-cpp", type=Path, help="llama.cpp's build/bin, to compare against")
    parser.add_argument("--quick", action="store_true", help="fewer and shorter runs")
    parser.add_argument("--no-tests", action="store_true", help="skip the GPU tests")
    parser.add_argument("--out", type=Path, default=ROOT / "reports", help="report directory")
    args = parser.parse_args()
    env = os.environ | {"DEV": os.environ.get("DEV", "AMD")}
    models = [m for m in _ggufs(args.models) if not _is_drafter(m)]
    drafters = [m for m in _ggufs(args.models) if _is_drafter(m)]
    report = Report(env["DEV"])
    report.machine(_machine(env))
    if not args.no_tests:
        report.tests(_tests(env))
    reps = "1" if args.quick else "3"
    for model in models:
        print(f"== {model.name}", file=sys.stderr, flush=True)
        row: dict = {"model": model.stem}
        row["leat"] = _leat(env, "bench", model, "-r", reps)
        if args.llama_cpp:
            row["llama.cpp"] = _llama_bench(args.llama_cpp, model, reps)
        if not args.quick:
            row["leat 4"] = _leat(env, "bench", model, "-r", reps, "-s", "4", "-p", "64")
            if args.llama_cpp:
                row["llama.cpp 4"] = _llama_batched(args.llama_cpp, model)
            draft = _drafter(model, drafters + _ggufs([model.parent]))
            if draft is not None:
                chat = ("bench", model, "--chat", "-n", "256", "-r", "2")
                row["chat"] = _leat(env, *chat)
                row["chat draft"] = _leat(env, *chat, "--draft", draft)
                row["chat draft 2"] = _leat(env, *chat, "--draft", draft, "-s", "2")
                row["chat 2"] = _leat(env, *chat, "-s", "2")
        report.model(row)
    if models:
        report.server(_server(env, models[0]))
    path = report.write(args.out)
    print(report.text())
    print(f"\nwritten to {path}", file=sys.stderr)


class Report:
    """The report's sections, in Markdown, and the raw results at its end."""

    def __init__(self, device: str):
        self.device, self.sections, self.raw = device, dict[str, str](), dict[str, object]()
        self.rows: list[dict] = []
        self.started = datetime.datetime.now()

    def machine(self, info: dict) -> None:
        self.raw["machine"] = info
        lines = [f"- {k}: {v}" for k, v in info.items()]
        self.sections["machine"] = "## Machine\n\n" + "\n".join(lines)

    def tests(self, result: dict) -> None:
        self.raw["tests"] = result
        verdict = "passed" if result["ok"] else "FAILED"
        text = f"## GPU tests: {verdict}\n\n`{result['summary']}` in {result['seconds']:.0f} s"
        if result["failures"]:
            text += "\n\n" + "\n".join(f"- {f}" for f in result["failures"])
        self.sections["tests"] = text

    def model(self, row: dict) -> None:
        self.rows.append(row)
        self.raw.setdefault("models", []).append(row)  # type: ignore[union-attr]

    def server(self, result: dict) -> None:
        self.raw["server"] = result
        verdict = "answered" if result["ok"] else "FAILED"
        detail = ", ".join(f"{k} {v}" for k, v in result.items() if k not in ("ok", "reply"))
        reply = result.get("reply", "").strip().replace("\n", " / ")[:300]
        self.sections["server"] = f"## Server: {verdict}\n\n{detail}\n\n> {reply}"

    def text(self) -> str:
        head = f"# leat on {self.device}, {self.started:%Y-%m-%d %H:%M}"
        raw = "## Raw\n\n```json\n" + json.dumps(self.raw, indent=1) + "\n```\n"
        self.sections["speed"] = self._speed()
        order = ("machine", "tests", "speed", "server")
        return "\n\n".join([head, *(self.sections[k] for k in order if k in self.sections), raw])

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname()
        path = directory / f"validate-{host}-{self.device}-{self.started:%Y%m%d-%H%M}.md"
        path.write_text(self.text())
        return path

    def _speed(self) -> str:
        # tokens per second: one sequence, four at once, and chat replies speculative or not
        def at(row: dict, key: str, field: str) -> str:
            value = (row.get(key) or {}).get(field)
            return "" if value is None else f"{value:.1f}"

        lines = [
            "## Speed, tokens per second\n",
            "| | llama.cpp pp512 | leat pp512 | llama.cpp tg128 | leat tg128 "
            "| llama.cpp 4 at once | leat 4 at once |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.rows:
            lines.append(
                f"| {r['model']} | {at(r, 'llama.cpp', 'prefill')} | {at(r, 'leat', 'prefill')} "
                f"| {at(r, 'llama.cpp', 'decode')} | {at(r, 'leat', 'decode')} "
                f"| {at(r, 'llama.cpp 4', 'decode')} | {at(r, 'leat 4', 'decode')} |"
            )
        chats = [r for r in self.rows if r.get("chat")]
        if chats:
            lines += [
                "\nChat replies, greedy, 256 tokens each, without a drafter and with one, and the "
                "tokens a reply takes per step:\n",
                "| | plain | speculative | tokens a step | plain, 2 at once | speculative, 2 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for r in chats:
                lines.append(
                    f"| {r['model']} | {at(r, 'chat', 'decode')} | {at(r, 'chat draft', 'decode')} "
                    f"| {at(r, 'chat draft', 'per_step')} | {at(r, 'chat 2', 'decode')} "
                    f"| {at(r, 'chat draft 2', 'decode')} |"
                )
        errors = [(r["model"], k, v["error"]) for r in self.rows for k, v in r.items()
                  if isinstance(v, dict) and "error" in v]  # fmt: skip
        if errors:
            lines += ["\nErrors:\n"] + [f"- {m}, {k}: `{e}`" for m, k, e in errors]
        return "\n".join(lines)


def _ggufs(paths: list[Path]) -> list[Path]:
    # the files given, and those in the directories given; split files by their first part
    found = []
    for path in paths:
        files = sorted(path.glob("*.gguf")) if path.is_dir() else [path]
        found += [f for f in files if not re.search(r"-0000[2-9]-of-|mmproj", f.name)]
    return list(dict.fromkeys(found))


def _is_drafter(path: Path) -> bool:
    # a Gemma 4 assistant, or a model's MTP layer alone
    with contextlib.suppress(Exception):
        return mtp(path) or GGUF.open(path).metadata["general.architecture"] == "gemma4-assistant"
    return False


def _drafter(model: Path, candidates: list[Path]) -> Path | None:
    # the model's own MTP layer, in its file or beside it, or a Gemma 4 assistant for Gemma 4
    if (own := own_drafter(model)) is not None:
        return own
    if GGUF.open(model).metadata["general.architecture"] == "gemma4":
        return next((c for c in candidates if _is_drafter(c) and not mtp(c)), None)
    return None


def _machine(env: dict) -> dict:
    info = {"host": socket.gethostname(), "kernel": platform.release(), "device": env["DEV"]}
    with contextlib.suppress(Exception):
        cpu = Path("/proc/cpuinfo").read_text()
        info["cpu"] = re.search(r"model name\s*:\s*(.*)", cpu).group(1)  # type: ignore[union-attr]
        mem = re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text())
        info["memory"] = f"{int(mem.group(1)) / 2**20:.0f} GiB"  # type: ignore[union-attr]
    probe = "from tinygrad import Device; d = Device[Device.DEFAULT]; print(getattr(d, 'arch', ''))"
    done = _run([sys.executable, "-c", probe], env, 300)
    info["arch"] = done["out"].strip().splitlines()[-1] if done["ok"] and done["out"] else "?"
    commit = _run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], env, 30)
    info["leat"] = commit["out"].strip()
    return info


def _tests(env: dict) -> dict:
    start = time.perf_counter()
    # one process on a real GPU, which the workers would share; many on the emulated one
    workers = ["-n", "auto"] if env["DEV"].startswith("MOCK") else []
    args = [sys.executable, "-m", "pytest", "-m", "gpu", "-q", *workers, "-p", "no:cacheprovider",
            str(ROOT / "tests")]  # fmt: skip
    done = _run(args, env, 7200, cwd=ROOT)
    lines = done["out"].strip().splitlines()
    summary = next((line for line in reversed(lines) if " in " in line), "no summary")
    failures = [line.removeprefix("FAILED ") for line in lines if line.startswith("FAILED ")]
    seconds = time.perf_counter() - start
    return {"ok": done["ok"], "summary": summary.strip("= "), "failures": failures,
            "seconds": seconds}  # fmt: skip


def _leat(env: dict, *args: object) -> dict:
    done = _run([sys.executable, "-m", "leat", *map(str, args), "--json"], env, 3600, cwd=ROOT)
    if not done["ok"]:
        return {"error": _last_line(done)}
    return json.loads(done["out"].strip().splitlines()[-1])


def _llama_bench(bin_dir: Path, model: Path, reps: str) -> dict:
    args = [str(bin_dir / "llama-bench"), "-m", str(model), "-p", "512", "-n", "128", "-r", reps,
            "-ngl", "99", "-fa", "1", "-o", "json"]  # fmt: skip
    done = _run(args, os.environ.copy(), 3600)
    if not done["ok"]:
        return {"error": _last_line(done)}
    rows = json.loads(done["out"])
    speeds = {("decode" if r["n_gen"] else "prefill"): r["avg_ts"] for r in rows}
    return speeds | {"build": rows[0].get("build_commit"), "backend": rows[0].get("backends")}


def _llama_batched(bin_dir: Path, model: Path) -> dict:
    # llama-batched-bench's speed of 4 sequences generating 128 tokens each after 1-token prompts
    args = [str(bin_dir / "llama-batched-bench"), "-m", str(model), "-c", "2048", "-ngl", "99",
            "-fa", "on", "-npp", "1", "-ntg", "128", "-npl", "4"]  # fmt: skip
    done = _run(args, os.environ.copy(), 3600)
    rows = [line.split("|") for line in done["out"].splitlines() if re.match(r"\|\s+1 \|", line)]
    if not done["ok"] or not rows:
        return {"error": _last_line(done)}
    return {"decode": float(rows[0][8])}  # the column S_TG t/s


def _server(env: dict, model: Path) -> dict:
    # leat serve on a free port, until it lists its model loaded, as it answers while it loads,
    # then a streamed chat reply
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    start = time.perf_counter()
    args = [sys.executable, "-m", "leat", "serve", str(model), "--port", str(port)]
    (ROOT / "reports").mkdir(exist_ok=True)
    log = open(ROOT / "reports" / ".server.log", "w")  # noqa: SIM115
    server = subprocess.Popen(args, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    url, result = f"http://127.0.0.1:{port}", {"ok": False}
    try:
        while time.perf_counter() - start < 1800 and server.poll() is None:
            with contextlib.suppress(OSError, ValueError):
                listed = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=5).read())
                if any(m.get("status") == "loaded" for m in listed["data"]):
                    break
            time.sleep(2)
        result["ready after"] = f"{time.perf_counter() - start:.0f} s"
        body = {"messages": [{"role": "user", "content": SERVER_PROMPT}], "max_tokens": 200,
                "stream": True, "chat_template_kwargs": {"enable_thinking": False}}  # fmt: skip
        request = urllib.request.Request(
            url + "/v1/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )  # fmt: skip
        sent, first, text = time.perf_counter(), None, []
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data: {"):
                    continue
                for choice in json.loads(line[6:]).get("choices", []):
                    if piece := choice.get("delta", {}).get("content"):
                        first = first or time.perf_counter()
                        text.append(piece)
        if first is not None:
            result["first token"] = f"{1000 * (first - sent):.0f} ms"
        result |= {"ok": bool("".join(text).strip()), "reply": "".join(text)}
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e)
    finally:
        server.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            server.wait(30)
        log.close()
    return result


def _run(args: list[str], env: dict, timeout: int, cwd: Path | None = None) -> dict:
    try:
        done = subprocess.run(args, env=env, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)  # fmt: skip
    except subprocess.TimeoutExpired:
        return {"ok": False, "out": "", "err": f"timed out after {timeout} s"}
    except OSError as e:
        return {"ok": False, "out": "", "err": str(e)}
    return {"ok": done.returncode == 0, "out": done.stdout, "err": done.stderr}


def _last_line(done: dict) -> str:
    lines = (done["err"] or done["out"]).strip().splitlines()
    return lines[-1][:300] if lines else "failed"


if __name__ == "__main__":
    main()
