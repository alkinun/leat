# leat

A minimal, fast LLM inference engine built on [tinygrad](https://github.com/tinygrad/tinygrad).

leat runs GGUF models with hand-written kernels that keep the weights quantized in memory. The goal is single-stream decode limited by memory bandwidth, not by the engine. It targets NVIDIA RTX 30-series GPUs first, then AMD Strix Halo.

> Status: early development. Nothing runs end to end yet.

## Requirements

- Linux, Python 3.12+, [uv](https://docs.astral.sh/uv/)
- `clang`: tinygrad compiles its GPU command submission with it
- NVIDIA: the open kernel module and NVRTC (from the CUDA toolkit); run with `DEV=NV`

## Development

```bash
uv sync
uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest
```

GPU and real-model tests are opt-in:

```bash
DEV=NV LEAT_MODEL=~/models/model.gguf uv run pytest
```

## License

MIT
