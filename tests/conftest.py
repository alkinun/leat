import os
from pathlib import Path

import pytest

from tests.helpers import write_tiny_llama

GPU_BACKENDS = {"NV", "CUDA", "AMD"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # GPU and real-model tests are opt-in so the default run stays hermetic and runs on any CPU.
    has_gpu = os.environ.get("DEV", "").split(":")[0] in GPU_BACKENDS
    has_model = bool(os.environ.get("LEAT_MODEL"))
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(pytest.mark.skip(reason="needs DEV=NV, CUDA or AMD"))
        if "model" in item.keywords and not has_model:
            item.add_marker(pytest.mark.skip(reason="needs LEAT_MODEL=path.gguf"))


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> tuple[Path, dict]:
    # a random llama GGUF and its weights decoded independently by gguf-py
    path = tmp_path_factory.mktemp("model") / "tiny.gguf"
    return path, write_tiny_llama(path)


@pytest.fixture(scope="session")
def model_path() -> Path:
    return Path(os.environ["LEAT_MODEL"]).expanduser()


@pytest.fixture(scope="session")
def llama_cpp() -> Path:
    # llama.cpp's build/bin directory, used as the reference implementation
    if not (path := os.environ.get("LLAMA_CPP")):
        pytest.skip("needs LLAMA_CPP=path/to/llama.cpp/build/bin")
    return Path(path).expanduser()


@pytest.fixture(scope="session")
def wikitext() -> Path:
    # wikitext-2's wiki.test.raw, the usual text for perplexity
    if not (path := os.environ.get("WIKITEXT")):
        pytest.skip("needs WIKITEXT=path/to/wiki.test.raw")
    return Path(path).expanduser()
