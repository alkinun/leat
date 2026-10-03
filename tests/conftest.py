import gc
import os
from collections.abc import Callable
from pathlib import Path

import pytest

# tinygrad would pick a GPU when it finds one: the default run stays on the CPU on any machine.
# It reads DEV when imported, so this comes first.
os.environ.setdefault("DEV", "CPU")
GPU_BACKENDS = {"NV", "CUDA", "AMD"}


@pytest.fixture(autouse=True)
def _collect(request):
    # a real model fills most of a GPU: free one test's before the next test loads another
    yield
    if request.node.get_closest_marker("model"):
        gc.collect()


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
def tiny(tmp_path_factory) -> Callable[[str], tuple[Path, dict]]:
    # tiny(arch): a random GGUF of an architecture and its weights decoded independently by
    # gguf-py, written once per session
    from tests.helpers import write_tiny_model

    made: dict[str, tuple[Path, dict]] = {}

    def model(arch: str) -> tuple[Path, dict]:
        if arch not in made:
            path = tmp_path_factory.mktemp(arch) / "tiny.gguf"
            made[arch] = path, write_tiny_model(path, arch)
        return made[arch]

    return model


@pytest.fixture(scope="session")
def tiny_model(tiny) -> tuple[Path, dict]:
    return tiny("llama")


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
