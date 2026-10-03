import os
from pathlib import Path

import pytest

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
def model_path() -> Path:
    return Path(os.environ["LEAT_MODEL"]).expanduser()


@pytest.fixture(scope="session")
def llama_cpp() -> Path:
    # llama.cpp's build/bin directory, used as the reference implementation
    if not (path := os.environ.get("LLAMA_CPP")):
        pytest.skip("needs LLAMA_CPP=path/to/llama.cpp/build/bin")
    return Path(path).expanduser()
