import gc
import os
import resource
from collections.abc import Callable
from pathlib import Path

import pytest

# tinygrad would pick a GPU when it finds one: the default run stays on the CPU on any machine.
# It reads DEV when imported, so this comes first.
os.environ.setdefault("DEV", "CPU")
GPU_BACKENDS = {"NV", "CUDA", "AMD", "MOCK+AMD"}
if worker := os.environ.get("PYTEST_XDIST_WORKER"):
    # each of pytest-xdist's workers its own cache of compiled kernels, beside tinygrad's: workers
    # sharing one crash now and then
    cache = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "tinygrad"
    os.environ.setdefault("CACHEDB", str(cache / f"cache-{worker}.db"))
    # and its kernels lowered and compiled in its own process, as the workers are the parallelism:
    # tinygrad's pool of processes beneath each, of the cores over the workers, is of one process
    # on 4 cores and 4 workers, as CI's, and hangs there now and then, worker and pool each
    # waiting on the other
    os.environ.setdefault("PARALLEL", "0")
else:
    # a stack for the workers xdist starts, which take this process's limit, deep enough for
    # tinygrad's rewrites of a model's whole graph, which recurse past the usual 8 MB, as the pool's
    # processes did them
    stack, most = resource.getrlimit(resource.RLIMIT_STACK)
    if stack != resource.RLIM_INFINITY and stack < 64 << 20:
        wanted = 64 << 20 if most == resource.RLIM_INFINITY else min(64 << 20, most)
        resource.setrlimit(resource.RLIMIT_STACK, (wanted, most))
if os.environ["DEV"].split(":")[0] == "MOCK+AMD":
    # tinygrad's emulated RDNA 3 GPU, whose kernels the system's clang compiles: see tests/hip.py.
    # Compiles run in this process, where the compiler is replaced, not in a pool of workers.
    os.environ.setdefault("PARALLEL", "0")
    from tests import hip

    hip.install()


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
            item.add_marker(pytest.mark.skip(reason="needs DEV=NV, CUDA, AMD or MOCK+AMD"))
        if "model" in item.keywords and not has_model:
            item.add_marker(pytest.mark.skip(reason="needs LEAT_MODEL=path.gguf"))


@pytest.fixture
def reference_ops(monkeypatch):
    # the plain tinygrad ops alone, for exact comparisons with the f64 references: the kernels
    # quantize activations to int8, and on the GPU the logits of up to 8 positions take them
    monkeypatch.setenv("LEAT_KERNELS", "ref")


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
def tiny_assistant(tmp_path_factory) -> tuple[Path, dict]:
    # a random Gemma 4 assistant, which drafts for tiny("gemma4"), and its weights
    from tests.helpers import write_tiny_assistant

    path = tmp_path_factory.mktemp("assistant") / "assistant.gguf"
    return path, write_tiny_assistant(path)


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


@pytest.fixture
def engine():
    # leat agent's tests' fake of leat serve, scripted reply by reply: see tests/agents.py
    import threading

    from tests.agents import FakeEngine

    engine = FakeEngine()
    threading.Thread(target=engine.serve_forever, args=(0.01,), daemon=True).start()
    yield engine
    engine.released.set()
    engine.shutdown()
    engine.server_close()


@pytest.fixture
def server(agent):
    # leat agent's server, of the test's agent, its owner set up: see tests/agents.py
    from tests.agents import serving

    with serving(agent) as url:
        yield url


@pytest.fixture
def workspace(tmp_path):
    # a workspace of leat agent's, its sandbox of LEAT_SANDBOX's environment, if given
    from leat.agent.workspace import Workspace
    from tests.agents import SANDBOX

    return Workspace(tmp_path / "workspace", Path(SANDBOX) if SANDBOX else None)
