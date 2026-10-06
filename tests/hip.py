"""Kernels for tinygrad's emulated AMD GPU, DEV=MOCK+AMD, compiled by the system's clang.

tinygrad renders C for AMD GPUs and compiles it with ROCm's comgr; without ROCm, its AMD device
falls back to rendering LLVM IR, which the kernels' inline intrinsics are not. This compiles that C
with clang's own HIP support instead, which needs clang with the AMDGPU target, as most distros'
is, and lld on PATH: a shell script named lld that runs `python -m ziglang ld.lld` without its
first two arguments, `-flavor gnu`, will do. The emulator itself is tinygrad's, in the tests of
its source tree, which PYTHONPATH must hold: the commit pyproject.toml pins.

    PYTHONPATH=path/to/tinygrad DEV=MOCK+AMD uv run pytest tests/test_kernels.py -k matvec
"""

import pathlib
import re
import subprocess
import tempfile

# what ROCm's device libraries would give tinygrad's prelude, as clang's builtins
PRELUDE = """
#define __ockl_get_local_id(d) ((d) == 0 ? __builtin_amdgcn_workitem_id_x() : \\
    (d) == 1 ? __builtin_amdgcn_workitem_id_y() : __builtin_amdgcn_workitem_id_z())
#define __ockl_get_group_id(d) ((d) == 0 ? __builtin_amdgcn_workgroup_id_x() : \\
    (d) == 1 ? __builtin_amdgcn_workgroup_id_y() : __builtin_amdgcn_workgroup_id_z())
#define __ocml_exp2_f32 __builtin_amdgcn_exp2f
#define __ocml_log2_f32 __builtin_amdgcn_logf
#define __ocml_sqrt_f32 __builtin_sqrtf
#define __ocml_trunc_f32 __builtin_truncf
"""


def install() -> None:
    """Makes tinygrad's HIP renderer compile with clang, so that its AMD device renders C."""
    import tinygrad.runtime.support.compiler_amd as compilers
    from tinygrad.device import CompileError, Compiler
    from tinygrad.renderer.cstyle import HIPRenderer
    from tinygrad.uop.ops import Ops

    class ClangHIPCompiler(Compiler):
        def __init__(self, arch: str):
            self.arch = arch
            super().__init__(f"compile_clang_hip_aligned_{arch}")  # a cache key of these flags

        def compile(self, src: str) -> bytes:
            # the device libraries' declarations, which PRELUDE's macros stand in for
            src = re.sub(
                r'extern "C" __attribute__\(\(device[^)]*\)\) [^;]*__o(?:ckl|cml)_[^;]*;\n', "", src
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp)
                out, source = path / "kernel.hsaco", path / "kernel.hip"
                # without unaligned access, which GPUs allow and the emulator does not: clang
                # would merge the halfword loads of blocks only halfword aligned into wide ones
                args = ["clang", "-x", "hip", "--cuda-device-only", "--no-gpu-bundle-output",
                        "-nogpulib", "-nogpuinc", f"--offload-arch={self.arch}", "-O3",
                        "-std=c++17", "-mcumode", "-Xclang", "-target-feature", "-Xclang",
                        "-unaligned-access-mode", "-o", str(out), str(source)]  # fmt: skip
                source.write_text(PRELUDE + src)
                done = subprocess.run(args, capture_output=True, text=True)
                if done.returncode:
                    raise CompileError(done.stderr)
                return out.read_bytes()

    compilers.HIPCompiler = ClangHIPCompiler  # type: ignore[misc]
    # the emulator's sines are 0, as RoPE's tables would be: tinygrad works them out of other ops
    HIPRenderer.code_for_op = {k: v for k, v in HIPRenderer.code_for_op.items() if k is not Ops.SIN}
