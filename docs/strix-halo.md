# Bringing up the Strix Halo

What to do on the first day with the box's machine, an AMD Ryzen AI Max+ 395 (Strix Halo) with
128 GB of unified memory, so that the day is spent measuring, not installing. Its GPU, the Radeon
8060S, is RDNA 3.5, `gfx1151`: leat's kernels take it as RDNA 3 (`leat/kernels/common.py`, of
tinygrad's `target`, major 11), WMMA tensor cores too. So far they have run on tinygrad's emulated
RDNA 3 alone. Items marked **verify** are what this machine has to confirm.

## Firmware

- **UMA frame buffer**: the least the firmware offers, 512 MB or so. The GPU takes the rest of the
  memory as GTT, system memory it maps, rather than as a carve-out the CPU never gets back.
- **fTPM** on: the box's encrypted partitions are sealed to it, and it registers itself of its
  endorsement key (`python -m leat.box.enroll --fingerprint`). **Verify** that the full exchange
  works on AMD's firmware TPM, as on the Dell's Intel one.

## Kernel

The image's Arch kernel (7.x) and `linux-firmware` know `gfx1151`. How much of the memory the GPU
may map is TTM's limit, in 4 KiB pages, set on the kernel's command line:

| Parameter | What | For 120 GiB |
|---|---|---|
| `ttm.pages_limit` | pages TTM may allocate in all, GTT's ceiling | `31457280` |
| `ttm.page_pool_size` | pages TTM keeps pooled | `31457280` |

`amdgpu.gttsize` is the older way, deprecated. **Verify** what the machine reports before and
after, in `/sys/class/drm/card*/device/mem_info_gtt_total`, and that tinygrad allocates a 30 GB
model's weights. tinygrad's AMD backend goes through KFD, `/dev/kfd`: the user leat serve runs as
needs the `render` group.

## The kernels' compiler

tinygrad compiles the kernels' HIP C with ROCm's comgr, Arch's `comgr`. It depends on `rocm-llvm`,
4.7 GB installed, more than half a box's 8 GB system partition, and in every update. Arch's own
clang compiles the same C for `gfx1151` (`clang -x hip --offload-arch=gfx1151`, tried with clang
22), as the emulated tests already do ([tests/hip.py](../tests/hip.py)): `clang`, `lld`,
`llvm-libs` and `compiler-rt`, with `gcc` as clang's dependency, take about 0.7 GB. So the box
should compile with clang, its unaligned-access flag left out, which only the emulator needs; this
moves into `leat/` once the machine confirms the kernels run and as fast. For the first day,
install `comgr` too, to compare.

## The first day

```bash
pacman -S uv clang lld comgr vulkan-radeon     # and the image's own packages, as box/image/mkosi.conf
git clone … leat && cd leat && uv sync
DEV=AMD uv run python -c "from tinygrad import Device; print(Device['AMD'].arch)"   # gfx1151
DEV=AMD uv run pytest -m gpu                     # every GPU test, on the real GPU
DEV=AMD uv run python scripts/validate.py /models --llama-cpp llama.cpp/build/bin
```

`scripts/validate.py` writes its report to `reports/`: the GPU tests, speed against llama.cpp on
the same files, one sequence and batched, speculative decoding over chat prompts, and the server
answering a streamed chat. Its numbers go into the README's measurements beside the 3090's (LEA-93).

llama.cpp to compare against, at the commit `.github/workflows/ci.yml` pins, of each backend:

```bash
cmake -B build-vulkan -DGGML_VULKAN=ON && cmake --build build-vulkan -j
cmake -B build-rocm -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 && cmake --build build-rocm -j
```

The models to bring on an SSD, as their files are tens of GB: the chat candidates in Q4_K_M or
MXFP4, each with its mmproj and its MTP or assistant drafter, and EmbeddingGemma 2 in Q8_0.
