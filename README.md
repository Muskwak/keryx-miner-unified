# Keryx Miner — Unified

> One codebase, one binary per OS, that mines [Keryx](https://keryx-labs.com) PoM on **NVIDIA (CUDA)**,
> **AMD / Intel (Vulkan)**, and **Apple Silicon (Metal)** GPUs — including a heterogeneous rig
> (e.g. one NVIDIA + one AMD card in the same machine) mined simultaneously from a single process.

This is a **unified fork** that merges three independently-evolved forks into one abstraction:

| Source fork | What it contributed |
|---|---|
| [`keryx-metal`](https://github.com/Muskwak/keryx-miner-metal) | Base — Apple Silicon (Metal), iOS, Android (Vulkan), the right `pom_gpu` module-aliasing shape |
| [`keryx-pascal-miner`](https://github.com/Muskwak/keryx-pascal-miner) | The hand-tuned Pascal `sm_61` CUDA kernel (~9 MH/s on a P40 vs ~4.5 generic) + multi-arch PTX build |
| [`keryx-miner-rdna3`](https://github.com/Muskwak/keryx-miner-rdna3) | Desktop Vulkan backend (RDNA3) + in-process llama.cpp (Vulkan) inference engine for AMD/Intel |

Each GPU mining backend and each inference engine is formalized behind a trait
([`src/device.rs`](src/device.rs), [`src/inference_engine.rs`](src/inference_engine.rs)), so a single
binary holds several live backends at once and dispatches to whichever one actually talks to a given
device at runtime — no per-vendor fork, no "which vendor" flag.

See [`plan.md`](plan.md) for the full architecture and phased plan.

---

## Status

The **code-structural work** (plan phases 0–4: device model, trait refactor, CUDA/Vulkan/Metal backend
unification, additive Cargo features, inference-engine trait) is done and the `keryx-vulkan` crate
compiles clean.

The **hardware-validation phases** (5–7: heterogeneous-rig test, Intel Arc validation, cutover) are
**not** done — they require a real GPU matrix (NVIDIA + AMD + Intel Arc + Apple Silicon + Android).
Treat every backend's runtime behaviour as unverified until tested on real hardware. The prebuilt
release binaries compile but have not been validated against a live pool/node or mining hardware.

---

## Precompiled binaries

Download from the [Releases page](https://github.com/Muskwak/keryx-miner-unified/releases).

---

## Build from source

The desktop build (Linux/Windows) requires the **CUDA Toolkit** unconditionally (candle's CUDA feature
is hardcoded for desktop — there is no CPU-only build). The desktop Vulkan backend + in-process
llama.cpp engine are an **opt-in** addition (`--features vulkan`) for AMD/Intel cards. macOS/iOS build
Metal; Android builds Vulkan.

### One-click build scripts

```bash
# Linux — auto-detects the Vulkan SDK + cmake + libclang and builds the best available feature set
./build-linux.sh

# Windows — same auto-detect logic
build-windows.bat
```

Force a specific feature set:

```bash
./build-linux.sh cuda           # CUDA only (NVIDIA)
./build-linux.sh cuda,vulkan    # full heterogeneous-rig build (NVIDIA + AMD/Intel in one binary)
```

### Manual build

Requires: Rust + Cargo ([rustup.rs](https://rustup.rs/)), `protoc` (`protobuf-compiler`), and the
**CUDA 12.2 toolkit** (12.2 specifically: its nvcc emits kernels that JIT on NVIDIA driver ≥ 535,
whereas 12.6 needs driver ≥ 560 — 12.2 runs on the widest range of mining rigs at no perf cost).

```bash
git clone https://github.com/Muskwak/keryx-miner-unified.git
cd keryx-miner-unified

# CUDA-only desktop build (the default — matches the historical per-fork binary)
CUDA_COMPUTE_CAP=86 cargo build --release --bin keryx-miner

# Heterogeneous-rig desktop build (one binary mines NVIDIA + AMD simultaneously)
CUDA_COMPUTE_CAP=86 cargo build --release --features vulkan --bin keryx-miner
```

For CUDA 13.x or gcc 13+ hosts (incompatible with the 12.2 toolkit), build inside a CUDA 12.2
container — see the build scripts or use:

```bash
podman run --rm --security-opt label=disable \
  -v "$PWD":/src -w /src -e CUDA_COMPUTE_CAP=86 -e CARGO_TARGET_DIR=/src/target-cuda \
  docker.io/nvidia/cuda:12.2.2-devel-ubuntu22.04 \
  bash -c 'apt-get update -qq && apt-get install -y -qq curl build-essential pkg-config libssl-dev ca-certificates protobuf-compiler >/dev/null 2>&1
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    . "$HOME/.cargo/env"; export CUDA_PATH=/usr/local/cuda PROTOC=/usr/bin/protoc
    cargo build --release --bin keryx-miner'
```

**`CUDA_COMPUTE_CAP` by GPU generation:** RTX 30xx (Ampere) → `86`, RTX 40xx (Ada) → `89`,
RTX 50xx (Blackwell) → `89` (not `100` — the 12.2 toolkit can't emit native `sm_100`, the `sm_89` PTX
JIT-forwards at runtime). On a P40 / GTX 1070 (Pascal, `sm_61`) no env var is needed — the build
emits a tuned `sm_61` kernel and PTX for `sm_61` through `sm_90`, runtime-selected per device.

> **Runtime deps.** PoW needs only the NVIDIA driver (`libcuda.so.1` / `nvcuda.dll`). GPU **inference**
> additionally `dlopen`s `libcublas.so.12` and `libcurand.so.10`, so install `libcublas-12-2` and
> `libcurand-12-2` (the miner auto-installs + ldconfig-registers them on HiveOS). The Vulkan path
> needs only the Vulkan loader (`libvulkan.so.1` / `vulkan-1.dll`), shipped by every modern GPU driver.

### The Vulkan feature (AMD / Intel)

`--features vulkan` pulls the optional `keryx-vulkan` crate (desktop Vulkan compute backend) and the
in-process `llama-cpp-2` (Vulkan) inference engine. The build additionally needs `cmake`, `ninja`
(Windows only — the VS generator mishandles a llama.cpp subproject), `libclang` (for the FFI bindgen
step), and the Vulkan SDK's `glslc`. The one-click build scripts auto-detect all of these.

On a machine with only one vendor's GPU, a both-features build runs fine — the other backend's probe
just finds nothing and contributes zero devices.

---

## Usage

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS
```

### PoM inference tiers

Under Proof-of-Model, each tier mines **and** serves exactly one model. The flag is a *ceiling*: each
GPU is auto-assigned the highest tier ≤ the ceiling that its VRAM holds, so a mixed rig mines a
different tier per card instead of the lowest common denominator.

| Flag | Model | Min VRAM |
|------|-------|----------|
| `--very-light` | Qwen3-1.7B | 2 GB |
| `--light` | Gemma-3-4B | 5 GB |
| *(none / default)* | Dolphin-3.0-Llama-3.1-8B | 8 GB |
| `--high` | Qwen3-32B | 24 GB |
| `--very-high` | Llama-3.3-70B | 30 GB |

Models load on demand and are cached between requests. Mining pauses during inference, then resumes.

### All options

```bash
./keryx-miner --help
```

---

## Architecture (one-pager)

* **Device model** — `GpuHandle { backend, index }` replaces every legacy bare `u32 device_id`.
  Vendor → backend routing: NVIDIA → CUDA, AMD/Intel → Vulkan, Apple → Metal.
* **Additive backends** — `cuda` / `vulkan` / `metal` Cargo features. Desktop Windows/Linux can
  compile `cuda` + `vulkan` together (`--features cuda,vulkan`) so one binary mines a heterogeneous
  rig. macOS/iOS compile `metal`; Android compiles `vulkan`.
* **Startup probe** — each compiled-in backend probes its own driver and contributes zero or more
  devices to a unified, backend-tagged list. The mining loop spawns one OS thread per device.
* **Inference engines** — candle (CUDA on NVIDIA, Metal on Apple) and llama.cpp+Vulkan (AMD/Intel/
  Android), picked once at startup by which backend serves the inference GPU. The zero-dup PoM walk
  reads each engine's own resident weights (no second VRAM copy).

Full detail in [`plan.md`](plan.md).

---

## Connect

* **Website:** [keryx-labs.com](https://keryx-labs.com)
* **X (Twitter):** [@Keryx_Labs](https://x.com/Keryx_Labs)
* **Discord:** [Join the Community](https://discord.gg/U9eDmBUKTF)

---

## Dev Fund

2% of mining rewards support development by default.

```bash
--devfund-percent XX.YY
```

---

> "Intelligence is the message. Keryx is the messenger."
