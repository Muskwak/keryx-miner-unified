# Keryx Miner

A high-performance miner for **Keryx**, combining GPU PoW (kHeavyHash) with on-chain AI inference (OPoI — Optimistic Proof of Inference).

---

## Precompiled Binaries

Download the latest release from the [Releases page](https://github.com/Keryx-Labs/keryx-miner/releases).

---

## Build from Source

### Standard build (PoW only, no inference)

Requires: Rust + Cargo ([rustup.rs](https://rustup.rs/)), `protoc` (`protobuf-compiler`)

```bash
git clone https://github.com/Keryx-Labs/keryx-miner.git
cd keryx-miner
cargo build --release --bin keryx-miner
```

Binary: `target/release/keryx-miner`

---

### CUDA build (PoW + GPU inference)

The inference engine (candle) builds with the **CUDA 12.x** toolkit. We recommend **CUDA 12.2**: nvcc 12.2 emits kernels that JIT on **NVIDIA driver ≥ 535**, whereas 12.6 needs driver ≥ 560. Building with 12.2 runs on the widest range of hosts and mining rigs (HiveOS commonly ships driver 535.x) at no performance cost.

#### Option A — CUDA 12.2 toolkit installed on host (recommended)

Install the toolkit side-by-side (runfile, toolkit-only, no driver), then point the build at it:

```bash
# one-time: install the CUDA 12.2 toolkit to ~/cuda-12.2 (no driver, no root needed)
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
bash cuda_12.2.2_535.104.05_linux.run --silent --toolkit --toolkitpath="$HOME/cuda-12.2" --override

cd keryx-miner
CUDA_COMPUTE_CAP=86 \
  CUDA_ROOT="$HOME/cuda-12.2" CUDA_PATH="$HOME/cuda-12.2" \
  PATH="$HOME/cuda-12.2/bin:$PATH" \
  cargo build --release --bin keryx-miner
```

Binary: `target/release/keryx-miner`

> Compiling with CUDA 12.2 requires **GCC ≤ 12** (Ubuntu 22.04 / GCC 11 works out of the box). On newer hosts use Option B.

#### Option B — CUDA 13.x or incompatible gcc on host (build via container)

If your system has CUDA 13.x or gcc 13+ (e.g. Fedora 40+, Ubuntu 25+), build inside a CUDA 12.2 container. The binary runs on the host via driver forward-compatibility.

Requires: [Podman](https://podman.io/) (rootless) or Docker, NVIDIA driver ≥ 535.

```bash
cd keryx-miner
podman run --rm --security-opt label=disable \
  -v "$PWD":/src -w /src \
  -e CUDA_COMPUTE_CAP=86 \
  -e CARGO_TARGET_DIR=/src/target-cuda \
  docker.io/nvidia/cuda:12.2.2-devel-ubuntu22.04 \
  bash -c '
    apt-get update -qq && apt-get install -y -qq \
      curl build-essential pkg-config libssl-dev ca-certificates protobuf-compiler >/dev/null 2>&1
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    . "$HOME/.cargo/env"
    export CUDA_PATH=/usr/local/cuda PROTOC=/usr/bin/protoc
    cargo build --release --bin keryx-miner'
```

Binary: `target-cuda/release/keryx-miner`

> **Always pass `-e CUDA_COMPUTE_CAP`.** The container does **not** inherit your host shell env, so you must set the compute cap with `-e` (as above). If you omit it, `candle-kernels` auto-detects the installed GPU and a Blackwell card resolves to `100` — which nvcc 12.2 rejects (`nvcc cannot target gpu arch 100`). On a 5090, set `-e CUDA_COMPUTE_CAP=89` (not `100`). If a previous run already cached the wrong value, clear the build dir first: `rm -rf target-cuda`.

> **Runtime dependencies.** PoW needs only `libcuda.so.1` (the driver). GPU **inference** additionally `dlopen`s `libcublas.so.12` and `libcurand.so.10` at runtime, so the host must have the matching CUDA 12.2 runtime libs (`libcublas-12-2`, `libcurand-12-2`). On HiveOS the miner installs and registers them automatically on first run; on other hosts install them via your package manager or the CUDA 12.2 toolkit.

**CUDA_COMPUTE_CAP by GPU generation:**

| GPU generation | Compute cap |
|----------------|-------------|
| RTX 30xx (Ampere) | `86` |
| RTX 40xx (Ada Lovelace) | `89` |
| RTX 50xx (Blackwell) | `89` |

> **Blackwell (RTX 50xx) note.** The CUDA 12.2 toolkit cannot emit native `sm_100`/`sm_120` SASS (that needs CUDA ≥ 12.8), so do **not** set `CUDA_COMPUTE_CAP=100` with Option A/B — the build will fail. Use `89`: the `sm_89` PTX JIT-forwards to Blackwell at runtime via the driver, at no performance cost for these kernels. A native `sm_120` build would require a CUDA ≥ 12.8 toolchain and is currently untested.

---

## Usage

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS
```

### Inference tiers (OPoI)

| Flag | Models supported | Min VRAM |
|------|-----------------|----------|
| *(none)* | TinyLlama 1.1B + DeepSeek-R1-8B | 8 GB |
| `--light` | TinyLlama 1.1B only | 4 GB |
| `--high` | TinyLlama 1.1B + DeepSeek-R1-8B + DeepSeek-R1-32B | 24 GB |
| `--very-high` | All 4 models (+ LLaMA-3.3-70B) | 32 GB |

Models are loaded **on demand** when a request arrives and cached between requests. Mining pauses during inference, then resumes automatically.

To run without inference (PoW only):

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS --no-opoi
```

### All options

```bash
./keryx-miner --help
```

---

## Connect

* **Website:** [keryx-labs.com](https://keryx-labs.com)
* **X (Twitter):** [@Keryx_Labs](https://x.com/Keryx_Labs)
* **Discord:** [Join the Community](https://discord.gg/U9eDmBUKTF)

---

> "Intelligence is the message. Keryx is the messenger."

---

## Dev Fund

2% of mining rewards support development by default.

```bash
--devfund-percent XX.YY
```
