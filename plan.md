# Keryx Miner — Unified Cross-Platform Fork: Plan

## 0. Goal

One codebase, building **one binary per OS**, that:

- Runs on Windows, Linux, macOS, iOS, Android.
- Mines PoM on NVIDIA (CUDA), AMD/Intel (Vulkan), and Apple Silicon (Metal) GPUs.
- Auto-detects the platform and every installed GPU at startup, and picks the right backend
  per device with no user-facing flags for "which vendor."
- Mines a **heterogeneous rig** (e.g. one NVIDIA + one AMD card in the same machine)
  simultaneously, each device driven by whichever backend actually talks to it.
- Serves OPoI inference via whichever engine matches the machine's designated inference GPU.

This is not a small refactor. It is a merge of three independently-evolved forks (each of
which itself diverged from and periodically re-synced with upstream) into one abstraction.
Treat this as a multi-phase project, not a single PR.

---

## 1. Current State — what exists today (audited this session)

| Fork | Platforms | GPU backend | Inference engine | Multi-tier / heterogeneous rig | Notes |
|---|---|---|---|---|---|
| **Keryx-Labs/keryx-miner** (upstream) | Windows, Linux | CUDA only | candle (CUDA) | Yes — per-GPU tier assignment by VRAM, OOM banlist + auto-downgrade | The trunk every fork re-syncs from |
| **keryx-pascal-miner** (Muskwak) | Windows, Linux | CUDA only, but with a hand-tuned Pascal (sm_61) kernel — compiled to PTX for sm_61/70/75/80/86/89/90, runtime-selected by compute capability | candle (CUDA) | Yes, resynced with upstream this session | Verified 9 MH/s on a real P40 (vs ~4.5 generic). v0.3.5 released this session (Windows + Linux binaries) |
| **keryx-miner-rdna3** (Muskwak) | Windows, Linux | Vulkan only (`ash`), replaced CUDA/candle entirely | **llama.cpp FFI (in-process, Vulkan backend)** — not candle | Yes — just ported upstream's CUDA multi-tier/OOM-banlist design to this Vulkan+llama.cpp architecture this session | Also has an unshipped WIP fast-modulo shader experiment. Build verification (llama.cpp CMake chain) in progress as of this doc |
| **keryx-metal** (Muskwak, "keryx-miner-metal") | macOS, iOS, Android, **+ unmodified stock CUDA path for Windows/Linux** | Metal (macOS/iOS) + Vulkan (Android, new this session) + CUDA (desktop, stock/untouched) | candle (Metal on Apple; CPU-only GGUF parse on Android, no GPU inference) | Not yet audited against latest upstream — likely behind on H2 multi-tier | **Already has the right shape**: `pom_gpu` is aliased via `#[path=...]` per-target to `pom_gpu.rs` (CUDA) / `pom_gpu_metal.rs` / `pom_gpu_vulkan.rs`, all exposing the same free-function API (`install`/`uninstall`/`is_installed`/`mine`/`ensure_installed`/`current_tier`/`set_mining_tier`/`device_for_model`) |

**Key insight:** `keryx-metal`'s existing `pom_gpu` module-aliasing pattern is *already* the right
interface shape for backend abstraction — it just needs to become a runtime choice instead of a
compile-time one, and needs the CUDA/Vulkan desktop paths upgraded to match what the other two
forks already built and validated.

### Backend-specific technical assets worth preserving, per fork

- **CUDA (pascal-miner)**: `cuda/pom_mine.cu` tuned kernel (2×128-bit `longlong2` `__ldg` gather,
  u32 prefix table, predicated `find_tensor`, Lemire magic-modulo `fast_mod`, 128 threads/block +
  shared-memory prefix), multi-arch PTX fat-binary build.rs, `get_gpu_ptx_by_id`/`get_pom_ptx`
  runtime-capability dispatch.
- **Vulkan desktop (rdna3)**: `keryx-vulkan` crate (`ash`-based), zero-dup streamed weight upload
  (`PomWalkGpu::new_streamed`, no packed host copy), `PomWalkShared` zero-dup walk over an
  in-process llama.cpp engine's resident buffers, `Resident::{Blob,Shared}` enum.
- **Vulkan mobile (keryx-metal/Android)**: the same `keryx-vulkan` foundation, trimmed for mobile
  and extended with a **dual shader path** — native `uint64_t` (`pom_walk.comp`) when the device
  supports `shaderInt64`, and a from-scratch `uvec2`-emulated fallback (`pom_walk_i32.comp`) with
  a **Barrett-reduction fast modulo** when it doesn't (proven necessary and shipped this session —
  Adreno GPUs, confirmed via direct device query, do not implement `shaderInt64` despite Vulkan 1.3
  conformance). This dual-path pattern is directly reusable for desktop Intel Arc, which has
  historically had gaps in obscure-but-optional Vulkan features too.
- **Metal (keryx-metal/macOS+iOS)**: zero-dup walk directly over candle's own resident
  `MTLBuffer`s via a vendored `candle-core` patch exposing `QTensor::metal_storage`.
- **Bit-exactness validation methodology** (proven this session, reusable for every new backend
  port): mirror the shader/kernel's arithmetic in a plain-Rust test using the same primitive types,
  cross-check against the canonical `pom.rs` reference over edge cases + hundreds of thousands of
  randomized inputs + an exhaustive small-parameter sweep, before trusting a kernel rewrite.

---

## 2. Target Architecture

### 2.1 Device model

Replace every bare `u32 device_id` (used today as a CUDA ordinal, a Vulkan enumeration index, *or*
a Metal device index depending on which fork you're in — never more than one meaning per binary)
with a backend-qualified handle:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend { Cuda, Vulkan, Metal }

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuHandle {
    pub backend: Backend,
    pub index: u32,   // the backend's own native ordinal/enumeration index
}
```

Every place that currently keys a `HashMap<u32, _>` (`MINING_TIERS`, `MINERS`, `OOM_BANLIST`,
per-worker hashrate maps, `device_for_model`) gets keyed by `GpuHandle` instead. This is the
single largest mechanical change and touches `pom_gpu*.rs`, `miner.rs`, and `main.rs`'s tier
assignment in every fork.

### 2.2 Backend trait

Today each fork implements the *same* free-function API independently
(`pom_gpu.rs`/`pom_gpu_metal.rs`/`pom_gpu_vulkan.rs`). Formalize it as a trait so a single binary
can hold multiple live implementations at once:

```rust
trait PomGpuBackend: Send + Sync {
    fn enumerate(&self) -> Vec<GpuDeviceInfo>;      // index, name, vram_mb, vendor
    fn ensure_installed(&self, device: u32, daa: u64) -> bool;
    fn mine(&self, device: u32, pph: &[u8;32], ts: u64, target: &[u8;32], start: u64, batch: u64) -> Option<u64>;
    fn uninstall(&self, device: u32);
    fn resident_bytes(&self, device: u32) -> u64;
    // ...current_tier / set_mining_tier / device_for_model already device-keyed, unchanged shape
}
```

At startup, probe every backend **compiled into this binary** (not every backend that exists —
see §2.4): CUDA via `cudarc`'s driver init (fails cleanly if no NVIDIA driver), Vulkan via `ash`
(fails cleanly if no Vulkan loader), Metal via `objc2-metal` (macOS/iOS only, always present
there). Build one flat, backend-tagged device list from whichever probes succeed. A machine with
an NVIDIA card and an AMD card gets two entries, one per backend, and mines both.

### 2.3 GPU-vendor → backend routing

| Vendor | Backend chosen |
|---|---|
| NVIDIA | CUDA (kept — it's the most tuned path, and candle's CUDA inference is mature) |
| AMD | Vulkan |
| Intel (Arc/integrated) | Vulkan — **unverified**, no hardware tested yet in any fork; treat as a real risk, not an assumption |
| Apple Silicon | Metal |

NVIDIA cards *can* also run Vulkan, but CUDA stays preferred there since it's the more mature,
more deeply tuned path (Pascal-specific kernel, existing candle-CUDA inference). Don't route
NVIDIA to Vulkan unless CUDA genuinely isn't available (e.g. driver present but CUDA toolkit
runtime missing — decide the exact fallback rule during Phase 1).

### 2.4 Build-time backend inclusion (this is what makes "one binary per OS" possible)

Backends are **not mutually exclusive** the way they are in each fork today. Cargo features,
additive per OS:

| OS | Features compiled in | Why |
|---|---|---|
| Windows | `cuda`, `vulkan` | Covers NVIDIA and AMD/Intel desktop cards in one binary |
| Linux | `cuda`, `vulkan` | Same |
| macOS | `metal` | Vulkan-via-MoltenVK is possible but adds nothing Metal doesn't already cover — out of scope |
| iOS | `metal` | Only backend Apple allows anyway |
| Android | `vulkan` | Only backend that exists there |

`build.rs` conditionally invokes `nvcc` (if `cuda` feature) and/or `glslc` (if `vulkan` feature),
embedding whichever kernel blobs apply via `include_bytes!`, exactly as each fork already does
individually. Neither toolchain is needed at *runtime* — CUDA only needs the NVIDIA driver
(`nvcuda.dll`/`libcuda.so.1`), Vulkan only needs the loader (`vulkan-1.dll`/`libvulkan.so.1`,
shipped by essentially every modern GPU driver including NVIDIA's own) — so a Windows/Linux
binary with both features compiled in runs fine on a machine with only one vendor's GPU; the
other backend's probe just finds nothing and contributes zero devices.

### 2.5 Inference engine unification

This is the second major axis of divergence, independent of the mining-kernel backend:

| Inference-serving GPU | Engine |
|---|---|
| NVIDIA | candle (CUDA) — unchanged from upstream/pascal-miner |
| Apple Silicon | candle (Metal) — unchanged from keryx-metal |
| AMD / Intel / Android | llama.cpp FFI (Vulkan) — the engine keryx-miner-rdna3 already built and validated |

Needs its own small trait (`InferenceEngine::load`/`generate`/`build_shared_walk` for zero-dup),
picked once at startup based on which backend serves `inference_device_index()`. Note the
zero-dup mechanism differs by engine (`QTensor::metal_storage` patch for Metal/candle,
`PomWalkShared` for llama.cpp/Vulkan, `load_shared` split-loader for candle/CUDA) — this is real,
non-mechanical porting work per pairing, not a single shared code path.

### 2.6 Heterogeneous-rig mining loop

Each fork's `miner.rs` already spawns one OS thread per device — that part generalizes cleanly.
The per-device tier-assignment logic (`assign_pom_tiers`, ported from CUDA to Vulkan this
session for the rdna3 fork) needs one more generalization pass: iterate the *unified* device
list (§2.1) instead of one backend's `enumerate_devices()`, and the "which model does this
device mine" map is keyed by `GpuHandle`.

---

## 3. Repo Strategy

**Recommendation: use `keryx-metal` as the base**, not a fresh clone of upstream. It already:

- Has the right abstraction shape (`pom_gpu` swapped per target via `#[path]`).
- Covers the broadest platform surface today (macOS, iOS, Android already work).
- Has the mobile app layers (SwiftUI, Kotlin/Compose) already built and JNI/C-ABI bridges proven.

What it's missing relative to the other two forks, to be ported in:

1. **From `keryx-pascal-miner`**: the Pascal-tuned CUDA kernel + multi-arch PTX build, and
   confirm/port the H2 multi-tier + OOM-banlist architecture (audit whether keryx-metal's
   `pom_gpu.rs` CUDA path is current with upstream at all — likely stale, needs a real
   upstream-sync pass first, same as was just done for the other two forks this session).
2. **From `keryx-miner-rdna3`**: the desktop Vulkan backend (today keryx-metal only has *mobile*
   Vulkan) and the llama.cpp+Vulkan inference engine for AMD/Intel desktop.

This order (metal as base, CUDA and Vulkan-desktop ported in) is less total work than starting
from upstream and re-deriving Apple/mobile support from scratch, which took the bulk of a prior
multi-session effort to get right the first time.

---

## 4. Phased Plan

**Phase 0 — Audit & freeze.** Diff `keryx-metal`'s CUDA path against current upstream and against
`keryx-pascal-miner`'s post-sync state; diff its module layout precisely against `rdna3`'s Vulkan
crate. Produce an exact list of divergences before writing new code. (This session's work on the
other two forks already produced most of the upstream-diff methodology and conflict-resolution
judgment calls to reuse here.)

**Phase 1 — Device model + trait refactor.** Introduce `GpuHandle`/`Backend`, convert
`pom_gpu.rs`/`pom_gpu_metal.rs`/`pom_gpu_vulkan.rs` into trait implementations behind one
dispatcher, without yet changing which backends are compiled in per OS (i.e. keep today's
`#[cfg]` exclusivity for now — prove the trait refactor is behavior-preserving before making
backends additive).

**Phase 2 — Desktop Vulkan port + CUDA re-sync.** Port `keryx-vulkan`'s desktop backend and the
Pascal-tuned CUDA kernel into keryx-metal side by side (still one-active-backend-per-build at
this point, just both now *present* in the codebase, selected by Cargo feature).

**Phase 3 — Make backends additive on desktop.** Flip Windows/Linux builds to compile `cuda` +
`vulkan` together; implement the startup multi-backend probe and unified device enumeration;
verify a synthetic "both backends present" build runs correctly with only one vendor's GPU
actually installed (the other backend's probe returning zero devices, no crash).

**Phase 4 — Inference engine unification.** Add the llama.cpp+Vulkan engine option for
AMD/Intel/Android inference-serving devices; keep candle-CUDA and candle-Metal as-is.

**Phase 5 — Heterogeneous rig validation.** Real hardware test: one machine, NVIDIA + AMD cards
both installed, confirm both mine simultaneously, confirm tier assignment and OOM-downgrade work
across the mixed device list.

**Phase 6 — Intel Arc validation.** The one vendor with zero prior validation in any fork. Vulkan
*should* work per spec, but expect at least one Adreno-style surprise (missing optional feature,
different subgroup size) — budget real debugging time here, don't assume it just works.

**Phase 7 — Cutover.** Once parity is confirmed against all three source forks (same hashrate
ballpark, same correctness guarantees, bit-exactness re-proven for every ported kernel), point
the other three forks' READMEs at the unified repo and stop maintaining them independently.

---

## 5. Testing & Validation Strategy

- **Bit-exactness, every backend, before trusting it**: reuse this session's proven method —
  mirror the kernel/shader arithmetic in plain Rust, cross-check against `pom.rs`'s native
  reference over edge cases + large randomized batches + an exhaustive small-parameter sweep.
  This already caught real issues (the Adreno `shaderInt64` gap, the correct Barrett-reduction
  correction-loop bound) and should gate every new kernel port the same way.
- **CI compiles the full OS × feature matrix** (Windows cuda+vulkan, Linux cuda+vulkan, macOS
  metal, iOS metal, Android vulkan) on every push — this validates linking/build correctness but
  **not** runtime GPU behavior.
- **Real hardware is mandatory for anything beyond compilation.** This session's own experience:
  CI-green does not mean "mines correctly" or "mines fast" — the Adreno `shaderInt64` gap and the
  0.06 MH/s-then-fixed modulo regression were both only discoverable by testing on the actual
  device. Plan for a real device matrix: at least one NVIDIA card, one AMD card, one Intel Arc
  card, one Apple Silicon Mac, one Android phone (Adreno and, ideally, Mali) before calling any
  platform "done."

---

## 6. Open Questions (need a decision before or during Phase 1, not assumed here)

1. **Repo/org for the unified fork** — new repo, or does one of the three existing forks get
   promoted in place? (This plan assumes a new repo seeded from `keryx-metal`'s history.)
2. **NVIDIA-but-no-CUDA-runtime fallback** — if a machine has an NVIDIA card but CUDA isn't
   usable for some reason (driver mismatch, etc.), does it fall back to mining that card via
   Vulkan, or just skip it? Affects how "strict" the vendor-routing table in §2.3 should be.
3. **Keep the legacy CPU kHeavyHash pre-PoM path at all?** PoM has been active on mainnet for a
   while now across all forks; carrying three GPU backends *and* three inference engines is
   already a lot of surface area — dropping the dead pre-PoM CPU path would shrink scope
   meaningfully and is worth considering as part of this consolidation, not just left as-is.
4. **Does llama.cpp+Vulkan become the default engine on desktop AMD/Intel**, replacing candle
   there too even though candle *does* have a path forward on those vendors in principle? (This
   plan assumes yes, since it's the one already validated by the rdna3 fork.)
5. **Intel Arc is unvalidated everywhere.** Treat every claim about it in this plan as a
   hypothesis, not a fact, until real hardware says otherwise.

---

## 7. Non-Goals (explicitly out of scope for this plan)

- Windows ARM64, Linux ARM64 desktop — not requested, not audited.
- Any GPU vendor beyond NVIDIA/AMD/Intel/Apple (no mobile Adreno-via-desktop, no ROCm-specific
  path distinct from Vulkan, etc.).
- OPoI `mining.challenge` answering on mobile — already a separately-tracked, deferred item on
  both iOS and Android; unrelated to this unification effort.
