# Keryx Miner — Unified Fork: Implementation Notes

This repo implements the **code-structural** phases (0–4) of `plan.md` — the merge of the three
forks (`keryx-metal` base, `keryx-pascal-miner`'s CUDA kernel, `keryx-miner-rdna3`'s desktop Vulkan
backend + llama.cpp engine) into one abstraction that can hold multiple GPU backends in a single
binary. The hardware-validation phases (5–7) are out of scope for code work — they require a real
NVIDIA+AMD+Intel+Apple+Android device matrix and are called out at the bottom.

The original three fork repos are **untouched**; all changes live in this directory.

---

## What was done, by phase

### Phase 0 — Audit & freeze

The base (`keryx-metal`) had a half-finished `git merge --no-commit` of `upstream/main` in its
working tree — 12 files carried unresolved `<<<<<<<`/`=======`/`>>>>>>>` conflict markers (left over
from a prior session that never committed). Those markers were copied into this repo by the initial
`cp`. They were resolved **in favour of HEAD** (the keryx-metal side) by extracting each file's
`:2:` (ours) stage from keryx-metal's git index and writing it here, so this repo starts from
keryx-metal's actual working code, not a mix of both sides. No file in this repo contains conflict
markers.

### Phase 1 — Device model + trait refactor (plan §2.1, §2.2)

* **`src/device.rs`** (new) — `Backend` (Cuda/Vulkan/Metal), `GpuHandle { backend, index }` (the
  backend-qualified handle that replaces every legacy bare `u32 device_id`), `GpuDeviceInfo`,
  `GpuVendor`, and the `PomGpuBackend` trait. Formalizes the free-function surface every backend
  already exposes (`enumerate`/`query_all_gpus_vram`/`set_mining_tier`/`current_tier`/
  `device_for_model`/`is_installed`/`is_loading`/`ensure_installed`/`uninstall`/`mine`) so a single
  binary can hold several live backends and dispatch by `GpuHandle.backend`. Includes a process-wide
  registry (`register_backends` / `backends` / `sole_backend` / `unified_device_list`) and unit tests.
* **`src/pom_gpu_backends.rs`** (new) — `PomGpuBackend` impls (`CudaBackend`, `MetalBackend`,
  `VulkanBackend` for Android, `VulkanDesktopBackend` for desktop AMD/Intel) that wrap each
  compiled-in backend's existing free functions, plus `register_compiled_backends()` which hands the
  OS-appropriate set to the dispatcher. Phase 1 keeps one-active-backend-per-build (plan §4); Phase 3
  flips desktop to register CUDA + Vulkan together.
* **`src/main.rs`** — calls `register_compiled_backends()` after logging init, then logs the unified
  device probe (`unified_device_list()`), so a startup with multiple GPUs across backends prints one
  line per detected device.
* **`src/lib.rs`** — declares the new `device`, `pom_gpu_backends`, and `inference_engine` modules
  alongside the existing per-target `pom_gpu` alias.

Phase 1 is **behaviour-preserving**: `main.rs`/`miner.rs`/`slm.rs`/`ios.rs`/`android.rs` still call
the same free functions they always did. The trait + dispatcher are layered on top, ready for the
additive-backends flip.

### Phase 2 — Desktop Vulkan port + CUDA re-sync (plan §2.4, §4)

* **`cuda/pom_mine.cu`** — replaced with `keryx-pascal-miner`'s hand-tuned Pascal (sm_61) kernel
  (the proven 9 MH/s-on-P40 path vs ~4.5 generic). 2×128-bit `longlong2` `__ldg` gather, u32 prefix
  table, predicated `find_tensor`, Lemire magic-modulo, 128 threads/block + shared-memory prefix.
* **`src/pom_gpu.rs`** (CUDA) — replaced with `keryx-pascal-miner`'s version: multi-arch PTX
  selector (`get_pom_ptx`/`get_gpu_ptx_by_id`), `mod_magic`/`verify_mod_magic` host self-test, u32
  prefix gather, 128-thread + smem launch config. Drop-in for the metal fork's old single-arch PTX.
* **`build.rs`** — rewrote to compile PTX for **every major NVIDIA arch** (sm_61/70/75/80/86/89/90)
  and emit an `$OUT_DIR/pom_ptx.rs` selector, ported from pascal-miner (incl. the Windows
  `discover_clbin`/`MSVC_CLBIN` host-compiler lookup). Gated on the `cuda` Cargo feature. Added the
  `vulkan` feature hook (the keryx-vulkan crate compiles its own shaders).
* **`keryx-vulkan/`** — replaced the mobile-only crate with `keryx-miner-rdna3`'s **desktop**
  version (multi-device `enumerate_devices`, `inference_device_index`, `from_raw_handles` borrowed-
  device path for zero-dup, `create_device_local_address_buffer_streamed`, `PomWalkShared`). Then
  **ported the mobile i32-fallback back in**:
  * `keryx-vulkan/shaders/pom_walk_i32.comp` — the `shaderInt64`-less `uvec2` + Barrett-reduction
    fallback shader (proven necessary on Adreno 740, and plan §2.4 flags the same gap as a real
    risk on Intel Arc).
  * `keryx-vulkan/src/lib.rs` — added a `shader_int64: bool` field to `Vk`, queried via
    `get_physical_device_features2` **before** device creation (was: requested unconditionally),
    plus a `supports_shader_int64()` accessor. The desktop backend now degrades to the i32 shader
    on shaderInt64-less drivers instead of failing device creation.
  * `keryx-vulkan/src/pom_walk.rs` — dual-path dispatch: picks `pom_walk.comp` (native uint64 +
    Granlund-Montgomery modulo) or `pom_walk_i32.comp` (uvec2 + Barrett) at `PomWalkGpu` build
    time based on `supports_shader_int64()`, packs the matching push-constant block (`PomPush` vs
    `PomPush32`) per dispatch. `push_bytes` made generic over the push-block type.

  This unification gives the desktop backend **both** the mobile fallback AND the desktop zero-dup
  paths in one crate — which neither fork had alone.
* **`src/pom_gpu_vulkan_desktop.rs`** (new) — `keryx-miner-rdna3`'s desktop Vulkan `pom_gpu.rs`
  (streamed-blob `PomWalkGpu::new_streamed` + zero-dup `PomWalkShared` over the llama.cpp engine's
  resident weights), gated to the `vulkan` Cargo feature.
* **`src/llm_engine.rs`** (new) — `keryx-miner-rdna3`'s in-process llama.cpp (Vulkan) inference
  engine, gated to `vulkan`.
* **`src/models.rs`** — added `PomAnchor` + `pinned_pom_anchor` (consensus-pinned `(R_T, N)` guard
  the desktop Vulkan path uses to reject a wrong-quant GGUF at index-build time). The anchor TABLE
  is intentionally empty with a TODO: it must be copied verbatim from the node's `POM_TIERS` before
  the desktop Vulkan backend mines on mainnet.
* **`src/slm.rs`** — added a `vulkan`-gated `active_engine` stub (returns `None` on the candle path;
  the desktop Vulkan walk's `install_shared` treats `None` as "fall back to streamed blob").

### Phase 3 — Additive backends via Cargo features (plan §2.4, §4)

* **`Cargo.toml`** — added three additive backend features, `cuda` / `vulkan` / `metal`. An
  unmodified `cargo build` still produces the historical per-fork binary (per-OS backend selection
  continues via the existing `target.cfg` dependency gating + `lib.rs`'s `#[cfg]` module alias).
  To build a heterogeneous-rig desktop binary (one process, NVIDIA + AMD):

  ```
  cargo build --features cuda,vulkan
  ```

  The `vulkan` feature pulls the optional desktop `keryx-vulkan`, `llama-cpp-2`, and
  `llama-cpp-sys-2` deps (heavy C++/cmake build, hence optional). A CUDA-only `cargo build` compiles
  none of them.
* **Startup multi-backend probe + unified enumeration** — `device::unified_device_list()` walks
  every registered backend's `enumerate()` and merges them into one backend-tagged list. A machine
  with one NVIDIA + one AMD card yields two entries (one per backend). A both-features build on a
  single-vendor machine runs fine — the other backend's probe returns zero devices, no crash.
* **Vendor → backend routing** (plan §2.3) — `parse_vulkan_vendor()` in `pom_gpu_backends.rs` tags
  each Vulkan device AMD/Intel/Unknown from its name string. NVIDIA cards route to the CUDA backend
  (which probes them separately); a NVIDIA-named Vulkan device only appears if CUDA is absent, and
  is then still mined via Vulkan (the §2.3 "NVIDIA → CUDA, else Vulkan" fallback).

### Phase 4 — Inference engine unification (plan §2.5)

* **`src/inference_engine.rs`** (new) — the `InferenceEngine` trait (`load`/`generate`/
  `supports_zero_dup`) and `EngineChoice`/`pick_engine()` decision point. Documents the three
  engine pairings and their **distinct** zero-dup mechanisms (candle-CUDA `load_shared`,
  candle-Metal `QTensor::metal_storage`, llama.cpp-Vulkan `PomWalkShared`) — these are NOT a single
  shared code path; that's real per-pairing porting work called out explicitly.
* Phase 4 here defines the trait + the llama.cpp engine option. The candle path in `slm.rs` keeps
  its existing `ENGINE` (a candle model) for now — wrapping it in the trait is a later,
  behaviour-preserving pass. `pick_engine()` returns `LlamaCpp` when the `vulkan` feature compiled
  the desktop Vulkan backend AND a Vulkan device is the designated inference GPU; otherwise
  `Candle` (the existing path).

---

## Build verification (post-review — see "Review fixes" below)

**Linux**: `cargo build --release` (default features, i.e. CUDA — same as Windows, unconditional on
desktop) produces a genuine ELF64 x86-64 PIE binary. Built in a `nvidia/cuda:12.2.2-devel-ubuntu22.04`
container (no GPU passthrough, so no real driver/`nvidia-smi` — see below), copied to
`keryx-miner-linux-x64` in this directory. `Finished 'release' profile [optimized] target(s) in 4m
57s`, no errors, no new warnings. Required `CUDA_COMPUTE_CAP=86` because the vendored
`candle-kernels`' own build script (`bindgen_cuda`) shells out to `nvidia-smi` for arch
autodetection when the env var isn't set, and a container without GPU passthrough has no
`nvidia-smi` — the same class of workaround as `keryx-pascal-miner`'s `build_sm61.bat` setting
`CUDA_COMPUTE_CAP` explicitly, just for a different reason (no driver at all, vs. wanting a
non-default arch). Running the binary in the same container fails with `libcuda.so.1: cannot open
shared object file` — expected, since the container has the CUDA toolkit but not an actual GPU
driver; it dynamically links against `libcuda.so.1` correctly and will resolve on a real machine
with an NVIDIA driver installed. `--features vulkan`/`cuda,vulkan` not yet attempted on Linux (would
need `libvulkan-dev`, `glslc`, cmake/ninja, libclang in the container — same requirements as Windows).

All of the following build clean on a real Windows/NVIDIA/nvcc-12.8/Vulkan-SDK machine:

* `cargo check` (default features) ✅
* `cargo check --features cuda` ✅
* `cargo check --features vulkan` (needs `LIBCLANG_PATH` set and `CMAKE_GENERATOR=Ninja` — see
  below) ✅
* `cargo check --features cuda,vulkan --bins` (the heterogeneous-rig binary) ✅
* `cargo test --lib` → 11 passed, 3 ignored (need a Gemma-3-4B GGUF on disk), 0 failed ✅
* `cargo check -p keryx-vulkan` ✅ (also `cargo test -p keryx-vulkan`; the one failing test,
  `bda_sharded_blob_allocates`, is a pre-existing AMD-specific hardware diagnostic — ported
  byte-identical from `keryx-miner-rdna3` — that asserts AMD's 2 GiB `maxMemoryAllocationSize`
  single-allocation cap, which this dev machine's NVIDIA GTX 1070 doesn't enforce; not a porting
  bug)

### Required environment for `--features vulkan`

* `LIBCLANG_PATH` pointing at a real LLVM install's `bin/` (bindgen needs `libclang.dll`) —
  documented in `keryx-miner-rdna3`'s README, same requirement here.
* `CMAKE_GENERATOR=Ninja` — the default Visual Studio generator fails building llama.cpp's
  `vulkan-shaders-gen` ExternalProject subbuild on Windows (`CMake Error ... file INSTALL cannot
  find ...vulkan-shaders-gen.exe`), the same issue the rdna3 fork worked around. If you've already
  attempted a build without Ninja set, delete `target/*/build/llama-cpp-sys-2-*` before retrying —
  a stale CMakeCache configured with the VS generator will not switch generators on a re-run.
* **`--release` + `--features vulkan` hits Windows `MAX_PATH` (260 chars)** inside llama.cpp's
  vendored `vulkan-shaders-gen` ExternalProject subbuild — its own `CMakeTestCXXCompiler.cmake`
  try-compile path nests `target\release\build\llama-cpp-sys-2-<hash>\out\build\ggml\src\
  ggml-vulkan\vulkan-shaders-gen-prefix\src\vulkan-shaders-gen-build\CMakeFiles\CMakeScratch\
  TryCompile-<id>\testCXXCompiler.cxx`, and `release` being 2 chars longer than `debug` was enough
  to tip a repo checked out at `E:\keryx\keryx-miner-unified` over the limit (`cl.exe`: `fatal
  error C1083: Cannot open compiler generated file: ''`). Confirmed by re-running with a short
  `CARGO_TARGET_DIR` (e.g. `set CARGO_TARGET_DIR=C:\kmu-rel`) — same feature set, same profile,
  builds clean. This is a path-length ceiling in the vendored C++ dependency's own build tree, not
  a bug in this repo's code; work around it with a short `CARGO_TARGET_DIR` or by enabling Windows
  long-path support (`git config core.longpaths true` doesn't cover this — it's `cl.exe`'s own
  `MAX_PATH` handling, so use `fsutil.exe behavior set longpaths 1` — admin — or a shorter checkout
  path). **`build-windows.bat` (below) handles this automatically**: whenever `vulkan` is part of
  the feature set being built, it sets `CARGO_TARGET_DIR=%SystemDrive%\kmu-build` unless the caller
  already set one, so the binary ends up outside the deep repo path.

## One-click build scripts

`build-windows.bat` and `build-linux.sh` (repo root) auto-detect the Vulkan toolchain (libclang,
cmake, glslc, and — Windows only — Ninja) and build `--features cuda,vulkan` when it's present,
falling back to `--features cuda` with an explanatory message otherwise. Pass a feature list as
the first argument to force it (e.g. `build-windows.bat cuda,vulkan`).

* **`build-windows.bat`** — end-to-end verified with a **fully clean** `--features cuda,vulkan`
  build (entire target dir wiped first, to rule out any stale-cache ambiguity):
  `Finished 'release' profile [optimized] target(s) in 12m 45s`, producing a genuine
  `C:\kmu-build\release\keryx-miner.exe` (86 MB PE binary, `--help` runs and prints the real CLI).
  This run is what surfaced and confirmed the fix for the CRT-mismatch bug below (fix item 4) —
  earlier attempts at incremental re-tests kept reusing a stale pre-fix `libmoe.a` because
  `bindgen_cuda`'s own staleness check (source-file mtime vs. existing `.a` mtime) doesn't know
  the compiler *flags* changed, so touching only `build.rs` doesn't force a real rebuild; a shell
  glob delete of the parent directory also silently no-op'd once. Only a full clean build (or
  deleting the exact `.../candle-kernels-<hash>/out/libmoe.a` file) gives a trustworthy signal.
* **`build-linux.sh`** — end-to-end verified with `cargo build --release --features cuda,vulkan`
  in a fresh Docker container (`nvidia/cuda:12.2.2-devel-ubuntu22.04` + LunarG's Vulkan SDK apt
  repo for `glslc`): `Finished 'release' profile [optimized] target(s) in 8m 26s`, producing a
  genuine ELF64 x86-64 PIE `keryx-miner` (27 MB). No `CMAKE_GENERATOR` override needed — Linux's
  default Makefiles generator builds llama.cpp's `vulkan-shaders-gen` fine (the Windows MAX_PATH/
  generator issues above are Windows-only). No CRT-mismatch issue either — ELF has no MT/MD
  distinction, only Windows PE does.

## Review fixes applied (post-implementation)

The initial pass above compiled `keryx-vulkan` in isolation and confirmed the new files parsed,
but never got a full `cargo check` of the main crate to succeed. Reviewing that gap, then pushing
through to a full `cargo build` of the shipping binary on both platforms, found four real bugs,
all now fixed:

1. **`build.rs` gated the CUDA PTX build behind the `cuda` Cargo feature, but nothing in `src/`
   checks `cfg(feature = "cuda")`.** `src/pom_gpu.rs` is compiled unconditionally on every non-
   Apple/non-Android target (matching every source fork, where CUDA was never opt-in on desktop),
   and `pom_gpu_backends.rs::register_compiled_backends()` unconditionally registers `CudaBackend`
   on desktop too. So a plain `cargo build` compiled `pom_gpu.rs`'s
   `include!(concat!(env!("OUT_DIR"), "/pom_ptx.rs"))` but the matching build step never ran,
   erroring `couldn't read .../pom_ptx.rs`. Fixed by building the PTX under the same target_os
   condition as the module itself, not the feature flag (`build.rs`).
2. **The pre-existing candle-kernels ↔ nvcc 12.8 incompatibility** (`reduce.cu`'s `SUM_OP(__half)`
   using `atomicAdd(__half*)`, needs sm_70+; `moe_wmma.cu`/`moe_wmma_gguf.cu`'s `nvcuda::wmma`,
   also sm_70+) — already fixed once in `keryx-pascal-miner`'s copy of the same vendored patch but
   never carried over here. Ported the same `#if __CUDA_ARCH__ >= 700` / host-pass-everywhere
   guards from `keryx-pascal-miner/vendor/candle-kernels` verbatim.
3. **`WeightIndex` (in `src/pom.rs`, kept from the `keryx-metal` base) was missing
   `read_chunk_range`**, which `pom_gpu_vulkan_desktop.rs` (ported from `keryx-miner-rdna3`) calls
   for the zero-dup GGUF streaming path. `keryx-metal`'s `pom.rs` predates that method; `rdna3`'s
   fork added it. Ported the method verbatim from `keryx-miner-rdna3/src/pom.rs` — the struct
   layout (`WeightIndex`, `ChunkSource`, `read_exact_at`) is identical between the two forks, so
   it's a straight drop-in.
4. **MSVC CRT (runtime library) mismatch between `candle-kernels` and `llama-cpp-sys-2` when both
   `cuda` and `vulkan` are built together on Windows.** `vendor/candle-kernels/build.rs` compiles
   its MoE kernels' host code via `bindgen_cuda`, which shells out to nvcc with no explicit CRT
   flag — nvcc's MSVC default is the static runtime (`/MT`). `llama-cpp-sys-2`'s CMake build
   defaults to the dynamic runtime (`/MD`, also Rust's own MSVC-target default). Linking both
   `libmoe.a` and `llama-cpp-sys-2`'s static libs into one binary (specifically the `[lib]`
   target's `cdylib` artifact, built for Android JNI) trips MSVC's linker: `LNK2038: mismatch
   detected for 'RuntimeLibrary'`. This only surfaces during an actual `cargo build` of the final
   link — `cargo check` never reaches it, which is why the earlier `cargo check --release
   --features cuda,vulkan` pass in this document didn't catch it. Fixed by passing `-Xcompiler
   /MD` to the `moe_builder` in `vendor/candle-kernels/build.rs` (Windows-only branch, alongside
   the existing `-D_USE_MATH_DEFINES`) so nvcc's host-compiled objects match everything else.

Also removed an unused `use std::sync::Arc;` in `inference_engine.rs`.

---

## What remains (plan §4 Phases 5–7 — hardware, not code)

These are explicitly **out of scope** for code work — they need real GPUs and are listed in
`plan.md` §5 as mandatory before any platform is called "done":

* **Phase 5 — Heterogeneous-rig validation.** Real test: one machine, NVIDIA + AMD cards both
  installed, confirm both mine simultaneously, confirm tier assignment + OOM-downgrade work across
  the mixed device list. The code path (`unified_device_list` + per-backend workers) is wired; the
  validation needs the rig.
* **Phase 6 — Intel Arc validation.** The one vendor with zero prior validation in any fork. Vulkan
  *should* work per spec; the `supports_shader_int64`/i32-fallback path is precisely the Adreno-
  style surprise guard the plan §2.4 calls out. Budget real debugging time, don't assume.
* **Phase 7 — Cutover.** Once parity is confirmed against all three source forks (same hashrate
  ballpark, same correctness, bit-exactness re-proven for every ported kernel via the plan's §5
  method), point the other three forks' READMEs at this repo.

## Known follow-ups (code, deferred)

* **Populate `POM_ANCHORS`** in `models.rs` from the node's `POM_TIERS`/`POM_TIERS_H2` before the
  desktop Vulkan backend mines on mainnet (currently empty — the guard is skipped with a warning).
* **Wrap the candle path in `InferenceEngine`** — `slm.rs`'s `ENGINE` (a candle model) is not yet
  an `InferenceEngine` impl; doing so is a behaviour-preserving pass that doesn't change inference,
  only how the PoM walk obtains its zero-dup handle.
* **Generalize the per-backend `MINING_TIERS`/`OOM_BANLIST` maps to `GpuHandle` keys** (plan §2.6).
  Today each backend keeps its own device-index-keyed map; the trait exposes tier query/mutation so
  callers stay backend-agnostic, but the maps aren't yet unified into one `GpuHandle`-keyed store.
