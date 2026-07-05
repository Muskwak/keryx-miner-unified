# Keryx Miner — Unified System Reference

Everything known about `keryx-miner-unified`: architecture, the vendored candle
patches, the PoM/OPoI mechanics, per-backend zero-dup, build/distribution, and the
gotchas that have bitten us. Written as the single source of truth so nobody has to
re-derive this from the code again.

Repo: `Muskwak/keryx-miner-unified` (fork of `Keryx-Labs/keryx-miner`).

---

## 1. What this is

A single backend-agnostic miner unifying **three** previously-separate forks:

| Source fork            | GPU backend | Inference engine        | Now covers            |
|------------------------|-------------|-------------------------|-----------------------|
| `keryx-pascal-miner`   | CUDA        | candle (CUDA)           | NVIDIA (Pascal→Blackwell) |
| `keryx-metal`          | Metal       | candle (Metal)          | Apple Silicon (macOS/iOS) |
| `keryx-miner-rdna3`    | Vulkan      | llama.cpp FFI (Vulkan)  | AMD / Intel / Android |

One binary per OS. Desktop (Windows/Linux) compiles **CUDA + desktop Vulkan**
together; macOS/iOS compile **Metal**; Android compiles **Vulkan**.

Two orthogonal axes:
- **Mining backend** — runs the memory-hard PoM walk over resident weights.
- **Inference engine** — serves OPoI challenges (candle on NVIDIA/Apple, llama.cpp on AMD/Intel/Android).

---

## 2. Workspace layout

```
Cargo.toml            # [workspace] members = ["inference", "plugins/*", "keryx-vulkan"]
                      # default-members = ["."]  ← plain `cargo build` builds ONLY the main binary
src/                  # the main keryx-miner binary + lib
  main.rs             # desktop entry: CLI, tier assignment, backend registration
  lib.rs              # PluginManager, Plugin/Worker traits, module decls
  miner.rs            # MinerManager: spawns CPU + GPU + Metal worker threads
  pom.rs              # PoM proof builder + verifier (byte-exact mirror of the node)
  pow.rs              # State, serialize_header, generate_block_if_pom
  pom_gpu.rs          # CUDA PoM walk + ensure_installed + zero-dup dispatch
  pom_gpu_metal.rs    # Metal PoM walk (bindless over candle MTLBuffers)
  pom_gpu_vulkan.rs / pom_gpu_vulkan_desktop.rs  # Vulkan PoM walk
  pom_gpu_backends.rs # PomGpuBackend trait impls (Cuda/Metal/Vulkan/VulkanDesktop) + register
  device.rs           # unified device model + backend registry
  slm.rs              # candle inference engine (ENGINE), load_engine, pom_shared, pom_force_split
  llm_engine.rs       # llama.cpp Vulkan inference engine
  inference_engine.rs # InferenceEngine trait + EngineChoice (candle vs llama.cpp)
  models.rs           # ModelSpec table, tiers, activation DAAs, pom_tier_index
  quantized_llama_split.rs / quantized_qwen3_split.rs  # custom split loaders (zero-dup)
  ios.rs / android.rs # mobile entry points (JNI / objc2 bridges)
plugins/cuda/         # keryxcuda — the CUDA kHeavyHash worker plugin (cdylib) ← REQUIRED, see §7
plugins/opencl/       # dead (OpenCL, no PoM proof) — no longer loaded
keryx-vulkan/         # Vulkan PoM+shaders crate (optional dep, activated by `vulkan` feature)
inference/            # keryx-inference crate
vendor/               # vendored candle patches (see §4)
build.rs              # top-level: PoM PTX multi-arch build, proto codegen
build-windows.bat / build-linux.sh  # one-click builds (main binary only — NOT the plugin!)
```

---

## 3. Consensus timeline (mainnet DAA gates)

These MUST match the node's `MAINNET_PARAMS`. Wrong values → rejected blocks / fork-off.

| Fork | DAA | Const (miner) | What it does |
|------|-----|---------------|--------------|
| **H** (PoM + OPoI-v2 + ratio-reward) | 37,780,000 | `POM_ACTIVATION_DAA`, `OPOI_V2_ACTIVATION_DAA` | kHeavyHash → Proof-of-Model; uncensored lineup |
| **H2** (5-tier lineup) | 38,951,445 | `VERY_LIGHT_ACTIVATION_DAA` | inserts very-light tier; top tier 70B-Q4→Q2 |
| **H3** (PoM block-level) | 43,450,000 | `POM_LEVEL_ACTIVATION_DAA` | salt pph words + commit `pom_final_state` to block hash |

`POM_ACTIVATION_DAA == u64::MAX` means PoM disabled (serve-only, legacy kHeavyHash).

### H3 details (the 2026-07-05 hardfork)
Two required miner changes, both mirroring node commit `45aa0866` / miner-side `42b93487`:
1. **`POM_H3_PPH_SALT`** = `[0x7C99D381176D4EC4, 0xC2E28E3E28118C36, 0xD496CE1B129B76CA, 0x47CF0979FA580BCE]`
   = `sha256("keryx-h3-pom-pph-salt")` as 4 LE u64 words. XORed into the pre_pow_hash
   words feeding **both** PoM folds (seed fold + pow-value fold) at/after H3.
   - Applied **host-side** as a raw-byte XOR on the pph before it reaches any GPU kernel
     (`pom::salt_pph_bytes_for_daa`) — CUDA/Vulkan/Metal kernels only fold pph into
     seed+pow (never Fiat-Shamir challenges), so salting the bytes once is byte-identical
     to a per-kernel salted fold, **zero kernel/shader changes**.
   - CPU proof builder keeps the raw hash for challenges but uses `pom_pow_value_h3` /
     `pom_block_seed_h3` for the folds (node-mirrored split).
2. **`RpcBlockHeader.pomFinalState`** (proto field 16) — filled from the winning walk's
   `final_state` on submit (like nonce), AND folded into the block hash by
   `serialize_header` when `!for_pre_pow && daa >= POM_LEVEL_ACTIVATION_DAA`.

Fold constants (byte-exact, never change): seed salt `0x4B65727978531`; pow-value
XORs `0x9E3779B97F4A7C15`, `0xC2B2AE3D27D4EB4F`, `0x165667B19E3779F9`, `0xD6E8FEB86659FD93`.

---

## 4. Vendored candle patches (`vendor/`, via `[patch.crates-io]`)

Three crates are vendored. **Each exists for a specific reason — do not drop them.**
The unification originally dropped `candle-transformers` and it silently broke Gemma3
zero-dup for a day (see §6).

| Crate | Why vendored |
|-------|--------------|
| `candle-kernels` | Upstream MoE bf16 WMMA kernels only compile sm_80+, excluding Turing/Pascal. Guarded with `#if __CUDA_ARCH__ >= 800`. **Also** rewritten to build a multi-arch **fatbin per kernel file** (Pascal 61 → Blackwell 120) instead of `bindgen_cuda`'s single autodetected arch — so OPoI inference runs native on every GPU gen, not just the build machine's. |
| `candle-core` | Upstream keeps `QTensor::storage` private + `device_ptr` unimplemented on Metal. Adds `QTensor::metal_storage() -> Option<&QMetalStorage>` so the Metal PoM walk reads candle's `MTLBuffer`s in place. Also: `Ptx::from_binary` path for loading the fatbins (device.rs `get_or_load_func`). |
| `candle-transformers` | Upstream `quantized_gemma3::ModelWeights` keeps QMatMul weight fields private → PoM zero-dup can't reuse resident Gemma weights (loads a dup ~2GB copy). Adds `pom_quant_tensors()` (mirrors `quantized_llama_split`). Also caps `MAX_SEQ_LEN` 131072→8192 so per-layer rotary sin/cos tables drop ~4.3GB→~270MB (mining inference is ≤2048 tokens). |

**GOTCHA:** the repo `.gitignore` has a broad `models/` rule that swallows
`vendor/candle-transformers/src/models/` — force-add (`git add -f`) or only ~15 of 209
files get committed and the build breaks for everyone else.

Module `ptx.rs` fatbin build + `Module::ptx()` returns `&'static [u8]` (fatbin), loaded
via `cudarc::nvrtc::Ptx::from_binary` (NOT `.into()` which is text-only).

---

## 5. Model tiers (post-H2 lineup — `specs_for(VERY_LIGHT_ACTIVATION_DAA, ..)`)

One model per tier. CLI flag picks the **ceiling**; each GPU mines the highest tier ≤
ceiling that its VRAM holds (`assign_pom_tiers` + `POM_TIER_LADDER`).

| Flag | Tier | Model | Loader / format | Zero-dup (CUDA)? |
|------|------|-------|-----------------|-------------------|
| `--very-light` | VeryLight | Qwen3-1.7B | `GgufQwen3` → Qwen3Split | ✅ split loader |
| `--light` | Light | Gemma-3-4B | `GgufGemma3` → QuantizedGemma3 | ✅ **only via vendored `pom_quant_tensors`** |
| (default) | Default | Dolphin-3.0-Llama-3.1-8B | `Gguf` → SplitWeights | ✅ split loader |
| `--high` | High | Qwen3-32B (Q4_K_M) | `GgufQwen3` → Qwen3Split | ✅ split loader |
| `--very-high` | VeryHigh | Llama-3.3-70B (Q2_K_L 32GB / Q4 48GB) | `Gguf` → SplitWeights | ✅ split loader |

`POM_TIER_LADDER` VRAM floors: VeryHigh 30000, High 24000, Default 8000, Light 5000, VeryLight 2000 (MB).

**Platform tier support:**
- **iOS** — hardcoded `Tier::VeryLight` only (`ios.rs`, no flag parsing) = Qwen3-1.7B post-H2.
- **macOS** — full flag-based tiers (builds the main binary → `main.rs`), any tier its unified memory holds.
- **Windows/Linux** — full, per-GPU VRAM-gated on a heterogeneous rig.

---

## 6. Zero-dup (sharing inference weights with the PoM walk) — **per-backend, DIFFERENT**

The walk needs the model weights resident in VRAM. Zero-dup avoids a second copy on the
GPU that also serves inference. The mechanism is **not** shared across backends:

### CUDA (`pom_gpu.rs` `ensure_installed_inner`)
- `slm::pom_shared(model_id)` returns the inference engine's resident `Arc<QTensor>` map
  IF: engine loaded, same model_id, `device.is_cuda()`, and inner is a **split-loader**
  variant (`QuantizedSplit` / `QuantizedQwen3Split` / **`QuantizedGemma3`**).
- Walk zero-dups iff `cuda_gpu_id(inference_device) == mining_device_id`. Else standalone `PomGpuMiner::load(gguf)`.
- `set_pom_force_split(true)` forces the split loader so mining models expose their tensors.
- **Non-inference GPUs cannot zero-dup** — no resident inference model to share (by design; they load a standalone copy).
- Requires **all five** model variants to have `pom_quant_tensors()`. Gemma3's lives in the vendored candle-transformers.

### Metal (`pom_gpu_metal.rs`) — Apple only
- `pom_shared` is CUDA-gated → always `None` on Metal. The walk **always** calls
  `PomGpuMiner::load(gguf)`, which reads candle's per-tensor `MTLBuffer`s in place via
  bindless GPU pointers (`QTensor::metal_storage`).
- "Zero-dup" here = **no packed 2nd blob**, NOT sharing with inference. There are
  effectively two resident copies (inference + walk) — fine on unified memory.
- Identical to the `keryx-metal` fork; no inference-sharing on Metal by design.

### Vulkan (desktop `pom_gpu_vulkan_desktop.rs`, Android `pom_gpu_vulkan.rs`)
- `PomWalkShared` over the llama.cpp engine's resident ggml-vulkan buffers.

---

## 7. Plugins & runtime dependencies (the ".dll question")

**GPU mining threads only spawn if a plugin loads** (`MinerManager::new` → `launch_gpu_threads`
gated on `manager.has_specs()`). The plugin is `keryxcuda` (the CUDA kHeavyHash worker); it
supplies the `WorkerSpec`s that become the GPU threads — and the PoM walk runs *inside* those
threads. **Without the plugin, desktop CUDA GPUs do not mine (CPU-only).**

Plugin WHITELIST (main.rs): `["libkeryxcuda", "keryxcuda"]` (the dead OpenCL/AMD plugin was dropped).

| Platform | Needs plugin? | Runtime deps (NOT bundled — target must have them) |
|----------|---------------|-----------------------------------------------------|
| **Windows** | ✅ `keryxcuda.dll` next to the exe | CUDA runtime DLLs (`cublas64_12`, `curand64_10`, …) + NVIDIA driver |
| **Linux** | ✅ `libkeryxcuda.so` next to the binary | `libcublas.so.12`, `libcublasLt.so.12`, `libcurand.so.10` (CUDA Toolkit 12.x) + `libcuda.so.1` (driver) |
| **macOS** | ❌ Metal PoM is built-in (`launch_metal_pom_miner` spawns even with no plugin) | none — uses the OS Metal framework |
| **iOS** | ❌ Metal built-in, ships as an `.ipa` app bundle | none |
| **Android** | ❌ Vulkan built-in | Vulkan loader (system) |

**`build-windows.bat` / `build-linux.sh` build ONLY the main binary, NOT the plugin.**
The plugin must be built separately: `cargo build --release -p keryxcuda` → `keryxcuda.dll`
(Windows) / `libkeryxcuda.so` (Linux). Bundle it alongside the binary.

The plugin ABI is the `Plugin`/`Worker` traits in `lib.rs` (`_plugin_create` C entry via
`declare_plugin!`). Rebuild the plugin from the same source tree as the binary to guarantee
a match — the plugin `.dll`/`.so` size/contents change when candle/deps change even if the
ABI is stable.

Other runtime: the miner starts a bundled **IPFS (kubo)** daemon for model distribution;
model GGUFs are fetched by CID and cached (`prefetch_models` blocks mining until ready).

---

## 8. Build & distribution

**Local (this machine):**
- Windows: `build-windows.bat` (auto short `CARGO_TARGET_DIR=C:\kmu-build` when `vulkan` is on, to dodge MAX_PATH on llama.cpp's CMake subbuild). Needs `LIBCLANG_PATH` for llama.cpp bindgen; MSVC; CUDA 12.8 (for Blackwell).
- Linux: cross-compiled in a Docker container (`kmu-linux-build`, `nvidia/cuda:12.2-devel`) inside WSL. `CUDA_COMPUTE_CAP=86` when no GPU/driver is visible (bindgen_cuda autodetect fails). **Strip CRLF** from `build-linux.sh` (`sed -i 's/\r$//'`) after syncing from Windows or the shebang breaks.
- Apple: GitHub Actions `Build Apple (macOS + iOS)` workflow (can't build on Windows). Artifacts: `keryx-miner-macos-arm64`, `keryx-miner-ios-unsigned-ipa`, `keryx-miner-ios-lib`.

**nvcc version gate:** Blackwell (sm_100/sm_120) needs nvcc 12.8+. Both `build.rs` and
`candle-kernels/build.rs` detect the nvcc version and **drop** sm_100/120 below 12.8, so an
older toolkit (e.g. the 12.2 Linux container) still produces a working Pascal→Hopper binary
instead of hard-failing the whole build.

**Release assets (v0.3.8):**
- `keryx-miner-windows-x64.zip` — `keryx-miner.exe` + `keryxcuda.dll`
- `keryx-miner-linux-x64.tar.gz` — cuda-only `keryx-miner` + `libkeryxcuda.so`
- `keryx-miner-linux-x64-vulkan.tar.gz` — cuda+vulkan `keryx-miner` + `libkeryxcuda.so`
- `keryx-miner-macos` — standalone Mach-O arm64 (Metal built-in)
- `keryx-ios-unsigned.ipa` — unsigned app bundle

---

## 9. Known gaps / caveats

- **Desktop Vulkan mining is NOT wired to spawn threads.** `VulkanDesktopBackend` is
  registered (shows in the startup GPU probe) but `MinerManager` only spawns CUDA-plugin
  threads + a macOS Metal thread. An AMD/Intel GPU on Windows/Linux is *detected & logged*
  but no mining thread spawns for it (falls to CPU). Marked "Phase 3" in comments — future work.
- **iOS is very-light only** (hardcoded).
- **Linux binaries built with `CUDA_COMPUTE_CAP=86`** (Ampere) for the inference-kernel
  default when no GPU is in the build container; the multi-arch fatbin still covers
  Pascal→Blackwell for both PoM and inference regardless.
- The kHeavyHash plugin still loads post-PoM because it's what spawns the GPU threads; its
  kHeavyHash solutions are ignored once `daa >= POM_ACTIVATION_DAA` (the PoM branch takes over).

---

## 10. Session fix log (what changed and why)

- **H3 hardfork** (`5ec1a5d`, `effc8cd`) — pph salt + `pom_final_state` header field + block-hash commitment. Verified salt against sha256 derivation; 4 regression tests added.
- **candle-kernels multi-arch fatbin** — inference kernels now native Pascal→Blackwell, not single-arch.
- **Blackwell nvcc gate** (`7e24451`) — detect 12.8+ instead of assuming it; older toolkits degrade gracefully.
- **Gemma3 zero-dup restore** (`d0137de`) — re-vendored candle-transformers (dropped in unification) with `pom_quant_tensors()` + `MAX_SEQ_LEN` cap; wired `QuantizedGemma3` into `pom_shared`. This is the fix for `--light` double-loading Gemma (~2GB) + the 4.3GB rotary tables.
- **Cherry-picked upstream** — `6e78ab6`/`a74a684` slm.rs stop-token fixes (Llama-3.3-70B, Qwen3-1.7B); `f8d84ae` CLI one-model-per-tier + drop dead OpenCL plugin.
- **iOS `-force_load`** (`34631f4`) — restored on OTHER_LDFLAGS (lost when ios-app/ copied from keryx-metal) + CI size-check guard.
- **Intel Arc Vulkan** (`bcc0f3b`) — i32-fallback-shader workgroup-size dispatch fix.

> Note: cherry-picks show as "N commits behind upstream" because git compares commit
> identity, not content — the content IS present. Only a real merge resets that counter.
