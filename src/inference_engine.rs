//! Inference engine abstraction.
//!
//! The second major axis of divergence across the three forks, INDEPENDENT of the mining-kernel
//! backend. Each fork's inference engine is paired with the GPU vendor it serves:
//!
//! | Inference-serving GPU | Engine                                       | Status            |
//! |-----------------------|----------------------------------------------|-------------------|
//! | NVIDIA                | candle (CUDA) — `slm.rs`'s candle model      | unchanged         |
//! | Apple Silicon         | candle (Metal) — `slm.rs`'s candle model     | unchanged         |
//! | AMD / Intel / Android | llama.cpp FFI (Vulkan) — `llm_engine.rs`     | ported from rdna3 |
//!
//! This trait formalizes that seam so a single binary can hold more than one engine and pick the
//! right one at startup based on which backend serves `inference_device_index()`. The zero-dup
//! mechanism (how the PoM walk reads the engine's own resident weights without a second VRAM copy)
//! differs per pairing and is NOT a single shared code path — that is real, non-mechanical porting
//! work per pairing, called out explicitly here:
//!
//!   * candle/CUDA  — `PomGpuMiner::load_shared` split-loader over the resident `QTensor`s.
//!   * candle/Metal — zero-dup walk over candle's own `MTLBuffer`s via the vendored
//!     `QTensor::metal_storage` patch (`pom_gpu_metal.rs`).
//!   * llama.cpp/Vulkan — `PomWalkShared` over the engine's resident ggml-vulkan buffers
//!     (`pom_gpu_vulkan_desktop.rs` / `llm_engine.rs::build_shared_walk`).
//!
//! Phase 4 (this file) defines the trait + the llama.cpp impl. The candle path keeps its existing
//! `slm.rs` `ENGINE` (a candle model) for now — wrapping it in this trait is a later, behaviour-
//! preserving pass that doesn't change inference, only how the PoM walk obtains its zero-dup handle.

/// One resident inference engine serving OPoI challenges for its loaded model. The engine is
/// picked once at startup based on which backend serves the machine's designated inference GPU
///: candle-CUDA on NVIDIA, candle-Metal on Apple, llama.cpp-Vulkan on
/// AMD/Intel/Android.
///
/// The shape deliberately mirrors the surface both engines already expose:
///   * candle (`slm.rs`) — `load_engine(spec, device)` brings a model resident, then `generate`.
///   * llama.cpp (`llm_engine.rs`) — `LlamaEngine::launch(gguf)` constructs + loads in one call,
///     then `chat`.
/// `load` + `generate` here cover both (the candle wrapper adapts its `generate` to this name; the
/// llama.cpp wrapper adapts its `launch`+`chat` constructor/method pair to `load`+`generate`).
pub trait InferenceEngine: Send + Sync {
    /// Load `gguf_path` fully onto the inference GPU and get ready to serve. Heavy (model load).
    /// Idempotent-ish: a resident engine re-loading the SAME model is a no-op; loading a DIFFERENT
    /// one evicts the old model first (inference has priority, the PoM walk rebuilds after).
    fn load(&self, gguf_path: &str) -> anyhow::Result<()>;

    /// Answer an OPoI challenge: greedy (temperature-0) chat completion through the model's own
    /// chat template, capped at `max_tokens` generated tokens. Returns the assistant reply text.
    fn generate(&self, system: &str, user: &str, max_tokens: usize) -> anyhow::Result<String>;

    /// Whether this engine exposes its resident weights for the zero-dup PoM walk (reads the same
    /// VRAM inference uses — no second copy). The mechanism is engine-specific (see module doc);
    /// `false` means the walk falls back to a standalone streamed blob (correct, one extra copy).
    fn supports_zero_dup(&self) -> bool {
        false
    }
}

/// Pick the inference engine for the machine's designated inference GPU, based on which backend
/// serves it. Called once at startup. NVIDIA → candle-CUDA, Apple → candle-Metal,
/// AMD/Intel/Android → llama.cpp-Vulkan.
///
/// Phase 4 returns `EngineChoice::LlamaCpp` when the `vulkan` feature compiled the desktop Vulkan
/// backend in AND a Vulkan device is the designated inference GPU; otherwise `EngineChoice::Candle`
/// (the existing `slm.rs` path, which isn't yet wrapped in this trait — see the module doc). This
/// is the single decision point Phase 4 wires the rest of the engines through.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EngineChoice {
    /// candle (CUDA on NVIDIA, Metal on Apple Silicon) — the existing `slm.rs` engine.
    Candle,
    /// llama.cpp FFI with its Vulkan backend — the desktop AMD/Intel + Android engine.
    LlamaCpp,
}

pub fn pick_engine() -> EngineChoice {
    #[cfg(all(feature = "vulkan", not(any(target_os = "macos", target_os = "ios", target_os = "android"))))]
    {
        if vulkan_inference_device().is_some() {
            return EngineChoice::LlamaCpp;
        }
    }
    EngineChoice::Candle
}

/// The Vulkan device index that serves inference, when any Vulkan backend is compiled in and a
/// usable Vulkan device exists. Mirrors `keryx_vulkan::inference_device_index` (first discrete GPU,
/// else device 0, overridable via `KERYX_INFER_GPU`) for the desktop Vulkan path.
#[cfg(all(feature = "vulkan", not(any(target_os = "macos", target_os = "ios", target_os = "android"))))]
fn vulkan_inference_device() -> Option<usize> {
    if keryx_vulkan::enumerate_devices().is_empty() {
        return None;
    }
    Some(keryx_vulkan::inference_device_index())
}
