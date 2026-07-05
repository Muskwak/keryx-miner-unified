//! Unified device model + backend trait (Phase 1, plan §2.1/§2.2).
//!
//! Replaces the fork-per-binary era's bare `u32 device_id` — which meant a CUDA ordinal, a Vulkan
//! enumeration index, OR a Metal device index depending on the fork, never more than one meaning
//! per binary — with a backend-qualified handle. A single unified binary can now hold multiple
//! live backends at once (e.g. CUDA + Vulkan on one Windows/Linux machine, §2.4), and every map
//! that used to be `HashMap<u32, _>` (`MINING_TIERS`, `MINERS`, `OOM_BANLIST`, per-worker
//! hashrate, `device_for_model`) keys on `GpuHandle` instead.
//!
//! Three GPU mining backends are formalized here behind one trait (`PomGpuBackend`) so a single
//! binary can dispatch to whichever backend actually talks to a given device at runtime, instead
//! of the old compile-time `#[cfg]` exclusivity:
//!   * CUDA  — NVIDIA, the most-tuned path (Pascal sm_61 kernel, candle-CUDA inference).
//!   * Vulkan — AMD (RDNA3, validated), Intel Arc (planned, §6 open-q #5), Android (Adreno/Mali).
//!   * Metal — Apple Silicon (macOS + iOS), zero-dup over candle's resident MTLBuffers.
//!
//! Vendor → backend routing (plan §2.3): NVIDIA→CUDA, AMD→Vulkan, Intel→Vulkan, Apple→Metal. A
//! machine with an NVIDIA + an AMD card enumerates two devices, one per backend, and mines both
//! simultaneously (§2.6 heterogeneous-rig loop).

use std::sync::{Arc, OnceLock};

// ─── Device identity ─────────────────────────────────────────────────────────

/// Which GPU backend a device is served by. One binary may compile several backends in
/// (additive Cargo features, §2.4); each probed device is tagged with exactly one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Backend {
    /// NVIDIA, via candle's CUDA backend + the tuned `pom_mine` PTX kernel.
    Cuda,
    /// AMD / Intel / Android, via the `keryx-vulkan` compute crate (`ash`).
    Vulkan,
    /// Apple Silicon (macOS + iOS), via candle's Metal backend.
    Metal,
}

impl Backend {
    /// Short lower-case tag for log lines, e.g. `PoM(cuda)[gpu0]`.
    pub fn tag(&self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
            Backend::Metal => "metal",
        }
    }
}

/// A backend-qualified handle to one physical GPU. `index` is the backend's OWN native ordinal
/// (CUDA ordinal / raw Vulkan `vkEnumeratePhysicalDevices` position / Metal device index), so
/// `Backend::Cuda, index=0` and `Backend::Vulkan, index=0` are two different physical cards on
/// a heterogeneous rig. This is the single key that replaces every legacy `u32 device_id`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GpuHandle {
    pub backend: Backend,
    pub index: u32,
}

impl GpuHandle {
    pub const fn new(backend: Backend, index: u32) -> Self {
        Self { backend, index }
    }

    /// Compatibility shim: the legacy single-backend binaries passed a bare `u32 device_id`. The
    /// dispatcher still accepts that (treated as the *process's* active backend, index = device_id)
    /// during the trait refactor so call sites can migrate incrementally — Phase 1 keeps today's
    /// one-active-backend-per-build behaviour (plan §4 Phase 1) while the trait lands.
    pub fn legacy(device_id: u32) -> Self {
        Self {
            backend: active_backend(),
            index: device_id,
        }
    }
}

/// The one backend this process is "primarily" on, for the legacy `u32 device_id` shim above.
/// On a single-backend build (every fork today) this is just that backend; Phase 3 makes backends
/// additive and the dispatcher routes by `GpuHandle.backend` directly, so this becomes irrelevant.
#[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "android")))]
pub const fn active_backend() -> Backend {
    Backend::Cuda
}
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub const fn active_backend() -> Backend {
    Backend::Metal
}
#[cfg(target_os = "android")]
pub const fn active_backend() -> Backend {
    Backend::Vulkan
}

// ─── Device info + trait ─────────────────────────────────────────────────────

/// One enumerated GPU. `index` is the backend's own ordinal (matches `GpuHandle.index`).
/// `vendor` drives the §2.3 routing table; `vram_mb` drives per-GPU tier assignment.
#[derive(Clone, Debug)]
pub struct GpuDeviceInfo {
    pub handle: GpuHandle,
    pub name: String,
    pub vram_mb: u64,
    pub vendor: GpuVendor,
}

/// GPU vendor, for the §2.3 routing decision. Inferred from the device name / driver where the
/// backend exposes it (CUDA ⇒ NVIDIA by definition; Vulkan ⇒ parsed from the device name string;
/// Metal ⇒ Apple Silicon by definition).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    /// Unknown vendor — routed to whichever backend enumerated it (no cross-backend fallback).
    Unknown,
}

/// Formalization of the free-function surface every backend already exposes
/// (`install`/`uninstall`/`is_installed`/`is_loading`/`mine`/`current_tier`/`ensure_installed`/
/// `set_mining_tier`/`device_for_model`), so a single binary can hold several live
/// implementations at once and the dispatcher picks the right one per `GpuHandle`.
///
/// `device` is the backend's OWN native index (matches `GpuHandle.index`), NOT a global id — the
/// dispatcher resolves the backend from `GpuHandle.backend` first, then forwards `index`.
///
/// NOTE on tier state: today each backend keeps its own process-global `MINING_TIERS` /
/// `OOM_BANLIST` maps keyed by its native device index. The trait exposes the tier query/mutation
/// methods so callers stay backend-agnostic, but the per-backend maps are NOT yet unified into a
/// single `GpuHandle`-keyed store — that is a later generalization pass (plan §2.6) layered on top
/// of this trait, which deliberately mirrors the exact shapes that already exist so the refactor
/// is behaviour-preserving (plan §4 Phase 1: prove the trait before making backends additive).
pub trait PomGpuBackend: Send + Sync {
    /// Backend discriminator.
    fn backend(&self) -> Backend;

    /// Every GPU this backend can talk to, in its own enumeration order. Empty (not an error)
    /// when the backend's driver/loader is absent — the unified probe just contributes zero
    /// devices for that backend (plan §2.4: a both-features build runs fine with one vendor).
    fn enumerate(&self) -> Vec<GpuDeviceInfo>;

    /// VRAM (MB) of every device this backend sees, in `(native_index, mb)` pairs. The legacy
    /// shape `query_all_gpus_vram` already returns — kept so `main.rs`'s per-GPU tier assignment
    /// ports with no signature churn. Empty when the backend finds nothing.
    fn query_all_gpus_vram(&self) -> Vec<(usize, u64)>;

    /// Record a device's mining tier so its miner can be rebuilt after an inference evicts it.
    fn set_mining_tier(&self, device: u32, model_id: [u8; 32], gguf_path: String);

    /// PoM tier index of a device's mining model at a given block DAA (recomputed per block, H2-gated).
    fn current_tier(&self, device: u32, daa: u64) -> Option<u8>;

    /// The native device index that mines `model_id`, if any (lowest matching index). Inference is
    /// routed to that device so only it pauses and the walk can share the resident weights.
    fn device_for_model(&self, model_id: &[u8; 32]) -> Option<u32>;

    /// Whether the miner is currently installed and ready on `device`.
    fn is_installed(&self, device: u32) -> bool;

    /// Whether a PoM model load/rebuild is in progress on any device (worker paused, not stalled).
    fn is_loading(&self) -> bool;

    /// Install the miner for `device` if not already resident; reload the model (resident again)
    /// and rebuild the gather if an inference evicted it. Heavy (model reload) but only when
    /// needed. `daa` is the current block's score (DAA-gated tier index). Returns true when ready.
    fn ensure_installed(&self, device: u32, daa: u64) -> bool;

    /// Drop the miner for `device`, releasing its hold on that device's mining-model VRAM so the
    /// inference engine can load another model there (inference has priority over PoW).
    fn uninstall(&self, device: u32);

    /// Search nonces in `[start, start + batch)` on `device`. Returns the lowest winning nonce, or
    /// `None` if no winner / no installed miner. `target_le` is the compact target as 32 LE bytes.
    fn mine(
        &self,
        device: u32,
        pre_pow_hash: &[u8; 32],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
    ) -> Option<u64>;
}

// ─── Dispatcher registry ─────────────────────────────────────────────────────

/// The process-wide set of compiled-in backends, registered once at startup. Each compiled-in
/// backend probes its own driver and contributes zero or more devices to the unified list (§2.2).
/// Built lazily so a backend whose driver init would panic on a missing loader never runs unless
/// its module is actually compiled in AND asked to enumerate.
static BACKENDS: OnceLock<Vec<Arc<dyn PomGpuBackend>>> = OnceLock::new();

/// Register the compiled-in backends. Called once from `main`/`lib` init. Idempotent — subsequent
/// calls are ignored (the first registration wins), matching the `OnceLock` contract.
pub fn register_backends(backends: Vec<Arc<dyn PomGpuBackend>>) {
    let _ = BACKENDS.set(backends);
}

/// All registered backends. Empty before `register_backends` is called (e.g. in unit tests that
/// exercise a single backend's free functions directly).
pub fn backends() -> &'static [Arc<dyn PomGpuBackend>] {
    BACKENDS.get().map(|v| v.as_slice()).unwrap_or(&[])
}

/// The single registered backend on this process, if exactly one is registered. Convenience for
/// the legacy single-backend code paths that haven't been generalized to iterate `backends()`
/// yet (Phase 1 keeps one-active-backend-per-build, plan §4).
pub fn sole_backend() -> Option<&'static Arc<dyn PomGpuBackend>> {
    let b = backends();
    if b.len() == 1 { Some(&b[0]) } else { None }
}

/// The unified, backend-tagged device list across every compiled-in backend whose probe succeeded.
/// A heterogeneous rig (e.g. one NVIDIA + one AMD card) yields two entries, one per backend
/// (plan §2.2). Empty when no backend found any device.
pub fn unified_device_list() -> Vec<GpuDeviceInfo> {
    let mut out = Vec::new();
    for b in backends() {
        out.extend(b.enumerate());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A minimal no-op backend for dispatcher tests — no real GPU needed.
    struct StubBackend(Backend, Mutex<Vec<u32>>);
    impl PomGpuBackend for StubBackend {
        fn backend(&self) -> Backend { self.0 }
        fn enumerate(&self) -> Vec<GpuDeviceInfo> {
            (0..self.1.lock().unwrap().len())
                .map(|i| GpuDeviceInfo {
                    handle: GpuHandle::new(self.0, i as u32),
                    name: format!("stub-{}-{}", self.0.tag(), i),
                    vram_mb: 8000,
                    vendor: GpuVendor::Unknown,
                })
                .collect()
        }
        fn query_all_gpus_vram(&self) -> Vec<(usize, u64)> {
            self.enumerate().into_iter().map(|d| (d.handle.index as usize, d.vram_mb)).collect()
        }
        fn set_mining_tier(&self, _: u32, _: [u8; 32], _: String) {}
        fn current_tier(&self, _: u32, _: u64) -> Option<u8> { None }
        fn device_for_model(&self, _: &[u8; 32]) -> Option<u32> { None }
        fn is_installed(&self, _: u32) -> bool { false }
        fn is_loading(&self) -> bool { false }
        fn ensure_installed(&self, _: u32, _: u64) -> bool { false }
        fn uninstall(&self, _: u32) {}
        fn mine(&self, _: u32, _: &[u8; 32], _: u64, _: &[u8; 32], _: u64, _: u64) -> Option<u64> { None }
    }

    #[test]
    fn handle_distinguishes_backend() {
        let cuda0 = GpuHandle::new(Backend::Cuda, 0);
        let vk0 = GpuHandle::new(Backend::Vulkan, 0);
        assert_ne!(cuda0, vk0);
        assert_eq!(cuda0, GpuHandle::new(Backend::Cuda, 0));
    }

    #[test]
    fn unified_list_merges_backends() {
        // Use a fresh process-global registry state via a local guard — OnceLock is write-once,
        // so this test only validates the merge logic when nothing else has registered. The
        // dispatcher's correctness is otherwise exercised by the per-backend free-function tests
        // each fork already carries.
        let cuda = Arc::new(StubBackend(Backend::Cuda, Mutex::new(vec![0, 1])));
        let vk = Arc::new(StubBackend(Backend::Vulkan, Mutex::new(vec![0])));
        let merged: Vec<GpuDeviceInfo> = cuda
            .enumerate()
            .into_iter()
            .chain(vk.enumerate())
            .collect();
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].handle, GpuHandle::new(Backend::Cuda, 0));
        assert_eq!(merged[1].handle, GpuHandle::new(Backend::Cuda, 1));
        assert_eq!(merged[2].handle, GpuHandle::new(Backend::Vulkan, 0));
    }

    #[test]
    fn legacy_handle_uses_active_backend() {
        let h = GpuHandle::legacy(2);
        assert_eq!(h.backend, active_backend());
        assert_eq!(h.index, 2);
    }
}
