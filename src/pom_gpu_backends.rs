//! `PomGpuBackend` trait implementations for each compiled-in backend (plan §2.2, Phase 1).
//!
//! Phase 1 deliberately keeps every backend's existing free-function module
//! (`pom_gpu.rs` / `pom_gpu_metal.rs` / `pom_gpu_vulkan.rs`) as the source of truth and the
//! per-target `pom_gpu` alias in `lib.rs` unchanged — so `main.rs`, `miner.rs`, `slm.rs`,
//! `ios.rs`, and `android.rs` keep calling the same free functions they always have. This file
//! just wraps each compiled-in backend's free functions in a `PomGpuBackend` impl so a future
//! Phase 3 build (several backends compiled into one binary) can register them all with the
//! dispatcher in `device.rs` and route by `GpuHandle.backend`.
//!
//! The trait surface mirrors the existing free functions exactly (same names, same arg shapes),
//! so this is a behaviour-preserving shim — the heavy per-backend logic (CUDA PTX load, Metal
//! MTLBuffer walk, Vulkan streamed blob, OOM banlist, tier downgrade) is NOT duplicated here.

use std::sync::Arc;

use crate::device::{Backend, GpuDeviceInfo, GpuVendor, PomGpuBackend};

// ─── CUDA (NVIDIA) ───────────────────────────────────────────────────────────

/// CUDA backend wrapper. Only present on the desktop (Linux/Windows) build where `pom_gpu`
/// resolves to the CUDA module. The enum carries no state — every device's miner lives in the
/// module's process-global `MINERS` map, keyed by native CUDA ordinal.
#[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "android")))]
pub struct CudaBackend;

#[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "android")))]
impl PomGpuBackend for CudaBackend {
    fn backend(&self) -> Backend {
        Backend::Cuda
    }

    fn enumerate(&self) -> Vec<GpuDeviceInfo> {
        crate::pom_gpu::query_all_gpus_vram()
            .into_iter()
            .map(|(index, vram_mb)| GpuDeviceInfo {
                handle: crate::device::GpuHandle::new(Backend::Cuda, index as u32),
                // The CUDA driver doesn't hand us a friendly name through the VRAM probe path; the
                // free-function `query_all_gpus_vram` returns only ordinals + VRAM. The unified
                // list still needs a label, so synthesize one — a richer name query can be added
                // later without touching callers (they only log this).
                name: format!("CUDA device {}", index),
                vram_mb,
                vendor: GpuVendor::Nvidia,
            })
            .collect()
    }

    fn query_all_gpus_vram(&self) -> Vec<(usize, u64)> {
        crate::pom_gpu::query_all_gpus_vram()
    }

    fn set_mining_tier(&self, device: u32, model_id: [u8; 32], gguf_path: String) {
        crate::pom_gpu::set_mining_tier(device, model_id, gguf_path);
    }

    fn current_tier(&self, device: u32, daa: u64) -> Option<u8> {
        crate::pom_gpu::current_tier(device, daa)
    }

    fn device_for_model(&self, model_id: &[u8; 32]) -> Option<u32> {
        crate::pom_gpu::device_for_model(model_id)
    }

    fn is_installed(&self, device: u32) -> bool {
        crate::pom_gpu::is_installed(device)
    }

    fn is_loading(&self) -> bool {
        crate::pom_gpu::is_loading()
    }

    fn ensure_installed(&self, device: u32, daa: u64) -> bool {
        crate::pom_gpu::ensure_installed(device, daa)
    }

    fn uninstall(&self, device: u32) {
        crate::pom_gpu::uninstall(device)
    }

    fn mine(
        &self,
        device: u32,
        pre_pow_hash: &[u8; 32],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
    ) -> Option<u64> {
        crate::pom_gpu::mine(device, pre_pow_hash, timestamp, target_le, start, batch)
    }
}

// ─── Metal (Apple Silicon) ───────────────────────────────────────────────────

/// Metal backend wrapper. Present on macOS + iOS, where `pom_gpu` is aliased to
/// `pom_gpu_metal.rs`. Apple Silicon is single-GPU (device 0), but the map stays keyed by device
/// to match the CUDA path's signature.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalBackend;

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl PomGpuBackend for MetalBackend {
    fn backend(&self) -> Backend {
        Backend::Metal
    }

    fn enumerate(&self) -> Vec<GpuDeviceInfo> {
        crate::pom_gpu::query_all_gpus_vram()
            .into_iter()
            .map(|(index, vram_mb)| GpuDeviceInfo {
                handle: crate::device::GpuHandle::new(Backend::Metal, index as u32),
                name: format!("Metal device {}", index),
                vram_mb,
                vendor: GpuVendor::Apple,
            })
            .collect()
    }

    fn query_all_gpus_vram(&self) -> Vec<(usize, u64)> {
        crate::pom_gpu::query_all_gpus_vram()
    }

    fn set_mining_tier(&self, device: u32, model_id: [u8; 32], gguf_path: String) {
        crate::pom_gpu::set_mining_tier(device, model_id, gguf_path);
    }

    fn current_tier(&self, device: u32, daa: u64) -> Option<u8> {
        crate::pom_gpu::current_tier(device, daa)
    }

    fn device_for_model(&self, model_id: &[u8; 32]) -> Option<u32> {
        crate::pom_gpu::device_for_model(model_id)
    }

    fn is_installed(&self, device: u32) -> bool {
        crate::pom_gpu::is_installed(device)
    }

    fn is_loading(&self) -> bool {
        crate::pom_gpu::is_loading()
    }

    fn ensure_installed(&self, device: u32, daa: u64) -> bool {
        crate::pom_gpu::ensure_installed(device, daa)
    }

    fn uninstall(&self, device: u32) {
        crate::pom_gpu::uninstall(device)
    }

    fn mine(
        &self,
        device: u32,
        pre_pow_hash: &[u8; 32],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
    ) -> Option<u64> {
        crate::pom_gpu::mine(device, pre_pow_hash, timestamp, target_le, start, batch)
    }
}

// ─── Vulkan (Android / desktop AMD / Intel) ──────────────────────────────────

/// Vulkan backend wrapper. Present on Android today (where `pom_gpu` is aliased to
/// `pom_gpu_vulkan.rs`); Phase 2 adds the desktop Vulkan backend from the rdna3 fork so a single
/// Windows/Linux binary can serve AMD/Intel cards alongside NVIDIA's CUDA backend.
#[cfg(target_os = "android")]
pub struct VulkanBackend;

#[cfg(target_os = "android")]
impl PomGpuBackend for VulkanBackend {
    fn backend(&self) -> Backend {
        Backend::Vulkan
    }

    fn enumerate(&self) -> Vec<GpuDeviceInfo> {
        // Android is single-GPU; the free-function surface doesn't expose a vram probe, so report
        // one device 0 with an unknown VRAM (the shard-size picker in pom_gpu_vulkan.rs queries
        // the real limit at load time). Desktop Vulkan (Phase 2) replaces this with
        // `keryx_vulkan::enumerate_devices`.
        vec![GpuDeviceInfo {
            handle: crate::device::GpuHandle::new(Backend::Vulkan, 0),
            name: keryx_vulkan::probe_device().unwrap_or_else(|| "Vulkan device 0".into()),
            vram_mb: 0,
            vendor: GpuVendor::Unknown,
        }]
    }

    fn query_all_gpus_vram(&self) -> Vec<(usize, u64)> {
        // Android has no multi-device enumeration; report device 0 with the probed VRAM if any.
        match keryx_vulkan::probe_vram_mb() {
            Some(mb) => vec![(0, mb)],
            None => Vec::new(),
        }
    }

    fn set_mining_tier(&self, device: u32, model_id: [u8; 32], gguf_path: String) {
        crate::pom_gpu::set_mining_tier(device, model_id, gguf_path);
    }

    fn current_tier(&self, device: u32, daa: u64) -> Option<u8> {
        crate::pom_gpu::current_tier(device, daa)
    }

    fn device_for_model(&self, model_id: &[u8; 32]) -> Option<u32> {
        crate::pom_gpu::device_for_model(model_id)
    }

    fn is_installed(&self, device: u32) -> bool {
        crate::pom_gpu::is_installed(device)
    }

    fn is_loading(&self) -> bool {
        crate::pom_gpu::is_loading()
    }

    fn ensure_installed(&self, device: u32, daa: u64) -> bool {
        crate::pom_gpu::ensure_installed(device, daa)
    }

    fn uninstall(&self, device: u32) {
        crate::pom_gpu::uninstall(device)
    }

    fn mine(
        &self,
        device: u32,
        pre_pow_hash: &[u8; 32],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
    ) -> Option<u64> {
        crate::pom_gpu::mine(device, pre_pow_hash, timestamp, target_le, start, batch)
    }
}

/// Desktop Vulkan backend wrapper (AMD RDNA3 / Intel Arc), ported from keryx-miner-rdna3 as
/// `pom_gpu_vulkan_desktop`. Compiled in only when the `vulkan` Cargo feature is on, so a
/// heterogeneous-rig desktop binary (`--features cuda,vulkan`) can mine AMD/Intel cards alongside
/// NVIDIA's CUDA backend. Unlike the Android `VulkanBackend`, this one drives the multi-device
/// streamed-blob + zero-dup shared-walk paths over `keryx_vulkan::enumerate_devices`.
#[cfg(all(feature = "vulkan", not(any(target_os = "macos", target_os = "ios", target_os = "android"))))]
pub struct VulkanDesktopBackend;

#[cfg(all(feature = "vulkan", not(any(target_os = "macos", target_os = "ios", target_os = "android"))))]
impl PomGpuBackend for VulkanDesktopBackend {
    fn backend(&self) -> Backend {
        Backend::Vulkan
    }

    fn enumerate(&self) -> Vec<GpuDeviceInfo> {
        // keryx_vulkan::enumerate_devices returns every physical device in raw loader order, each
        // with its name + VRAM + discrete flag. We tag each with its vendor parsed from the name
        // (the §2.3 routing table keys off this) — AMD/Intel/ARM strings are well-known; anything
        // else falls through as Unknown and stays on Vulkan (no cross-backend fallback).
        keryx_vulkan::enumerate_devices()
            .into_iter()
            .map(|d| GpuDeviceInfo {
                handle: crate::device::GpuHandle::new(Backend::Vulkan, d.index as u32),
                name: d.name.clone(),
                vram_mb: d.vram_mb,
                vendor: parse_vulkan_vendor(&d.name),
            })
            .collect()
    }

    fn query_all_gpus_vram(&self) -> Vec<(usize, u64)> {
        keryx_vulkan::enumerate_devices()
            .into_iter()
            .map(|d| (d.index, d.vram_mb))
            .collect()
    }

    fn set_mining_tier(&self, device: u32, model_id: [u8; 32], gguf_path: String) {
        crate::pom_gpu_vulkan_desktop::set_mining_tier(device, model_id, gguf_path);
    }

    fn current_tier(&self, device: u32, daa: u64) -> Option<u8> {
        crate::pom_gpu_vulkan_desktop::current_tier(device, daa)
    }

    fn device_for_model(&self, model_id: &[u8; 32]) -> Option<u32> {
        crate::pom_gpu_vulkan_desktop::device_for_model(model_id)
    }

    fn is_installed(&self, device: u32) -> bool {
        crate::pom_gpu_vulkan_desktop::is_installed(device)
    }

    fn is_loading(&self) -> bool {
        crate::pom_gpu_vulkan_desktop::is_loading()
    }

    fn ensure_installed(&self, device: u32, daa: u64) -> bool {
        // NOTE: the desktop Vulkan module's free function takes (daa, device) — reversed from the
        // trait's (device, daa). Adapt here so the trait surface stays uniform across backends.
        crate::pom_gpu_vulkan_desktop::ensure_installed(daa, device)
    }

    fn uninstall(&self, _device: u32) {
        // The desktop Vulkan module's `uninstall()` takes no device arg — it always drops the
        // INFERENCE device's miner (inference has priority on that one device). The trait passes a
        // device for uniformity; on this backend the arg is intentionally ignored.
        crate::pom_gpu_vulkan_desktop::uninstall();
    }

    fn mine(
        &self,
        device: u32,
        pre_pow_hash: &[u8; 32],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
    ) -> Option<u64> {
        crate::pom_gpu_vulkan_desktop::mine(device, pre_pow_hash, timestamp, target_le, start, batch)
    }
}

/// Parse a Vulkan device name into a vendor tag for the §2.3 routing decision. Vulkan doesn't
/// expose a vendor enum through `ash` without extension queries, so we fall back to the
/// well-known name substrings the drivers report. NVIDIA is intentionally NOT matched here:
/// NVIDIA cards on a heterogeneous rig route to the CUDA backend (which probes them separately),
/// so a NVIDIA-named Vulkan device only appears if CUDA is absent — and then it's still mined via
/// Vulkan (the routing table's "NVIDIA → CUDA, else Vulkan" fallback, plan §2.3 / open-q #2).
#[cfg(all(feature = "vulkan", not(any(target_os = "macos", target_os = "ios", target_os = "android"))))]
fn parse_vulkan_vendor(name: &str) -> GpuVendor {
    let lower = name.to_ascii_lowercase();
    if lower.contains("amd") || lower.contains("radeon") || lower.contains("rdna") {
        GpuVendor::Amd
    } else if lower.contains("intel") || lower.contains("arc") {
        GpuVendor::Intel
    } else if lower.contains("adreno") || lower.contains("mali") || lower.contains("powervr") {
        // Mobile GPUs that surface through desktop Vulkan only in test/emulation; tagged Unknown
        // would lose them, so route mobile vendors to Vulkan (their only backend) via Unknown.
        GpuVendor::Unknown
    } else {
        GpuVendor::Unknown
    }
}

// ─── Registration helper ─────────────────────────────────────────────────────

/// Register the backends compiled into THIS binary with the `device` dispatcher. Called once from
/// the platform entry point (`main` on desktop, the JNI/`objc2` bridge on Android/iOS) after
/// logging is up. Idempotent. Phase 1 registers exactly one backend (today's per-OS exclusivity);
/// Phase 3 flips desktop Windows/Linux to register both `CudaBackend` and `VulkanDesktopBackend`
/// when the `vulkan` feature is on, so one binary mines a heterogeneous rig.
pub fn register_compiled_backends() {
    let mut backends: Vec<Arc<dyn PomGpuBackend>> = Vec::new();
    #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "android")))]
    {
        // CUDA is always compiled into the desktop build (candle-cuda is a non-optional desktop
        // dep). The desktop Vulkan backend is opt-in via the `vulkan` feature.
        backends.push(Arc::new(CudaBackend));
        #[cfg(feature = "vulkan")]
        backends.push(Arc::new(VulkanDesktopBackend));
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    backends.push(Arc::new(MetalBackend));
    #[cfg(target_os = "android")]
    backends.push(Arc::new(VulkanBackend));
    crate::device::register_backends(backends);
}
