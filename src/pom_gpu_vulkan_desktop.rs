//! Proof-of-Model GPU mining — **Vulkan** backend (RDNA3). Streams the mining tier's GGUF quant
//! bytes into per-GPU Vulkan storage buffers using the canonical name-sorted, 32-byte-chunk layout
//! (identical to [`crate::pom::WeightIndex`]), then drives the verified `keryx_vulkan` PoM walk
//! kernel to find a winning nonce. The host `WeightIndex` is also built so a winning nonce can be
//! turned into a `PomProof` (`State::generate_block_if_pom`).
//!
//! Zero-dup load path: the VRAM blob is filled straight from the GGUF on disk through the
//! `WeightIndex` chunk table (bounded 256 MiB staging window) — the packed full-model host `Vec`
//! the old loader built (~4.6 GiB for the 8B tier, ~25-40 GiB for 32B/70B) no longer exists.
//!
//! Multi-GPU, heterogeneous tiers: each mining device can hold a DIFFERENT tier (the highest its
//! VRAM fits — see main.rs's `assign_pom_tiers`), matching Keryx-Labs upstream's per-GPU design,
//! ported here from CUDA to Vulkan. `MINING_TIERS` keys the assignment by device; the host
//! possession index (`pom::active_index_for_tier`) is keyed by tier instead, so two GPUs mining
//! the SAME tier share one index build.
//!
//! The seed/walk/pow folds are byte-identical across the GPU kernel, `pom.rs`, and the node, so a
//! nonce found here builds a proof the node accepts.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use keryx_vulkan::pom_walk::PomWalkGpu;
use log::info;

/// Lowercase hex of a 32-byte digest, for diagnostics.
fn hex32(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

/// One resident PoM miner: either a miner-owned streamed weight blob, or (zero-dup) the walk
/// over the in-process inference engine's own resident weight buffers on the inference GPU —
/// no extra VRAM there. The Shared variant holds the engine Arc so the model can never be
/// freed underneath an in-flight walk dispatch (field order: walk drops first).
#[derive(Clone)]
enum Resident {
    Blob(Arc<PomWalkGpu>),
    Shared { walk: Arc<keryx_vulkan::pom_walk::PomWalkShared>, _engine: Arc<crate::llm_engine::LlamaEngine> },
}

impl Resident {
    fn mine(&self, pph: &[u8; 32], ts: u64, target: &[u8; 32], start: u64, batch: u32) -> Option<u64> {
        match self {
            Resident::Blob(m) => m.mine(pph, ts, target, start, batch),
            Resident::Shared { walk, .. } => walk.mine(pph, ts, target, start, batch),
        }
    }

    /// EXTRA VRAM the miner itself holds for this entry: the blob's n_chunks*32, or 0 for the
    /// shared walk (weights belong to the inference engine either way).
    fn extra_vram_bytes(&self) -> u64 {
        match self {
            Resident::Blob(m) => m.n_chunks() * 32,
            Resident::Shared { .. } => 0,
        }
    }
}

/// Resident GPU PoM miners, one per mining device (raw Vulkan device index). An entry is dropped
/// to free that device's VRAM (inference has priority on the inference device). The payloads are
/// `Arc`ed so `mine` can dispatch without holding the map lock — N GPUs must not serialize each
/// other's batches.
static MINERS: Mutex<Option<HashMap<u32, Resident>>> = Mutex::new(None);

/// Guards the one-time-per-tier host index build. Workers on multiple devices mining the SAME tier
/// may race into PoM activation together, but the heavy GGUF -> WeightIndex build must happen
/// exactly once per tier. One lock shared across all tiers (not per-tier) — a rare simultaneous
/// first-build of two DIFFERENT tiers on a heterogeneous rig serializes, which only costs a little
/// extra startup latency, not correctness.
fn index_build_lock() -> &'static Mutex<()> {
    static INDEX_BUILD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    INDEX_BUILD_LOCK.get_or_init(|| Mutex::new(()))
}

/// Whether the GPU PoM miner is resident and ready on `device`.
pub fn is_installed(device: u32) -> bool {
    MINERS.lock().map(|g| g.as_ref().is_some_and(|m| m.contains_key(&device))).unwrap_or(false)
}

/// The resident miner for `device`, if installed.
fn miner_on(device: u32) -> Option<Resident> {
    MINERS.lock().ok()?.as_ref()?.get(&device).cloned()
}

/// EXTRA VRAM the miner holds on the INFERENCE device, or 0 if nothing is installed there.
/// Inference uses this to decide whether the blob can stay resident alongside the served model.
/// A zero-dup shared walk reports 0 — the weights are the engine's own, so there is nothing to
/// evict. Blobs on other mining devices never compete with inference, so they are not counted.
pub fn resident_blob_bytes() -> u64 {
    miner_on(keryx_vulkan::inference_device_index() as u32).map(|m| m.extra_vram_bytes()).unwrap_or(0)
}

/// Drop the INFERENCE device's GPU PoM miner, freeing its weight-blob VRAM so inference (priority)
/// can use that GPU. Mining rebuilds the blob when it next runs there. Blobs on other mining
/// devices stay resident — they do not contend with the served model. The host `WeightIndex`
/// stays (cheap, disk-backed).
pub fn uninstall() {
    let infer = keryx_vulkan::inference_device_index() as u32;
    if let Ok(mut g) = MINERS.lock() {
        if let Some(m) = g.as_mut() {
            m.remove(&infer);
        }
    }
}

/// Search nonces `[start, start + batch)` on `device`. None if not installed or no winner.
pub fn mine(
    device: u32,
    pre_pow_hash: &[u8; 32],
    timestamp: u64,
    target_le: &[u8; 32],
    start: u64,
    batch: u64,
) -> Option<u64> {
    // Clone the (Arc-backed) entry out and dispatch lock-free: each device has exactly one
    // worker thread, and holding the map lock across a walk batch would serialize other GPUs.
    let m = miner_on(device)?;
    let batch = batch.min(u32::MAX as u64) as u32;
    m.mine(pre_pow_hash, timestamp, target_le, start, batch)
}

/// Number of in-flight one-time index/blob loads (workers intentionally paused, not stalled).
static LOADING: AtomicUsize = AtomicUsize::new(0);

/// Whether a PoM index/blob load is in progress on any device (worker intentionally paused).
pub fn is_loading() -> bool {
    LOADING.load(Ordering::Relaxed) > 0
}

/// Per-GPU mining-tier identity for rebuilds: `device_id -> (model_id, gguf_path)`. A heterogeneous
/// rig mines a different tier per GPU (the highest its VRAM holds — see main.rs's
/// `assign_pom_tiers`), so this is keyed by device rather than a single process-wide tier.
static MINING_TIERS: OnceLock<Mutex<HashMap<u32, ([u8; 32], String)>>> = OnceLock::new();

fn mining_tiers() -> &'static Mutex<HashMap<u32, ([u8; 32], String)>> {
    MINING_TIERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record a GPU's mining tier so its miner can be rebuilt after an inference swapped the model away.
pub fn set_mining_tier(device_id: u32, model_id: [u8; 32], gguf_path: String) {
    if let Ok(mut g) = mining_tiers().lock() {
        g.insert(device_id, (model_id, gguf_path));
    }
}

/// PoM tier index of a device's mining model at a given block DAA. Recomputed per block (not
/// frozen at index-build time) so the tier reindexing at the very-light hardfork (H2) is applied
/// at the exact boundary — e.g. Gemma 0→1 — rather than from a stale build-time value. The proof's
/// `tier` field MUST come from here, keyed on the block's own DAA, or a post-H2 block carries the
/// stale 4-tier index and the node rejects it (`BadWeightPath`).
pub fn current_tier(device_id: u32, daa: u64) -> Option<u8> {
    let model_id = mining_tiers().lock().ok()?.get(&device_id).map(|(id, _)| *id)?;
    crate::models::pom_tier_index(&model_id, daa)
}

/// The device that mines `model_id` (from the per-GPU tier assignment), if any. Inference for a
/// model is routed to the device that already holds it, so only that GPU pauses mining and the
/// walk can share the resident weights (zero-dup). Returns the lowest matching `device_id` when
/// several GPUs mine the same tier; `None` when no GPU is assigned this model.
pub fn device_for_model(model_id: &[u8; 32]) -> Option<u32> {
    let g = mining_tiers().lock().ok()?;
    g.iter().filter(|(_, (id, _))| id == model_id).map(|(dev, _)| *dev).min()
}

/// Models that OOM'd when loading on a given GPU: `(device_id, model_id)`. Once banlisted, that GPU
/// never retries that model (avoids a hot-spin reloading a model that doesn't fit); the OOM handler
/// downgrades the GPU to a smaller downloaded tier instead.
static OOM_BANLIST: OnceLock<Mutex<HashSet<(u32, [u8; 32])>>> = OnceLock::new();

fn oom_banlist() -> &'static Mutex<HashSet<(u32, [u8; 32])>> {
    OOM_BANLIST.get_or_init(|| Mutex::new(HashSet::new()))
}

fn is_oom_banlisted(device_id: u32, model_id: &[u8; 32]) -> bool {
    oom_banlist().lock().map(|g| g.contains(&(device_id, *model_id))).unwrap_or(false)
}

fn oom_banlist_add(device_id: u32, model_id: [u8; 32]) {
    if let Ok(mut g) = oom_banlist().lock() {
        g.insert((device_id, model_id));
    }
}

/// After a GPU fails to load its assigned tier (allocation failure), reassign it to the largest
/// **already-downloaded** PoM model strictly smaller than the failed one that hasn't itself been
/// banlisted on this GPU — so a card whose VRAM estimate was optimistic (driver overhead +
/// fragmentation) mines a smaller tier instead of idling. Returns true if a downgrade was applied.
/// No extra prefetch is needed: the candidate set is the served union (a mixed rig already
/// downloaded the smaller tiers).
fn downgrade_after_oom(device_id: u32, failed_model: &[u8; 32], daa: u64) -> bool {
    let Some(failed_tier) = crate::models::pom_tier_index(failed_model, daa) else {
        return false;
    };
    let pick = crate::slm::served_pom_specs()
        .into_iter()
        .filter_map(|s| crate::models::pom_tier_index(&s.model_id, daa).map(|t| (t, s)))
        .filter(|(t, s)| *t < failed_tier && !is_oom_banlisted(device_id, &s.model_id))
        .max_by_key(|(t, _)| *t);
    match pick {
        Some((tier, spec)) => {
            let gguf = crate::slm::gguf_path_for(spec).to_string_lossy().into_owned();
            info!("PoM[gpu{}]: allocation failure on tier {} — downgrading to tier {} ({}).", device_id, failed_tier, tier, spec.name);
            set_mining_tier(device_id, spec.model_id, gguf);
            true
        }
        None => {
            log::warn!("PoM[gpu{}]: allocation failure and no smaller downloaded tier available — this GPU will not mine PoM.", device_id);
            false
        }
    }
}

/// Ensure the GPU PoM miner is installed on `device`; build the host possession index (first
/// activation for that tier, shared across every GPU mining it) and stream the weight blob into
/// that device's VRAM if needed. Returns true when ready to mine.
///
/// `daa` MUST be the current block's live DAA score — the per-tier index's tier byte is baked in
/// on first build and reused for every proof afterwards, so if this is called with a stale/wrong
/// DAA at startup the miner declares the wrong tier for the rest of the process's life
/// (BadWeightPath / "block invalid" on every submission, exactly like the pre-H2/post-H2
/// tier-index regression on the CUDA fork).
pub fn ensure_installed(daa: u64, device: u32) -> bool {
    if is_installed(device) {
        return true;
    }
    LOADING.fetch_add(1, Ordering::Relaxed);
    let ok = ensure_installed_inner(daa, device);
    LOADING.fetch_sub(1, Ordering::Relaxed);
    ok
}

fn ensure_installed_inner(daa: u64, device: u32) -> bool {
    let (model_id, gguf) = match mining_tiers().lock().ok().and_then(|g| g.get(&device).cloned()) {
        Some(x) => x,
        None => return false,
    };
    // This GPU's tier at the current block DAA (recomputed per block, H2-gated).
    let tier = match crate::models::pom_tier_index(&model_id, daa) {
        Some(t) => t,
        None => return false,
    };
    if is_oom_banlisted(device, &model_id) {
        return false; // this model OOM'd on this GPU before — don't retry (avoids a hot reload spin).
    }

    // Build THIS tier's possession index once (host, heavy) — deferred from boot so the pre-PoM
    // legacy phase starts immediately, and keyed by tier so a mixed rig builds one index per
    // distinct tier it mines (shared across every GPU on that tier). Also doubles as the zero-dup
    // GPU upload source (its chunk table maps canonical chunks to GGUF file offsets).
    if crate::pom::active_index_for_tier(tier).is_none() {
        let _guard = match index_build_lock().lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if crate::pom::active_index_for_tier(tier).is_none() {
            // Defer the heavy index build until the mining-tier GGUF is fully downloaded. The `.ok`
            // sentinel sits next to model.gguf (written by slm after a verified download). Building
            // from a partial GGUF fails with a confusing partial-read/ENOENT; returning false here
            // just lets the mining loop retry on its next tick once the download lands.
            let model_ready = std::path::Path::new(&gguf)
                .parent()
                .map(|d| d.join(".ok").exists())
                .unwrap_or(false);
            if !model_ready {
                info!("PoM: tier {} model not fully downloaded yet (.ok absent) — deferring index build.", tier);
                return false;
            }

            info!("PoM: building host weight index for tier {} (gpu{}) — this can take a while…", tier, device);
            // Enforces the consensus-pinned (R_T, N) for this tier — a wrong-quant / corrupt /
            // truncated GGUF is rejected HERE, once, instead of silently producing PoM blocks
            // every one of which the node rejects with BadWeightPath.
            let expected = crate::models::pinned_pom_anchor(&model_id);
            match crate::pom::WeightIndex::build_from_gguf(&gguf) {
                Ok(idx) => {
                    if let Some(anchor) = expected {
                        if idx.n_chunks != anchor.chunks {
                            log::error!(
                                "PoM: index chunk count {} != consensus-pinned {} for tier {} — wrong/corrupt GGUF; refusing to mine",
                                idx.n_chunks, anchor.chunks, tier
                            );
                            return false;
                        }
                        if idx.r_t != anchor.root {
                            log::error!(
                                "PoM: computed R_T {} != consensus-pinned root for tier {} — wrong/corrupt GGUF; refusing to mine",
                                hex32(&idx.r_t), tier
                            );
                            return false;
                        }
                        info!("PoM: index R_T + N match the consensus-pinned anchor for tier {}.", tier);
                    } else {
                        log::warn!("PoM: no consensus-pinned anchor for this model_id — skipping the R_T/N check.");
                    }
                    info!("PoM: tier {} host index ready — N={} chunks", tier, idx.n_chunks);
                    crate::pom::set_index(tier, idx);
                }
                Err(e) => {
                    log::error!("PoM: host index build failed for tier {}: {}", tier, e);
                    return false;
                }
            }
        }
    }

    let idx = match crate::pom::active_index_for_tier(tier) {
        Some(x) => x,
        None => return false,
    };

    // Zero-dup: on the inference GPU, walk the in-process engine's own resident weight
    // buffers — 0 extra VRAM — instead of installing a second copy. Any failure falls back
    // to the streamed blob below (correct, just costs the duplicate VRAM).
    if device == keryx_vulkan::inference_device_index() as u32 {
        match install_shared(&idx, &model_id) {
            Ok(entry) => {
                if let Ok(mut g) = MINERS.lock() {
                    g.get_or_insert_with(HashMap::new).insert(device, entry);
                }
                return true;
            }
            Err(e) => {
                log::warn!("PoM(zero-dup): shared install failed ({e}) — falling back to the streamed blob");
            }
        }
    }

    // Stream the canonical weight blob from the GGUF straight into this device's VRAM through the
    // index's chunk table — no packed host copy. The blob N equals the index N by construction
    // (same table), so the proof-vs-blob N-guard the packed loader needed is structural here. A
    // load failure surfaces as an Err or, on some Vulkan drivers, a panic; catch both so the OOM
    // handler can banlist + downgrade instead of crashing the mining thread or hot-spinning on a
    // model that doesn't fit this GPU.
    info!("PoM(vulkan): streaming weight blob into VRAM on device {}…", device);
    let idx_for_stream = idx.clone();
    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let mut source = |first_chunk: u64, out: &mut [u8]| {
            idx_for_stream.read_chunk_range(first_chunk, out).map_err(|e| format!("GGUF chunk stream failed: {e}"))
        };
        PomWalkGpu::new_streamed(Some(device as usize), idx_for_stream.n_chunks, &mut source)
    }));
    match loaded {
        Ok(Ok(gpu)) => {
            info!(
                "PoM(vulkan): GPU miner ready on {} (device {}) — N={} chunks resident",
                gpu.device_name(),
                device,
                idx.n_chunks
            );
            if let Ok(mut g) = MINERS.lock() {
                g.get_or_insert_with(HashMap::new).insert(device, Resident::Blob(Arc::new(gpu)));
            }
            true
        }
        Ok(Err(e)) => {
            log::error!("PoM(vulkan)[gpu{}]: device miner build failed: {} — banlisting this model and downgrading.", device, e);
            oom_banlist_add(device, model_id);
            downgrade_after_oom(device, &model_id, daa);
            false
        }
        Err(_) => {
            log::error!("PoM(vulkan)[gpu{}]: device miner load panicked (likely an allocation failure) — banlisting this model and downgrading.", device);
            oom_banlist_add(device, model_id);
            downgrade_after_oom(device, &model_id, daa);
            false
        }
    }
}

/// Build the zero-dup shared walk over the inference engine's resident weights and refuse to
/// mine on any layout disagreement with the possession index: per-tensor chunk table in the
/// canonical name-sorted order, hard N equality, plus a random chunk sample fetched through the
/// GPU table and compared byte-for-byte against the GGUF-backed index — a mismatch would mean
/// every mined block gets rejected, so it aborts the shared path entirely.
fn install_shared(idx: &crate::pom::WeightIndex, model_id: &[u8; 32]) -> Result<Resident, String> {
    // The engine must be serving the MINING model (post-PoM: serving == mining tier). This
    // loads it resident on the inference GPU if it is not already.
    if !crate::slm::ensure_loaded(model_id) {
        return Err("inference engine failed to load the mining model".into());
    }
    let engine = crate::slm::active_engine(model_id).ok_or("engine not resident after ensure_loaded")?;

    // Canonical table over the engine's tensors: VK-resident walked in place, host-side ones
    // (llama keeps token_embd on CPU under Vulkan) supplemented into a small owned buffer.
    let (walk, supl_bytes) = engine.build_shared_walk().map_err(|e| e.to_string())?;

    if walk.n_chunks() != idx.n_chunks {
        return Err(format!(
            "shared table N={} != index N={} — layout mismatch, refusing to mine on shared weights",
            walk.n_chunks(),
            idx.n_chunks
        ));
    }

    // Sample-verify: 256 deterministic pseudo-random chunks through the GPU table vs the index.
    // Catches any addressing/ordering error with overwhelming probability before a single
    // nonce is mined off the shared weights.
    let mut x: u64 = 0x9E3779B97F4A7C15 ^ idx.n_chunks;
    for _ in 0..256 {
        // splitmix64 step
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let off = (z ^ (z >> 31)) % idx.n_chunks;

        let gpu = walk.read_chunk(off);
        let mut gpu_words = [0u64; 4];
        for (i, w) in gpu_words.iter_mut().enumerate() {
            *w = u64::from_le_bytes(gpu[i * 8..i * 8 + 8].try_into().unwrap());
        }
        if gpu_words != idx.read_chunk(off) {
            return Err(format!("chunk {off} differs between shared weights and the possession index"));
        }
    }

    info!(
        "PoM(zero-dup): walking inference's resident weights on {} — N={} chunks, {} MB supplemented \
         (host-side tensors), rest walked in place (sample-verify passed)",
        walk.device_name(),
        walk.n_chunks(),
        supl_bytes / (1024 * 1024)
    );
    Ok(Resident::Shared { walk: Arc::new(walk), _engine: engine })
}
