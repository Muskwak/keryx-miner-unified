/// Phase-3 OPoI: multi-model inference engine (safetensors + GGUF) via candle.
///
/// Models are loaded on demand when an AiRequest arrives and cached between
/// consecutive requests for the same model. Mining pauses during inference.
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, LlamaConfig, Llama};
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3Weights;
use crate::quantized_llama_split::ModelWeights as SplitWeights;
use crate::quantized_qwen2_split::ModelWeights as Qwen2SplitWeights;
use crate::quantized_qwen3_moe_split::ModelWeights as Qwen3MoeSplitWeights;
use crate::quantized_qwen3_split::ModelWeights as Qwen3SplitWeights;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;

use crate::models::{ModelFormat, ModelSpec};

const IPFS_GATEWAY: &str = "https://keryx-labs.com";
const SYSTEM_PROMPT_TINYLLAMA: &str =
    "You are a Keryx Network AI — a decentralized assistant running on GPU miners. \
     No internet access. Be concise.";

const SYSTEM_PROMPT_DEEPSEEK: &str =
    "You are a Keryx Network AI — a decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention DeepSeek, Anthropic, OpenAI, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be concise.";

const SYSTEM_PROMPT_LLAMA70B: &str =
    "You are a Keryx Network AI — a high-capability decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention Meta, Llama, OpenAI, Anthropic, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be thorough but concise.";

const SYSTEM_PROMPT_QWEN3: &str =
    "You are a Keryx Network AI — a high-capability decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention Qwen, Alibaba, OpenAI, Anthropic, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be thorough but concise.";

// ── Static engine state ──────────────────────────────────────────────────────

static SUPPORTED_SPECS: OnceLock<&'static [&'static ModelSpec]> = OnceLock::new();
static ENGINE: Mutex<Option<SlmEngine>> = Mutex::new(None);

/// When true, OPoI inference is forced onto the CPU even if a CUDA device is present.
/// Set once at startup from the `--cpu-inference` CLI flag so weak GPUs (e.g. GTX 1060)
/// can dedicate the GPU to hashing while the CPU runs the SLM.
static FORCE_CPU_INFERENCE: AtomicBool = AtomicBool::new(false);

/// Force OPoI inference onto the CPU (see [`FORCE_CPU_INFERENCE`]). Call once at startup.
pub fn set_cpu_inference(enabled: bool) {
    FORCE_CPU_INFERENCE.store(enabled, AtomicOrdering::Relaxed);
}

/// Whether OPoI inference is forced onto the CPU.
pub fn cpu_inference_enabled() -> bool {
    FORCE_CPU_INFERENCE.load(AtomicOrdering::Relaxed)
}

/// When true, GGUF models are loaded with their layers split evenly across all
/// available CUDA devices (layer-split VRAM pooling). Set once at startup from
/// the `--vram-pool` CLI flag. EXPERIMENTAL.
static VRAM_POOL: AtomicBool = AtomicBool::new(false);

/// Enable layer-split VRAM pooling across all CUDA GPUs (see [`VRAM_POOL`]). Call once at startup.
pub fn set_vram_pool(enabled: bool) {
    VRAM_POOL.store(enabled, AtomicOrdering::Relaxed);
}

/// Whether layer-split VRAM pooling is enabled.
pub fn vram_pool_enabled() -> bool {
    VRAM_POOL.load(AtomicOrdering::Relaxed)
}

enum ModelInner {
    Full { model: Llama, config: Config, cache_dtype: DType },
    Quantized(ModelWeights),
    /// GGUF llama-arch model with layers split across multiple CUDA devices (--vram-pool).
    QuantizedSplit(SplitWeights),
    QuantizedQwen2(Qwen2Weights),
    /// GGUF Qwen2-arch model (DeepSeek-R1-32B) with layers split across CUDA devices (--vram-pool).
    QuantizedQwen2Split(Qwen2SplitWeights),
    QuantizedQwen3(Qwen3Weights),
    /// GGUF Qwen3-arch dense model (Qwen3-32B) with layers split across CUDA devices (--vram-pool).
    QuantizedQwen3Split(Qwen3SplitWeights),
    /// GGUF Qwen3-MoE model (Qwen3-235B), layers + stacked experts split across
    /// CUDA devices (--vram-pool). Always loaded via the split loader.
    QuantizedQwen3MoeSplit(Qwen3MoeSplitWeights),
}

struct SlmEngine {
    model_id: [u8; 32],
    name: &'static str,
    inner: ModelInner,
    tokenizer: Tokenizer,
    device: Device,
    /// All token IDs that terminate generation (EOS, EOT, role-start tokens, etc.).
    stop_token_ids: Vec<u32>,
    /// Literal stop strings — safety net for tokenizers that emit control markers
    /// as plain text (e.g. a GGUF whose tokenizer.json lacks the ChatML special
    /// tokens) so the matching `stop_token_ids` are never produced. Generation is
    /// cut at the earliest occurrence of any of these in the decoded output.
    stop_strings: Vec<&'static str>,
}

unsafe impl Send for SlmEngine {}
unsafe impl Sync for SlmEngine {}

// ── File management ──────────────────────────────────────────────────────────

fn model_dir(spec: &ModelSpec) -> std::path::PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    exe_dir.join("models").join(spec.dir_name)
}

/// Downloads `url` to `dest` with automatic resume. A partially downloaded file is
/// continued via an HTTP `Range` request instead of restarting from zero, and both
/// connect-time and mid-stream failures are retried with a fixed backoff. Designed
/// for the huge (10-40 GB) model GGUFs served over the flaky IPFS gateway: the
/// content is immutable (CID-addressed), so appending resumed bytes is always
/// consistent, and an already-complete file (e.g. pre-staged with `wget -c`) is
/// detected via a 416 response and left untouched instead of being re-downloaded.
fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    const MAX_ATTEMPTS: u32 = 240; // survives long gateway outages (~40 min of retries)
    const BACKOFF_SECS: u64 = 10;
    eprintln!("[keryx-miner] Downloading {} ...", url);
    let mut attempt = 0u32;
    loop {
        // Resume offset = how many bytes we already have on disk.
        let resume_from = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);

        let mut req = ureq::get(url);
        if resume_from > 0 {
            req = req.set("Range", &format!("bytes={}-", resume_from));
        }
        let response = match req.call() {
            Ok(r) => r,
            Err(e) => {
                attempt += 1;
                if attempt >= MAX_ATTEMPTS {
                    return Err(anyhow!("HTTP GET {} failed after {} attempts: {}", url, attempt, e));
                }
                eprintln!("\n[keryx-miner] connect error ({e}); retry {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s (resume @ {} MB)…",
                    resume_from / 1_000_000);
                std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
                continue;
            }
        };
        let status = response.status();

        // Decide whether to append (server honored the range) or (re)start, and the total size.
        let (mut file, mut downloaded, total): (std::fs::File, u64, Option<u64>) =
            if resume_from > 0 && status == 206 {
                // Content-Range: "bytes <start>-<end>/<total>"
                let total = response
                    .header("Content-Range")
                    .and_then(|cr| cr.rsplit('/').next())
                    .and_then(|t| t.trim().parse::<u64>().ok());
                let f = std::fs::OpenOptions::new()
                    .append(true)
                    .open(dest)
                    .with_context(|| format!("open append {}", dest.display()))?;
                (f, resume_from, total)
            } else if resume_from > 0 && status == 416 {
                // Range not satisfiable ⇒ the file is already fully downloaded.
                eprintln!("\r  already complete ({} MB).            ", resume_from / 1_000_000);
                return Ok(());
            } else {
                // 200, or the server ignored Range ⇒ (re)start from scratch.
                let total = response.header("Content-Length").and_then(|s| s.parse::<u64>().ok());
                let f = std::fs::File::create(dest)
                    .with_context(|| format!("create {}", dest.display()))?;
                (f, 0u64, total)
            };

        let mut reader = response.into_reader();
        let mut buf = vec![0u8; 65_536];
        let mut stream_err: Option<String> = None;
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if let Err(e) = file.write_all(&buf[..n]) {
                        stream_err = Some(e.to_string());
                        break;
                    }
                    downloaded += n as u64;
                    if let Some(t) = total {
                        eprint!("\r  {:.1}/{:.1} MB ({}%)   ",
                            downloaded as f64 / 1_000_000.0,
                            t as f64 / 1_000_000.0,
                            downloaded * 100 / t.max(1));
                        let _ = std::io::stderr().flush();
                    }
                }
                Err(e) => { stream_err = Some(e.to_string()); break; }
            }
        }
        let _ = file.flush();

        // Done only if the stream ended cleanly AND we reached the known total.
        let complete = stream_err.is_none() && total.map_or(true, |t| downloaded >= t);
        if complete {
            eprintln!();
            return Ok(());
        }

        attempt += 1;
        if attempt >= MAX_ATTEMPTS {
            return Err(anyhow!("download {} interrupted after {} attempts (got {} MB)",
                url, attempt, downloaded / 1_000_000));
        }
        let why = stream_err.unwrap_or_else(|| "short read".into());
        eprintln!("\n[keryx-miner] interrupted ({why}); resuming {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s @ {} MB…",
            downloaded / 1_000_000);
        std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
    }
}

fn ipfs_url(cid: &str) -> String {
    format!("{}/ipfs/{}", IPFS_GATEWAY, cid)
}

fn ensure_safetensors(spec: &ModelSpec) -> Result<(std::path::PathBuf, std::path::PathBuf, Vec<std::path::PathBuf>)> {
    let dir = model_dir(spec);
    let tok = dir.join("tokenizer.json");
    let cfg = dir.join("config.json");
    let ok_flag = dir.join(".ok");
    let wts: Vec<_> = spec.weight_cids.iter().enumerate().map(|(i, _)| {
        if spec.weight_cids.len() == 1 { dir.join("model.safetensors") }
        else { dir.join(format!("model-{:05}-of-{:05}.safetensors", i + 1, spec.weight_cids.len())) }
    }).collect();

    // .ok sentinel written only after a complete download — guards against truncated files
    if tok.exists() && cfg.exists() && wts.iter().all(|p| p.exists()) && ok_flag.exists() {
        log::info!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
        return Ok((tok, cfg, wts));
    }
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(&ok_flag); // clear stale flag before re-downloading
    eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
    if !tok.exists() { download_file(&ipfs_url(spec.tokenizer_cid), &tok)?; }
    if !cfg.exists() { download_file(&ipfs_url(spec.config_cid), &cfg)?; }
    for (i, (cid, path)) in spec.weight_cids.iter().zip(wts.iter()).enumerate() {
        if spec.weight_cids.len() > 1 { eprintln!("[keryx-miner] Shard {}/{}", i + 1, spec.weight_cids.len()); }
        download_file(&ipfs_url(cid), path)?;
    }
    std::fs::write(&ok_flag, b"").with_context(|| format!("write .ok flag {}", ok_flag.display()))?;
    eprintln!("[keryx-miner] Model '{}' ready.\n", spec.name);
    Ok((tok, cfg, wts))
}

fn ensure_gguf(spec: &ModelSpec) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let dir = model_dir(spec);
    let tok = dir.join("tokenizer.json");
    let gguf = dir.join("model.gguf");
    let ok_flag = dir.join(".ok");

    // .ok sentinel written only after a complete download — guards against truncated files
    if tok.exists() && gguf.exists() && ok_flag.exists() {
        log::info!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
        return Ok((tok, gguf));
    }
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(&ok_flag); // clear stale flag before re-downloading
    eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
    if !tok.exists() { download_file(&ipfs_url(spec.tokenizer_cid), &tok)?; }
    download_file(&ipfs_url(spec.weight_cids[0]), &gguf)?;
    std::fs::write(&ok_flag, b"").with_context(|| format!("write .ok flag {}", ok_flag.display()))?;
    eprintln!("[keryx-miner] Model '{}' ready.\n", spec.name);
    Ok((tok, gguf))
}

// ── Engine loading ───────────────────────────────────────────────────────────

/// Build the list of stop token IDs for a model.
///
/// Tries `token_to_id` for each name first; falls back to the corresponding
/// hardcoded ID so generation always terminates even if the tokenizer exposes
/// special tokens differently (e.g. via `added_tokens` vs the regular vocab).
fn collect_stop_ids(tokenizer: &Tokenizer, names: &[&str], fallbacks: &[u32]) -> Vec<u32> {
    let mut ids: Vec<u32> = names.iter().zip(fallbacks.iter())
        .map(|(name, &fallback)| tokenizer.token_to_id(name).unwrap_or(fallback))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Per-model terminating tokens and literal stop strings, keyed by model name so
/// it stays coherent with `format_prompt`. Two models can share a `ModelFormat`
/// (DeepSeek-R1-8B and LLaMA-3.3-70B are both LLaMA-arch GGUF) yet need different
/// stop conventions, so this branches on name rather than format.
fn stop_config(tokenizer: &Tokenizer, name: &str) -> (Vec<u32>, Vec<&'static str>) {
    match name {
        // DeepSeek-R1-Distill-Llama-8B — DeepSeek chat template:
        //   128001 = <｜end▁of▁sentence｜> (real EOS), 128011 = <｜User｜> (new turn),
        //   128009 = <|eot_id|> (LLaMA-3 EOT, kept as a fallback).
        "deepseek-r1-8b" => (
            collect_stop_ids(tokenizer,
                &["<｜end▁of▁sentence｜>", "<｜User｜>", "<|eot_id|>"],
                &[128001, 128011, 128009]),
            vec!["<｜end▁of▁sentence｜>", "<｜User｜>", "<|eot_id|>", "<|end_of_text|>"],
        ),
        // DeepSeek-R1-Distill-Qwen-32B — DeepSeek chat template (NOT ChatML):
        //   151643 = <｜end▁of▁sentence｜> (real EOS), 151644 = <｜User｜> (new turn).
        //   (151645 is <｜Assistant｜>, NOT an end token — must not stop on it.)
        "deepseek-r1-32b" => (
            collect_stop_ids(tokenizer,
                &["<｜end▁of▁sentence｜>", "<｜User｜>"],
                &[151643, 151644]),
            // ASCII ChatML markers kept as an extra net if the model parrots them.
            vec!["<｜end▁of▁sentence｜>", "<｜User｜>", "<|im_end|>", "<|im_start|>"],
        ),
        // Genuine LLaMA-3.3-70B-Instruct — LLaMA-3 header template. The DeepSeek
        // full-width markers do NOT exist in this vocab, so stop on the real
        // LLaMA-3 terminators (the official `eos_token_id` set for 3.3 Instruct):
        //   128009 = <|eot_id|> (end of turn), 128001 = <|end_of_text|> (base EOS),
        //   128008 = <|eom_id|> (end of message, tool turns).
        "llama-3.3-70b" => (
            collect_stop_ids(tokenizer,
                &["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"],
                &[128009, 128001, 128008]),
            // Cut if the model tries to open a fresh turn instead of stopping.
            vec!["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
        ),
        // Qwen3-32B / Qwen3-235B — ChatML template:
        //   151645 = <|im_end|> (end of turn), 151643 = <|endoftext|> (base EOS).
        "qwen3-32b" | "qwen3-235b" => (
            collect_stop_ids(tokenizer,
                &["<|im_end|>", "<|endoftext|>"],
                &[151645, 151643]),
            // Cut if the model opens a fresh turn instead of stopping.
            vec!["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        ),
        // TinyLlama / Zephyr: </s> ends a turn; 0 = padding safety net.
        _ => (
            collect_stop_ids(tokenizer, &["</s>"], &[2, 0]),
            vec!["</s>", "<|user|>", "<|system|>", "<|assistant|>"],
        ),
    }
}

/// Enumerate every available CUDA device for --vram-pool, starting from the
/// already-open primary device (ordinal 0). Stops at the first ordinal that
/// fails to open.
fn cuda_pool_devices(primary: &Device) -> Vec<Device> {
    let mut devices = vec![primary.clone()];
    for ordinal in 1.. {
        match Device::new_cuda(ordinal) {
            Ok(d) => devices.push(d),
            Err(_) => break,
        }
    }
    devices
}

/// Total VRAM (MB) of the smallest CUDA card in the rig, via `nvidia-smi`.
/// Conservative (min across cards) so a model that fits the smallest card can be
/// loaded on a single GPU. Returns `None` if nvidia-smi is unavailable.
fn min_gpu_vram_mb() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .min()
}

/// VRAM headroom (MB) reserved on top of the GGUF weight footprint for the KV
/// cache, activations and CUDA workspace when deciding single-GPU vs layer-split.
const SINGLE_GPU_HEADROOM_MB: u64 = 2_048;

/// Under `--vram-pool`, decide whether a GGUF model must be layer-split across
/// GPUs. A model that comfortably fits one card is loaded on a single GPU to avoid
/// needless inter-GPU pipeline overhead (e.g. TinyLlama on a 6-GPU rig). When the
/// per-card VRAM cannot be determined, defaults to splitting (previous behaviour).
fn model_needs_pooling(gguf_path: &std::path::Path) -> bool {
    let Some(card_mb) = min_gpu_vram_mb() else {
        return true;
    };
    let model_mb = std::fs::metadata(gguf_path)
        .map(|m| m.len() / (1024 * 1024))
        .unwrap_or(u64::MAX);
    model_mb.saturating_add(SINGLE_GPU_HEADROOM_MB) > card_mb
}

fn load_engine(spec: &'static ModelSpec, device: Device) -> Result<SlmEngine> {
    log::info!("SlmEngine: loading '{}'…", spec.name);

    match spec.format {
        ModelFormat::Safetensors => {
            let (tok_path, cfg_path, wt_paths) = ensure_safetensors(spec)?;
            let config: LlamaConfig = serde_json::from_str(
                &std::fs::read_to_string(&cfg_path)?
            ).context("parse config.json")?;
            let config = config.into_config(false);
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let wt_refs: Vec<_> = wt_paths.iter().map(|p| p.as_path()).collect();
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&wt_refs, DType::F32, &device)
            }.map_err(|e| anyhow!("mmap weights: {}", e))?;
            let model = Llama::load(vb, &config).map_err(|e| anyhow!("build model: {}", e))?;
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner: ModelInner::Full { model, config, cache_dtype: DType::F32 },
                tokenizer, device, stop_token_ids, stop_strings,
            })
        }
        ModelFormat::Gguf => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file = std::fs::File::open(&gguf_path)
                .with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file)
                .map_err(|e| anyhow!("read gguf: {}", e))?;
            // --vram-pool: split layers evenly across every CUDA device so the
            // pooled VRAM can hold models no single card fits. Falls back to a
            // regular single-device load when only one GPU is present.
            let inner = if vram_pool_enabled() && device.is_cuda() && model_needs_pooling(&gguf_path) {
                let devices = cuda_pool_devices(&device);
                if devices.len() >= 2 {
                    log::info!(
                        "SlmEngine: --vram-pool — splitting '{}' layers evenly across {} GPUs",
                        spec.name,
                        devices.len()
                    );
                    let model = SplitWeights::from_gguf(content, &mut gguf_file, &devices)
                        .map_err(|e| anyhow!("load gguf weights (split): {}", e))?;
                    ModelInner::QuantizedSplit(model)
                } else {
                    log::warn!(
                        "SlmEngine: --vram-pool set but only one CUDA device found — single-GPU load"
                    );
                    let model = ModelWeights::from_gguf(content, &mut gguf_file, &device)
                        .map_err(|e| anyhow!("load gguf weights: {}", e))?;
                    ModelInner::Quantized(model)
                }
            } else {
                if vram_pool_enabled() && device.is_cuda() {
                    log::info!(
                        "SlmEngine: '{}' fits a single GPU — loading on one device (no --vram-pool split)",
                        spec.name
                    );
                }
                let model = ModelWeights::from_gguf(content, &mut gguf_file, &device)
                    .map_err(|e| anyhow!("load gguf weights: {}", e))?;
                ModelInner::Quantized(model)
            };
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner,
                tokenizer, device, stop_token_ids, stop_strings,
            })
        }
        ModelFormat::GgufQwen2 => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file = std::fs::File::open(&gguf_path)
                .with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file)
                .map_err(|e| anyhow!("read gguf: {}", e))?;
            // --vram-pool: split layers across every CUDA device (e.g. DeepSeek-R1-32B
            // served by a rig of 8 GB cards). Falls back to single-device otherwise.
            let inner = if vram_pool_enabled() && device.is_cuda() && model_needs_pooling(&gguf_path) {
                let devices = cuda_pool_devices(&device);
                if devices.len() >= 2 {
                    log::info!(
                        "SlmEngine: --vram-pool — splitting '{}' (Qwen2) layers evenly across {} GPUs",
                        spec.name,
                        devices.len()
                    );
                    let model = Qwen2SplitWeights::from_gguf(content, &mut gguf_file, &devices)
                        .map_err(|e| anyhow!("load qwen2 gguf weights (split): {}", e))?;
                    ModelInner::QuantizedQwen2Split(model)
                } else {
                    log::warn!(
                        "SlmEngine: --vram-pool set but only one CUDA device found — single-GPU load"
                    );
                    let model = Qwen2Weights::from_gguf(content, &mut gguf_file, &device)
                        .map_err(|e| anyhow!("load qwen2 gguf weights: {}", e))?;
                    ModelInner::QuantizedQwen2(model)
                }
            } else {
                if vram_pool_enabled() && device.is_cuda() {
                    log::info!(
                        "SlmEngine: '{}' fits a single GPU — loading on one device (no --vram-pool split)",
                        spec.name
                    );
                }
                let model = Qwen2Weights::from_gguf(content, &mut gguf_file, &device)
                    .map_err(|e| anyhow!("load qwen2 gguf weights: {}", e))?;
                ModelInner::QuantizedQwen2(model)
            };
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner,
                tokenizer, device, stop_token_ids, stop_strings,
            })
        }
        ModelFormat::GgufQwen3 => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file = std::fs::File::open(&gguf_path)
                .with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file)
                .map_err(|e| anyhow!("read gguf: {}", e))?;
            // --vram-pool: split layers across every CUDA device (Qwen3-32B served by
            // a multi-GPU rig). Falls back to single-device otherwise.
            let inner = if vram_pool_enabled() && device.is_cuda() && model_needs_pooling(&gguf_path) {
                let devices = cuda_pool_devices(&device);
                if devices.len() >= 2 {
                    log::info!(
                        "SlmEngine: --vram-pool — splitting '{}' (Qwen3) layers evenly across {} GPUs",
                        spec.name,
                        devices.len()
                    );
                    let model = Qwen3SplitWeights::from_gguf(content, &mut gguf_file, &devices)
                        .map_err(|e| anyhow!("load qwen3 gguf weights (split): {}", e))?;
                    ModelInner::QuantizedQwen3Split(model)
                } else {
                    log::warn!(
                        "SlmEngine: --vram-pool set but only one CUDA device found — single-GPU load"
                    );
                    let model = Qwen3Weights::from_gguf(content, &mut gguf_file, &device)
                        .map_err(|e| anyhow!("load qwen3 gguf weights: {}", e))?;
                    ModelInner::QuantizedQwen3(model)
                }
            } else {
                if vram_pool_enabled() && device.is_cuda() {
                    log::info!(
                        "SlmEngine: '{}' fits a single GPU — loading on one device (no --vram-pool split)",
                        spec.name
                    );
                }
                let model = Qwen3Weights::from_gguf(content, &mut gguf_file, &device)
                    .map_err(|e| anyhow!("load qwen3 gguf weights: {}", e))?;
                ModelInner::QuantizedQwen3(model)
            };
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner,
                tokenizer, device, stop_token_ids, stop_strings,
            })
        }
        ModelFormat::GgufQwen3Moe => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file = std::fs::File::open(&gguf_path)
                .with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file)
                .map_err(|e| anyhow!("read gguf: {}", e))?;
            // Qwen3-235B needs pooled VRAM: always load via the split loader. With
            // --vram-pool it spreads layers + stacked experts across every CUDA
            // device; without it, a single device (which will OOM unless huge —
            // the OPoI VRAM gate normally prevents announcing it on such rigs).
            let devices = if vram_pool_enabled() && device.is_cuda() {
                cuda_pool_devices(&device)
            } else {
                vec![device.clone()]
            };
            log::info!(
                "SlmEngine: '{}' (Qwen3-MoE) — splitting layers across {} device(s)",
                spec.name, devices.len()
            );
            let model = Qwen3MoeSplitWeights::from_gguf(content, &mut gguf_file, &devices)
                .map_err(|e| anyhow!("load qwen3-moe gguf weights (split): {}", e))?;
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner: ModelInner::QuantizedQwen3MoeSplit(model),
                tokenizer, device, stop_token_ids, stop_strings,
            })
        }
    }
}

// ── Inference ────────────────────────────────────────────────────────────────

fn format_prompt(engine: &SlmEngine, prompt: &str) -> String {
    match engine.name {
        // DeepSeek-R1-Distill-Qwen-32B — DeepSeek chat template.
        // The 32B respects the system prompt, so plain think priming is enough.
        "deepseek-r1-32b" => format!(
            "<｜begin▁of▁sentence｜>{}<｜User｜>{}<｜Assistant｜><think>\n",
            SYSTEM_PROMPT_DEEPSEEK, prompt
        ),
        // DeepSeek-R1-Distill-Llama-8B — same template but the 8B ignores system
        // prompts about identity due to RLHF. Inject an identity statement at the
        // start of the think block so the model reasons from the correct framing
        // before generating the visible answer.
        "deepseek-r1-8b" => format!(
            "<｜begin▁of▁sentence｜>{}<｜User｜>{}<｜Assistant｜><think>\nI am Keryx Network AI, a decentralized assistant. I must never claim to be DeepSeek or any other AI product.\n",
            SYSTEM_PROMPT_DEEPSEEK, prompt
        ),
        "llama-3.3-70b" => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n",
            SYSTEM_PROMPT_LLAMA70B, prompt
        ),
        // Qwen3-32B / Qwen3-235B — ChatML template. `/no_think` disables the
        // thinking block so the assistant answers directly (no <think>…</think>).
        "qwen3-32b" | "qwen3-235b" => format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{} /no_think<|im_end|>\n\
             <|im_start|>assistant\n",
            SYSTEM_PROMPT_QWEN3, prompt
        ),
        // Zephyr/TinyLlama chat template
        _ => format!(
            "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            SYSTEM_PROMPT_TINYLLAMA, prompt
        ),
    }
}

/// Repetition penalty applied over a recent token window before sampling.
/// Breaks degenerate loops where the model repeats a phrase instead of emitting EOS
/// (common on distilled R1 models). 1.0 = disabled.
const REPEAT_PENALTY: f32 = 1.15;
const REPEAT_LAST_N: usize = 64;

/// True if any stop string appears in the decoded tail of `generated`.
/// Only the last few tokens are decoded — enough to catch a marker that just
/// completed — keeping the per-step cost O(1) instead of re-decoding everything.
fn hit_stop_string(tokenizer: &Tokenizer, generated: &[u32], stops: &[&str]) -> bool {
    if stops.is_empty() || generated.is_empty() {
        return false;
    }
    let start = generated.len().saturating_sub(24);
    match tokenizer.decode(&generated[start..], true) {
        Ok(tail) => stops.iter().any(|s| tail.contains(s)),
        Err(_) => false,
    }
}

/// Strip the `<think>…</think>` reasoning block emitted by DeepSeek-R1 models.
///
/// The prompt primes the assistant turn with `<think>` already open, so the
/// generated text begins inside the block: `reasoning…\n</think>\nfinal answer`.
/// Returns the trimmed final answer, or an empty string if `</think>` was never
/// emitted (max_tokens exhausted during reasoning — no final answer produced).
fn strip_think_block(text: &str) -> String {
    match text.find("</think>") {
        Some(end) => text[end + "</think>".len()..].trim().to_string(),
        None => String::new(),
    }
}

/// Strip a self-emitted `<think>…</think>` block from models that are NOT primed
/// with an already-open `<think>` (e.g. Qwen3 with `/no_think` still emits an empty
/// `<think></think>` pair). Unlike [`strip_think_block`], when no closing tag is
/// present the original text is returned (the model answered directly) — never an
/// empty string — so a direct answer is preserved.
fn strip_think_tags(text: &str) -> String {
    match text.find("</think>") {
        Some(end) => text[end + "</think>".len()..].trim().to_string(),
        None => text.trim().to_string(),
    }
}

fn generate(engine: &mut SlmEngine, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let formatted = format_prompt(engine, prompt);
    // Detect whether the assistant turn was primed with <think> (R1 models).
    // Uses contains() to handle both plain priming and think-injected variants.
    let is_think_primed = formatted.contains("<｜Assistant｜><think>");
    let enc = engine.tokenizer.encode(formatted.as_str(), true)
        .map_err(|e| anyhow!("encode: {}", e))?;
    let mut all_tokens: Vec<u32> = enc.get_ids().to_vec();
    let mut generated: Vec<u32> = Vec::new();
    let mut lp = LogitsProcessor::new(42, Some(0.7), Some(0.9));
    let model_max = match engine.name {
        "deepseek-r1-8b" => 1024,
        "deepseek-r1-32b" => 2048,
        "llama-3.3-70b" => 1024,
        _ => 2048,
    };
    // R1 models spend tokens on chain-of-thought before the visible answer.
    // Reserve extra budget so the think block never eats the entire quota and
    // leaves no room for </think> + the actual response.
    // R1-32B reasons more extensively than 8B — give it more overhead to avoid
    // the think block consuming all steps and producing an empty answer.
    let think_overhead = if is_think_primed {
        match engine.name {
            "deepseek-r1-32b" => 1536,
            _ => 512,
        }
    } else {
        0
    };
    let max_steps = max_new_tokens.saturating_add(think_overhead).min(model_max);

    match &mut engine.inner {
        ModelInner::Full { model, config, cache_dtype } => {
            let mut cache = Cache::new(true, *cache_dtype, config, &engine.device)
                .map_err(|e| anyhow!("create KV cache: {}", e))?;
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos, &mut cache)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::Quantized(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedSplit(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedQwen2(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedQwen2Split(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedQwen3(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedQwen3Split(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedQwen3MoeSplit(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
    }

    let text = engine.tokenizer.decode(&generated, true)
        .map_err(|e| anyhow!("decode: {}", e))?;
    // Truncate at the earliest stop string in case a control marker leaked into
    // the output (tokenizer that renders special tokens as plain text).
    let cut = engine.stop_strings.iter()
        .filter_map(|s| text.find(s))
        .min()
        .unwrap_or(text.len());
    let answer = text[..cut].trim();
    // For R1 models, strip the <think>…</think> reasoning preamble so only the
    // final answer is published. Empty string = think block was cut by max_tokens
    // (no final answer produced) — the caller should not upload this to IPFS.
    // R1 models are primed with an open <think>; Qwen3 (ChatML + /no_think) still
    // emits an empty <think></think> pair. Strip both so only the answer is published.
    Ok(if is_think_primed {
        strip_think_block(answer)
    } else if matches!(engine.name, "qwen3-32b" | "qwen3-235b") {
        strip_think_tags(answer)
    } else {
        answer.to_string()
    })
}

fn sample_next(logits: &Tensor, lp: &mut LogitsProcessor, context: &[u32]) -> Result<u32> {
    let dims = logits.dims();
    let last = match dims.len() {
        3 => logits.narrow(1, dims[1] - 1, 1)?.squeeze(1)?.squeeze(0)?,
        2 => logits.narrow(0, dims[0] - 1, 1)?.squeeze(0)?,
        1 => logits.clone(),
        _ => return Err(anyhow!("unexpected logits shape {:?}", dims)),
    };
    // Penalize recently-generated tokens to break degenerate repetition loops.
    let last = if REPEAT_PENALTY != 1.0 && !context.is_empty() {
        let start = context.len().saturating_sub(REPEAT_LAST_N);
        let f32_logits = last.to_dtype(DType::F32).map_err(|e| anyhow!("logits dtype: {}", e))?;
        candle_transformers::utils::apply_repeat_penalty(&f32_logits, REPEAT_PENALTY, &context[start..])
            .map_err(|e| anyhow!("repeat penalty: {}", e))?
    } else {
        last
    };
    lp.sample(&last).map_err(|e| anyhow!("sample: {}", e))
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Register the set of models this miner supports. Called once at startup.
pub fn init_supported(specs: &'static [&'static ModelSpec]) {
    let _ = SUPPORTED_SPECS.set(specs);
}

/// Outcome of the startup GPU inference probe.
pub enum GpuProbe {
    /// A GPU matmul succeeded — cuBLAS is loaded and full-speed inference is available.
    Ok,
    /// No CUDA device present — inference will fall back to CPU (acceptable for small models only).
    NoCuda,
    /// A CUDA device exists but cuBLAS could not be loaded — GPU inference is impossible.
    CublasMissing,
}

/// Verify that GPU inference actually works *before* mining starts.
///
/// `Device::new_cuda` succeeds with only the NVIDIA driver installed, but cudarc loads
/// cuBLAS lazily on the first GPU matmul and **panics** (it does not return an `Err`) when
/// `libcublas` cannot be `dlopen`'d. Discovering that mid-challenge poisons the engine and
/// spams the logs. So we force the failure here, once, with a tiny 2×2 matmul wrapped in
/// `catch_unwind`, and report a clean, actionable result.
pub fn probe_gpu_inference() -> GpuProbe {
    // candle's `Device::new_cuda` eagerly creates a cuBLAS handle, and cudarc *panics*
    // (it does not return an Err) when libcublas cannot be loaded. A genuinely absent
    // CUDA device, by contrast, returns Err cleanly. So the whole sequence — including
    // new_cuda — must live inside catch_unwind, and we distinguish the three outcomes:
    //   Ok(Ok)  -> CUDA + cuBLAS work
    //   Ok(Err) -> no usable CUDA device (clean error) -> CPU fallback
    //   Err     -> panic -> cuBLAS missing
    //
    // Silence the default panic hook for the probe so its scary backtrace doesn't pollute
    // the logs; we report a clean, actionable message ourselves from the caller.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let probe = std::panic::catch_unwind(|| {
        let device = Device::new_cuda(0)?;
        let a = Tensor::new(&[[1f32, 2.0], [3.0, 4.0]], &device)?;
        let b = Tensor::new(&[[5f32, 6.0], [7.0, 8.0]], &device)?;
        a.matmul(&b)?.to_vec2::<f32>()?;
        anyhow::Ok(())
    });
    std::panic::set_hook(prev_hook);
    match probe {
        Ok(Ok(())) => GpuProbe::Ok,
        Ok(Err(_)) => GpuProbe::NoCuda,
        Err(payload) => {
            // Surface the real panic message (e.g. which CUDA library failed to load) instead
            // of hiding it — candle creates cuBLAS, cuBLASLt and cuRAND handles at device init,
            // and any one of them missing panics here.
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            log::error!("GPU inference probe panicked: {}", msg);
            GpuProbe::CublasMissing
        }
    }
}

/// Pre-download all registered model files before mining starts.
///
/// Does not load weights into GPU memory — just ensures files are on disk so
/// the first inference request doesn't stall the mining workers mid-session.
/// Returns Err if any model fails to download; mining must not start in that case.
pub fn prefetch_models(specs: &'static [&'static ModelSpec]) -> Result<()> {
    for spec in specs {
        log::info!("SlmEngine: prefetching model '{}'…", spec.name);
        let result = match spec.format {
            ModelFormat::Safetensors => ensure_safetensors(spec).map(|_| ()),
            ModelFormat::Gguf | ModelFormat::GgufQwen2 | ModelFormat::GgufQwen3 | ModelFormat::GgufQwen3Moe => ensure_gguf(spec).map(|_| ()),
        };
        match result {
            Ok(()) => log::info!("SlmEngine: '{}' files ready.", spec.name),
            Err(e) => {
                log::error!("SlmEngine: prefetch '{}' failed: {} — cannot start mining.", spec.name, e);
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Return the model_ids of supported models that have fully-downloaded files (.ok flag present).
pub fn loaded_model_ids() -> Vec<[u8; 32]> {
    SUPPORTED_SPECS.get()
        .map(|specs| specs.iter()
            .filter(|s| model_dir(s).join(".ok").exists())
            .map(|s| s.model_id)
            .collect())
        .unwrap_or_default()
}

/// The model currently resident in the GPU engine (loaded in VRAM), if any.
/// Used to answer the synthetic-liveness task with the already-loaded model so the
/// task never forces an eviction/reload. Returns `None` when nothing is resident.
pub fn resident_model_id() -> Option<[u8; 32]> {
    ENGINE.lock().ok().and_then(|g| g.as_ref().map(|e| e.model_id))
}

/// True only when the model is supported and its files are completely downloaded.
pub fn is_model_ready(model_id: &[u8; 32]) -> bool {
    let Some(specs) = SUPPORTED_SPECS.get() else { return false; };
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else { return false; };
    model_dir(spec).join(".ok").exists()
}

/// Load the requested model on demand (evicting a cached different model if needed),
/// then run inference. Blocking — call from `spawn_blocking`.
pub fn load_and_run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
    let specs = SUPPORTED_SPECS.get()?;
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;

    // catch_unwind prevents any internal panic (cudarc, candle, OOM…) from permanently
    // poisoning ENGINE. Without this, one panic bricks inference for the entire session.
    let result = std::panic::catch_unwind(|| {
        let mut guard = match ENGINE.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                log::warn!("SlmEngine: ENGINE mutex was poisoned — recovering and evicting cached model");
                let mut g = poisoned.into_inner();
                *g = None;
                g
            }
        };

        let needs_load = guard.as_ref().map_or(true, |e| &e.model_id != model_id);
        if needs_load {
            if let Some(ref old) = *guard {
                log::info!("SlmEngine: evicting '{}' to load '{}'", old.name, spec.name);
            }
            *guard = None;
            let device = if cpu_inference_enabled() {
                log::info!("SlmEngine: --cpu-inference set — running on CPU device");
                Device::Cpu
            } else {
                match Device::new_cuda(0) {
                    Ok(d) => { log::info!("SlmEngine: CUDA device 0 active"); d }
                    Err(e) => { log::warn!("SlmEngine: CUDA unavailable ({e}) — CPU fallback"); Device::Cpu }
                }
            };
            match load_engine(spec, device) {
                Ok(e) => { *guard = Some(e); }
                Err(e) => {
                    log::error!("SlmEngine: failed to load '{}': {}", spec.name, e);
                    return None;
                }
            }
        }

        let engine = guard.as_mut()?;
        match generate(engine, prompt, max_tokens) {
            Ok(text) if !text.is_empty() => Some(text),
            Ok(_) => {
                log::warn!("SlmEngine '{}': think block cut by max_tokens, skipping response", engine.name);
                None
            }
            Err(e) => {
                log::warn!("SlmEngine '{}' generate error: {}", engine.name, e);
                Some(format!("[inference error: {}]", e))
            }
        }
    });

    match result {
        Ok(output) => output,
        Err(_) => {
            log::error!("SlmEngine: inference panicked — engine evicted, will retry on next challenge");
            log::error!("SlmEngine: cuBLAS missing? Run: sudo apt-get install -y libcublas-12-2 then restart the miner");
            if let Ok(mut g) = ENGINE.lock() { *g = None; }
            None
        }
    }
}
