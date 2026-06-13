//! Device-mapped fork of candle-transformers' `quantized_qwen3_moe` for
//! multi-GPU VRAM pooling via layer-split (pipeline) inference.
//!
//! Same split idea as the dense llama/qwen splits (each transformer block on one
//! device of a caller-provided list, hidden state moved across devices at split
//! boundaries, per-block KV cache), but for the **MoE** Qwen3 architecture
//! (e.g. Qwen3-235B-A22B). Lets a rig pool its VRAM to serve a 235B model.
//!
//! The hard part — reading the *stacked* quantized experts
//! (`ffn_{gate,up,down}_exps.weight`) and the fused top-k expert GEMM — is
//! reused as-is from candle's `fused_moe::FusedMoeGGUF` (all-public). We only
//! place each MoE block's experts/router on that block's device and run the
//! standard split forward around it.
//!
//! Qwen3 architecture specifics:
//! - NO q/k/v bias; per-head q_norm/k_norm RMSNorm on Q/K before RoPE.
//! - `head_dim` read explicitly from `{arch}.attention.key_length`.
//! - Non-interleaved RoPE.
//! - Metadata keys use the GGUF `general.architecture` value (e.g. `qwen3moe`),
//!   read generically rather than hard-coded.
//! - Each block's feed-forward is a fused MoE (router + 128 stacked experts,
//!   top-k) instead of a dense MLP. (Qwen3-MoE has no shared expert.)
//!
//! GGUF only.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Activation, Embedding, Linear, Module};
use candle_transformers::fused_moe::FusedMoeGGUF;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;

pub const MAX_SEQ_LEN: usize = 4096;

/// Dense feed-forward (used only if a layer is not MoE; Qwen3-235B is all-MoE).
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

/// A block's feed-forward: dense MLP or fused MoE. `FusedMoeGGUF` is neither
/// `Clone` nor `Debug`, so this enum (and everything holding it) derives nothing.
enum Ffn {
    Dense(Mlp),
    Moe(FusedMoeGGUF),
}

impl Ffn {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Ffn::Dense(m) => m.forward(xs),
            Ffn::Moe(m) => m.forward(xs, is_prefill),
        }
    }
}

struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    // Qwen3: per-head RMSNorm on Q and K (no q/k/v bias).
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    attention_norm: RmsNorm,
    ffn: Ffn,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    /// Device this block's weights live on; the hidden state is moved here
    /// before the block runs.
    device: Device,
    /// Index of `device` in the model's device list (Device is not hashable).
    device_idx: usize,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        // Qwen3 uses non-interleaved RoPE.
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Qwen3: per-head RMSNorm on Q and K (flatten heads, norm over head_dim).
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
        let k = k_flat.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Grouped-query attention: repeat the KV heads to match the Q heads.
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;

        // Output is num_head * head_dim wide (may differ from embedding_length).
        let y = y
            .transpose(1, 2)?
            .reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    /// Causal masks cached per (seq_len, device index).
    masks: HashMap<(usize, usize), Tensor>,
    devices: Vec<Device>,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    /// Load a GGUF Qwen3-MoE-arch model with its transformer blocks split evenly
    /// across `devices`. The token embedding lives on the first device, the
    /// output norm/head on the last one. Each MoE block's router + stacked
    /// experts live on that block's device, so expert VRAM is pooled too.
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        devices: &[Device],
    ) -> Result<Self> {
        if devices.is_empty() {
            candle_core::bail!("from_gguf: device list must not be empty");
        }
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Metadata keys are namespaced by the GGUF architecture (e.g. `qwen3moe`).
        let arch = md_get("general.architecture")?.to_string()?.to_string();
        let mdk = |suffix: &str| format!("{arch}.{suffix}");

        let head_count = md_get(&mdk("attention.head_count"))?.to_u32()? as usize;
        let head_count_kv = md_get(&mdk("attention.head_count_kv"))?.to_u32()? as usize;
        let block_count = md_get(&mdk("block_count"))?.to_u32()? as usize;
        let embedding_length = md_get(&mdk("embedding_length"))?.to_u32()? as usize;
        // Qwen3 carries head_dim explicitly; fall back to embedding/head_count.
        let head_dim = match md_get(&mdk("attention.key_length")) {
            Ok(v) => v.to_u32()? as usize,
            Err(_) => embedding_length / head_count,
        };
        let rms_norm_eps = md_get(&mdk("attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;
        let rope_freq_base = md_get(&mdk("rope.freq_base"))
            .and_then(|m| m.to_f32())
            .unwrap_or(1_000_000f32);

        // MoE parameters.
        let n_expert = md_get(&mdk("expert_count")).and_then(|v| v.to_u32()).unwrap_or(0) as usize;
        let n_expert_used = md_get(&mdk("expert_used_count"))
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        // A shared-expert FF length (>0) means the topk probs are NOT renormalised
        // (the shared expert carries the rest). Qwen3-MoE has no shared expert, so
        // this key is absent → renormalise. Mirrors candle's quantized_qwen3_moe.
        let shared_ff = md_get(&mdk("expert_shared_feed_forward_length"))
            .and_then(|v| v.to_u32())
            .unwrap_or(0);
        let norm_topk_prob = shared_ff == 0;
        // MoE compute dtype: F32 to match the rest of this split's F32 pipeline
        // (deterministic across GPU archs — required for OPoI reproducibility).
        let moe_dtype = DType::F32;

        // RoPE tables and -inf constants are tiny; build one copy per device.
        let mut cos_sin = Vec::with_capacity(devices.len());
        let mut neg_infs = Vec::with_capacity(devices.len());
        for device in devices {
            cos_sin.push(precomput_freqs_cis(head_dim, rope_freq_base, device)?);
            neg_infs.push(Tensor::new(f32::NEG_INFINITY, device)?);
        }

        let first_device = &devices[0];
        let last_device = devices.last().unwrap();
        let tok_embeddings = ct
            .tensor(reader, "token_embd.weight", first_device)?
            .dequantize(first_device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", last_device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", last_device) {
            Ok(tensor) => tensor,
            // Tied embeddings: re-read token_embd on the *last* device.
            Err(_) => ct.tensor(reader, "token_embd.weight", last_device)?,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let device_idx = layer_idx * devices.len() / block_count;
            let device = &devices[device_idx];
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            // Feed-forward: fused MoE if the model has experts, else dense MLP.
            // (Qwen3-MoE has experts on every layer.)
            let ffn = if n_expert > 0 {
                // Router: ffn_gate_inp dequantized to F32 (per candle's qwen3_moe).
                let gate_ws = ct
                    .tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?
                    .dequantize(device)?
                    .to_dtype(DType::F32)?;
                let gate = Linear::new(gate_ws, None);
                // Stacked quantized experts [n_expert, ...] kept on this device.
                let gate_experts =
                    Arc::new(ct.tensor(reader, &format!("{prefix}.ffn_gate_exps.weight"), device)?);
                let up_experts =
                    Arc::new(ct.tensor(reader, &format!("{prefix}.ffn_up_exps.weight"), device)?);
                let down_experts =
                    Arc::new(ct.tensor(reader, &format!("{prefix}.ffn_down_exps.weight"), device)?);
                Ffn::Moe(FusedMoeGGUF {
                    gate,
                    gate_experts,
                    up_experts,
                    down_experts,
                    act: Activation::Silu,
                    norm_topk_prob,
                    num_experts_per_tok: n_expert_used,
                    dtype: moe_dtype,
                })
            } else {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                Ffn::Dense(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            };

            let (cos, sin) = &cos_sin[device_idx];
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                q_norm,
                k_norm,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                ffn,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_infs[device_idx].clone(),
                kv_cache: None,
                device: device.clone(),
                device_idx,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            devices: devices.to_vec(),
        })
    }

    fn mask(&mut self, t: usize, device_idx: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&(t, device_idx)) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.devices[device_idx])?;
            self.masks.insert((t, device_idx), mask.clone());
            Ok(mask)
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        // Prompt processing (seq_len > 1) vs single-token decode. FusedMoeGGUF
        // takes this hint; in candle 0.9 both paths are correctness-equivalent.
        let is_prefill = seq_len > 1;
        let masks: Vec<Option<Tensor>> = if seq_len == 1 {
            vec![None; self.devices.len()]
        } else {
            (0..self.devices.len())
                .map(|i| self.mask(seq_len, i).map(Some))
                .collect::<Result<_>>()?
        };
        let x = if x.device().same_device(&self.devices[0]) {
            x.clone()
        } else {
            x.to_device(&self.devices[0])?
        };
        let mut layer_in = self.tok_embeddings.forward(&x)?;
        for layer in self.layers.iter_mut() {
            let x = if layer_in.device().same_device(&layer.device) {
                layer_in
            } else {
                layer_in.to_device(&layer.device)?
            };
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, masks[layer.device_idx].as_ref(), index_pos)?;
            let x = (attn + residual)?;

            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.ffn.forward(&x, is_prefill)?;
            layer_in = (x + residual)?;
        }
        let last_device = self.devices.last().unwrap();
        let layer_in = if layer_in.device().same_device(last_device) {
            layer_in
        } else {
            layer_in.to_device(last_device)?
        };
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        self.output.forward(&x)
    }
}
