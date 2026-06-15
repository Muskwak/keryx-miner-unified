//! Device-mapped fork of candle-transformers' `quantized_qwen2` for multi-GPU
//! VRAM pooling via layer-split (pipeline) inference.
//!
//! Same idea as `quantized_llama_split`: each transformer block is assigned to
//! one device of a caller-provided list, and the hidden state is moved across
//! devices at split boundaries during `forward`. The KV cache of each block
//! lives on that block's device, so cache memory is pooled too. Lets a mining
//! rig serve a 32B Qwen2-arch model (DeepSeek-R1-32B, ~20 GB) that no single
//! 8 GB card can hold.
//!
//! Differences from the llama split fork (Qwen2 architecture specifics):
//! - Attention q/k/v projections carry a learned **bias** (`attn_*.bias`),
//!   added after each projection.
//! - RoPE uses `rotary_emb::rope` (NOT the interleaved `rope_i` used by llama).
//! - Metadata keys are `qwen2.*`.
//!
//! GGUF only. MoE rejected (Keryx serves dense models).

use std::collections::HashMap;

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;

pub const MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    // Qwen2: q/k/v projections have a learned bias (dequantized to a plain tensor).
    attention_bq: Tensor,
    attention_bk: Tensor,
    attention_bv: Tensor,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp: Mlp,
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
        // Qwen2 uses the non-interleaved RoPE (rope), unlike llama (rope_i).
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;
        // Qwen2: add the learned q/k/v bias before reshaping into heads.
        let q = q.broadcast_add(&self.attention_bq)?;
        let k = k.broadcast_add(&self.attention_bk)?;
        let v = v.broadcast_add(&self.attention_bv)?;

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

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

#[derive(Debug, Clone)]
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
    /// Load a GGUF Qwen2-arch model with its transformer blocks split evenly
    /// across `devices`. The token embedding lives on the first device, the
    /// output norm/head on the last one.
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

        let n_expert = md_get("qwen2.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        if n_expert > 1 {
            candle_core::bail!("from_gguf: MoE models are not supported by the split loader");
        }
        let head_count = md_get("qwen2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("qwen2.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("qwen2.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("qwen2.embedding_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen2.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let head_dim = embedding_length / head_count;

        // RoPE tables and -inf constants are tiny; build one copy per device so
        // attention never has to fetch them across the PCIe bus.
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
            // Tied embeddings: re-read token_embd on the *last* device, where
            // the output head must live.
            Err(_) => ct.tensor(reader, "token_embd.weight", last_device)?,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            // Even split: block i goes to devices[i * n_dev / n_blocks].
            let device_idx = layer_idx * devices.len() / block_count;
            let device = &devices[device_idx];
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_bq = ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device)?;
            let attention_bk = ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device)?;
            let attention_bv = ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let feed_forward_w1 =
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_w2 =
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
            let feed_forward_w3 =
                ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let (cos, sin) = &cos_sin[device_idx];
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_bq: attention_bq.dequantize(device)?,
                attention_bk: attention_bk.dequantize(device)?,
                attention_bv: attention_bv.dequantize(device)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp: Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                },
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
        // One causal mask per device, built before the block loop because
        // `mask` needs `&mut self` while the loop borrows the layers mutably.
        let masks: Vec<Option<Tensor>> = if seq_len == 1 {
            vec![None; self.devices.len()]
        } else {
            (0..self.devices.len())
                .map(|i| self.mask(seq_len, i).map(Some))
                .collect::<Result<_>>()?
        };
        // Token IDs must sit where the embedding table lives.
        let x = if x.device().same_device(&self.devices[0]) {
            x.clone()
        } else {
            x.to_device(&self.devices[0])?
        };
        let mut layer_in = self.tok_embeddings.forward(&x)?;
        for layer in self.layers.iter_mut() {
            // Split boundary: ship the hidden state to the device holding this
            // block. A no-op within a device group.
            let x = if layer_in.device().same_device(&layer.device) {
                layer_in
            } else {
                layer_in.to_device(&layer.device)?
            };
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, masks[layer.device_idx].as_ref(), index_pos)?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            layer_in = (x + residual)?;
        }
        // The output norm/head live on the last device.
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
