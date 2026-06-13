/// Registry of supported inference models.
///
/// model_id = sha2-256(primary_weight_file) = CIDv0_bytes[2..34].
/// Verifiable: decode the weight CID from base58btc, skip the 2-byte multihash prefix.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Full-precision safetensors (one or more shards).
    Safetensors,
    /// GGUF quantized — LLaMA/LLaMA3 architecture.
    Gguf,
    /// GGUF quantized — Qwen2 architecture (DeepSeek-R1-32B distill).
    GgufQwen2,
    /// GGUF quantized — Qwen3 architecture (Qwen3-32B, 5090 tier).
    GgufQwen3,
    /// GGUF quantized — Qwen3-MoE architecture (Qwen3-235B-A22B, multi-GPU tier).
    /// Served via the layer-split loader (`quantized_qwen3_moe_split`); fused
    /// stacked experts pooled across the device list.
    GgufQwen3Moe,
}

#[derive(Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    /// 32-byte on-chain identifier embedded in AiRequest payloads.
    pub model_id: [u8; 32],
    pub format: ModelFormat,
    pub tokenizer_cid: &'static str,
    /// Unused for GGUF (architecture embedded in file).
    pub config_cid: &'static str,
    /// Safetensors: one entry per shard. GGUF: single entry.
    pub weight_cids: &'static [&'static str],
    /// Local directory name under `<exe_dir>/models/`.
    pub dir_name: &'static str,
    /// Minimum VRAM (MB) required to actually serve this model: weights +
    /// KV cache + CUDA workspace. Used by the OPoI capability gate so `ai:cap`
    /// never announces a model the miner cannot load. 0 = never gated.
    pub min_vram_mb: u64,
}

pub const TINYLLAMA: ModelSpec = ModelSpec {
    name: "tinyllama",
    // sha2-256(QmdqcmS8aMngiZWYYdeZEaW22N6XRTd9zK5ZCJG1MPmrQ3)
    model_id: [
        0xe6, 0x4a, 0xf3, 0x68, 0xec, 0x93, 0x51, 0xa5,
        0xa4, 0xc0, 0xec, 0x7a, 0xe4, 0x7d, 0x42, 0xad,
        0xa7, 0xf6, 0xb3, 0xf1, 0xa6, 0xe6, 0x0f, 0xc7,
        0x3d, 0x0e, 0xb6, 0xca, 0x29, 0x53, 0x64, 0x5c,
    ],
    format: ModelFormat::Safetensors,
    tokenizer_cid: "QmSKrRu8HRt9v2dUeVdABKDkuREa5xFhPLZdevvvBfDYmp",
    config_cid: "QmbLTR3GLjBUKw8Lj14isiwG3XZJaL61ES852vkNqNPhyd",
    weight_cids: &["QmdqcmS8aMngiZWYYdeZEaW22N6XRTd9zK5ZCJG1MPmrQ3"],
    dir_name: "TinyLlama-1.1B",
    // Baseline model — never gated. GPUs too small for it must use --cpu-inference.
    min_vram_mb: 0,
};

pub const DEEPSEEK_R1_8B: ModelSpec = ModelSpec {
    name: "deepseek-r1-8b",
    // sha2-256(QmYK1faUGNMYZ2UKeSpUoUoFpRarZQEwfPCHbYNG2ib2mR)
    model_id: [
        0x94, 0x29, 0x67, 0x33, 0x16, 0xbc, 0x40, 0xec,
        0x06, 0x67, 0x89, 0x45, 0x34, 0x57, 0x8b, 0x41,
        0x23, 0x6f, 0xc7, 0xee, 0xa4, 0xd9, 0x31, 0xf1,
        0x48, 0x9c, 0x34, 0xc5, 0x83, 0x7f, 0x42, 0xf4,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmXVdcr2FJuHtXcBbYbBuCMic2pJTkM1LJ6WpyfvhDytHg",
    config_cid: "",
    weight_cids: &["QmYK1faUGNMYZ2UKeSpUoUoFpRarZQEwfPCHbYNG2ib2mR"],
    dir_name: "DeepSeek-R1-8B",
    // ~4.9 GB Q4 weights + KV cache.
    min_vram_mb: 5_500,
};

pub const DEEPSEEK_R1_32B: ModelSpec = ModelSpec {
    name: "deepseek-r1-32b",
    // sha2-256(model.gguf)
    model_id: [
        0xbe, 0xd9, 0xb0, 0xf5, 0x51, 0xf5, 0xb9, 0x5b,
        0xf9, 0xda, 0x58, 0x88, 0xa4, 0x8f, 0x0f, 0x87,
        0xc3, 0x7a, 0xd6, 0xb7, 0x25, 0x19, 0xc4, 0xcb,
        0xd7, 0x75, 0xf5, 0x4a, 0xc0, 0xb9, 0xfc, 0x62,
    ],
    format: ModelFormat::GgufQwen2,
    tokenizer_cid: "Qmf3uZwnuxZUhDbhup8Q51soVMRmNxohYctG9wZemNEPHm",
    config_cid: "",
    weight_cids: &["QmSrmkEoJUPf7r9t4o79F5APycnGrRu2icaU3KKPdFVUk7"],
    dir_name: "DeepSeek-R1-32B",
    // ~19 GB Q4 weights + KV cache.
    min_vram_mb: 20_000,
};

pub const LLAMA_3_3_70B: ModelSpec = ModelSpec {
    name: "llama-3.3-70b",
    // CIDv0[2..34] of model.gguf — Q4_K_M (candle 0.8.4 cannot read the old IQ3 quant)
    model_id: [
        0xed, 0xf4, 0x76, 0xbd, 0x67, 0xa2, 0xf7, 0xb1,
        0x9b, 0x40, 0xa1, 0x7d, 0xef, 0x4c, 0xaa, 0x3c,
        0x84, 0x7b, 0x68, 0xfd, 0xa1, 0x8a, 0x3c, 0x31,
        0x29, 0x35, 0xb0, 0xb3, 0x43, 0xae, 0xb3, 0x3e,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmPd7WQvoQupfzpPVnVVc1Zra5SH4jKnGqNrdTHFtdQuvd",
    config_cid: "",
    weight_cids: &["QmeMXYDJMu916BSfunHEHWkz8Uc4FwTic5phtQXoJs3p5j"],
    dir_name: "Llama-3.3-70B",
    // ~42.5 GB Q4_K_M weights + KV cache (matches the --very-high startup gate).
    min_vram_mb: 46_000,
};

pub const QWEN3_32B: ModelSpec = ModelSpec {
    name: "qwen3-32b",
    // CIDv0[2..34] of model.gguf — Q6_K (bartowski, arch qwen3, candle-readable)
    model_id: [
        0xf0, 0x7e, 0x57, 0xb1, 0x1e, 0xd6, 0xcc, 0xe5,
        0x63, 0x31, 0xae, 0xff, 0x60, 0xcf, 0xdb, 0x36,
        0x24, 0xbd, 0x97, 0xe7, 0x03, 0x78, 0x8c, 0xba,
        0x02, 0xce, 0x00, 0xfa, 0xe7, 0x9a, 0xb0, 0x43,
    ],
    format: ModelFormat::GgufQwen3,
    tokenizer_cid: "QmcuGkJvR343ry3b4jy7u5L9ior3ujas3yGAFMSyZdACb5",
    config_cid: "",
    weight_cids: &["QmeXSJ4bYmtnk9wVKUgb5S9GWTRqz15Tgya7gyqDbqv8Wn"],
    dir_name: "Qwen3-32B",
    // ~27 GB Q6_K weights + KV cache. Gated above 24 GB so it lands as the
    // 5090-only tier (32 GB): excludes 24 GB cards, fits with KV headroom.
    min_vram_mb: 30_000,
};

pub const QWEN3_235B: ModelSpec = ModelSpec {
    name: "qwen3-235b",
    // TODO(release): set to CIDv0[2..34] of the pinned Q4_K_M model.gguf.
    // Qwen3-235B-A22B (arch qwen3moe, Apache-2.0). PLACEHOLDER — must be the real
    // weight CID before any build/announce, or the OPoI capability gate will
    // advertise a model that cannot be fetched.
    model_id: [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
    format: ModelFormat::GgufQwen3Moe,
    tokenizer_cid: "TODO_PIN_QWEN3_235B_TOKENIZER_CID",
    config_cid: "",
    weight_cids: &["TODO_PIN_QWEN3_235B_Q4KM_GGUF_CID"],
    dir_name: "Qwen3-235B",
    // ~135 GB Q4_K_M weights + KV cache. Gated so it only lands on rigs that can
    // pool enough VRAM (e.g. 6×5090 = 192 GB). --very-ultra tier.
    min_vram_mb: 140_000,
};

pub const REGISTRY: &[&ModelSpec] = &[
    &TINYLLAMA,
    &DEEPSEEK_R1_8B,
    &DEEPSEEK_R1_32B,
    &QWEN3_32B,
    &QWEN3_235B,
    &LLAMA_3_3_70B,
];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.name == name)
}

pub fn available_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|m| m.name).collect()
}
