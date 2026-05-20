/// Registry of supported inference models.
///
/// Each entry maps a CLI name to its on-chain model_id and IPFS CIDs.
/// model_id = blake2b-256(model_name) — used to match AiRequest payloads.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Full-precision safetensors (one or more shards).
    Safetensors,
    /// GGUF quantized (single file, tokenizer loaded separately).
    Gguf,
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
}

pub const TINYLLAMA: ModelSpec = ModelSpec {
    name: "tinyllama",
    model_id: [
        0x7b, 0x24, 0x0e, 0xd7, 0x6b, 0xa8, 0x46, 0x6c,
        0xf9, 0x7d, 0xeb, 0x9b, 0xe6, 0x35, 0x81, 0x2d,
        0x12, 0xd0, 0x49, 0x96, 0xcb, 0x56, 0x12, 0xcb,
        0xf9, 0x9d, 0x24, 0x4d, 0x6d, 0x55, 0x1e, 0x9f,
    ],
    format: ModelFormat::Safetensors,
    tokenizer_cid: "QmSKrRu8HRt9v2dUeVdABKDkuREa5xFhPLZdevvvBfDYmp",
    config_cid: "QmbLTR3GLjBUKw8Lj14isiwG3XZJaL61ES852vkNqNPhyd",
    weight_cids: &["QmdqcmS8aMngiZWYYdeZEaW22N6XRTd9zK5ZCJG1MPmrQ3"],
    dir_name: "TinyLlama-1.1B",
};

pub const DEEPSEEK_R1_8B: ModelSpec = ModelSpec {
    name: "deepseek-r1-8b",
    model_id: [
        0x9b, 0x59, 0xa0, 0xa3, 0x73, 0x3c, 0xe8, 0xcf,
        0x52, 0x0c, 0xa2, 0xa3, 0x55, 0x6a, 0x8b, 0x0c,
        0xa3, 0x21, 0x64, 0xbd, 0x7f, 0xd3, 0x0b, 0xa3,
        0x2d, 0x8d, 0xdf, 0x89, 0x68, 0x50, 0xf0, 0xb8,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmXVdcr2FJuHtXcBbYbBuCMic2pJTkM1LJ6WpyfvhDytHg",
    config_cid: "",
    weight_cids: &["QmYK1faUGNMYZ2UKeSpUoUoFpRarZQEwfPCHbYNG2ib2mR"],
    dir_name: "DeepSeek-R1-8B",
};

pub const REGISTRY: &[&ModelSpec] = &[&TINYLLAMA, &DEEPSEEK_R1_8B];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.name == name)
}

pub fn available_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|m| m.name).collect()
}
