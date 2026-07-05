//! PoM-walk kernel micro-benchmark. Measures the walk in isolation on a large VRAM blob and
//! compares byte-identical kernel variants (baseline / wave32 / magic-modulo). See `bench.rs`.
//!
//!   cargo run -p keryx-vulkan --example bench_pom_walk --features bench --release
//!
//! Env knobs: POM_BENCH_BLOB_MB (default 1024), POM_BENCH_NONCES (default 16777216),
//! POM_BENCH_ITERS (default 5).

fn main() {
    keryx_vulkan::bench::run();
}
