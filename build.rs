use std::env;
use std::path::PathBuf;
use time::{format_description, OffsetDateTime};

/// Locate cl.exe for nvcc on Windows when it's not on PATH. Searches the standard VS 2022
/// Community/BuildTools install for the newest MSVC toolchain's Hostx64/x64 cl.exe. Returns
/// Err on non-Windows or if nothing is found (nvcc then falls back to PATH).
///
/// Ported from keryx-pascal-miner: the multi-arch PTX build needs a working host compiler on
/// Windows, and nvcc does not always find it itself.
fn discover_clbin() -> Result<String, std::env::VarError> {
    if env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("windows") {
        return Err(std::env::VarError::NotPresent);
    }
    let roots = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
    ];
    for root in roots {
        let tools = std::path::Path::new(root).join("VC\\Tools\\MSVC");
        let Ok(entries) = std::fs::read_dir(&tools) else { continue };
        // Newest toolchain version sorts last lexicographically.
        let mut versions: Vec<_> = entries.flatten().filter_map(|e| e.file_name().into_string().ok()).collect();
        versions.sort();
        if let Some(v) = versions.last() {
            let cl = tools.join(v).join("bin\\Hostx64\\x64\\cl.exe");
            if cl.exists() {
                return Ok(cl.to_string_lossy().into_owned());
            }
        }
    }
    Err(std::env::VarError::NotPresent)
}

/// Find the nvcc binary, honouring NVCC / CUDA_PATH env vars then PATH. Same resolution order as
/// keryx-pascal-miner's build.rs (which this multi-arch PTX logic is ported from).
fn find_nvcc() -> String {
    if let Ok(nvcc) = env::var("NVCC") {
        return nvcc;
    }
    if let Some(base) = env::var_os("CUDA_PATH") {
        let mut p = PathBuf::from(base);
        p.push("bin");
        p.push(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
        if p.exists() {
            return p.to_string_lossy().into_owned();
        }
    }
    "nvcc".to_string()
}

/// Blackwell (sm_100/sm_120) needs nvcc 12.8+ — older toolkits (e.g. a CUDA 12.2 container built
/// before Blackwell existed) hard-error with "nvcc fatal: Value 'sm_100' is not defined for
/// option 'gpu-architecture'" and abort the WHOLE build, including every older arch that toolkit
/// CAN compile. Detected once via `nvcc --version` so older toolchains still produce a working
/// (Pascal-through-Hopper) binary instead of failing outright.
fn nvcc_supports_blackwell(nvcc: &str) -> bool {
    let Ok(out) = std::process::Command::new(nvcc).arg("--version").output() else { return false };
    let text = String::from_utf8_lossy(&out.stdout);
    // Line of interest looks like "Cuda compilation tools, release 12.8, V12.8.61".
    text.split("release ")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .and_then(|v| {
            let mut parts = v.trim().split('.');
            let major: u32 = parts.next()?.parse().ok()?;
            let minor: u32 = parts.next()?.parse().ok()?;
            Some((major, minor) >= (12, 8))
        })
        .unwrap_or(false)
}

/// Compile the PoM CUDA kernel to per-arch PTX for every major NVIDIA compute capability
/// (Pascal 61 → Hopper 90) and emit an `$OUT_DIR/pom_ptx.rs` with a `get_pom_ptx((maj,min))`
/// selector. The miner picks the matching PTX at runtime by querying each device's compute
/// capability — so a P40 (sm_61) keeps the hand-tuned Pascal kernel while an Ada/Hopper card
/// gets its native PTX. Ported verbatim from keryx-pascal-miner (the 9 MH/s-on-P40 path).
///
/// Runs on every non-Apple/non-Android target — the same condition `src/lib.rs` uses to compile
/// in `src/pom_gpu.rs` (which `include!`s the `pom_ptx.rs` this emits). CUDA is not gated behind
/// the `cuda` Cargo feature: no `#[cfg(feature = "cuda")]` exists in src/, matching every source
/// fork where CUDA was unconditional on desktop.
fn build_cuda_ptx(out_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=cuda/pom_mine.cu");
    let nvcc = find_nvcc();

    // Compile PTX for all major NVIDIA compute capabilities: Pascal (61), Volta (70), Turing (75),
    // Ampere (80/86), Ada (89), Hopper (90), Blackwell (100 datacenter B100/B200, 120 consumer
    // RTX 50-series). Every arch is compiled from the SAME tuned cuda/pom_mine.cu, so sm_61
    // (P40/1070) keeps its full tuning while newer GPUs get native PTX. Blackwell needs nvcc
    // 12.8+ (the first CUDA 12.x release recognizing sm_100/sm_120) — no separate toolchain or
    // prebuilt fatbin required, unlike forks that pin an older nvcc and paper over the gap with a
    // manually-rebuilt binary blob.
    let all_archs = [
        ("61", "PomMinerSm61"), ("70", "PomMinerSm70"), ("75", "PomMinerSm75"),
        ("80", "PomMinerSm80"), ("86", "PomMinerSm86"), ("89", "PomMinerSm89"),
        ("90", "PomMinerSm90"), ("100", "PomMinerSm100"), ("120", "PomMinerSm120"),
    ];
    let has_blackwell = nvcc_supports_blackwell(&nvcc);
    let archs: Vec<_> = all_archs.into_iter().filter(|(arch, _)| has_blackwell || (*arch != "100" && *arch != "120")).collect();

    for (arch, _name) in &archs {
        let ptx_path = format!("{out_dir}/pom_mine_sm{arch}.ptx");
        let mut cmd = std::process::Command::new(&nvcc);
        cmd.args(["-ptx", "-O3", "--use_fast_math",
                  &format!("-arch=sm_{arch}"), "cuda/pom_mine.cu", "-o", &ptx_path]);
        if let Ok(c) = env::var("MSVC_CLBIN").or_else(|_| discover_clbin()) {
            cmd.arg("-ccbin").arg(&c);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        let out = cmd.output().unwrap_or_else(|e| panic!("nvcc ({nvcc}) failed to run: {e}"));
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            panic!(
                "nvcc failed to compile cuda/pom_mine.cu (sm_{arch}):\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
            );
        }
    }

    // Generate the per-arch selector module included by src/pom_gpu.rs. Compute capability
    // major.minor is the arch string split before its LAST digit (sm_61 -> 6.1, sm_100 -> 10.0,
    // sm_120 -> 12.0) — NOT simply its first two characters, which breaks for 3-digit archs like
    // Blackwell's 100/120 (would misparse sm_100 as compute 1.0).
    let mut match_arms = String::new();
    for (arch, _name) in &archs {
        let split_at = arch.len() - 1;
        let (major, minor) = (&arch[..split_at], &arch[split_at..]);
        match_arms.push_str(&format!(
            "        ({}, {}) => include_str!(concat!(env!(\"OUT_DIR\"), \"/pom_mine_sm{}.ptx\")),\n",
            major, minor, arch
        ));
    }
    let lookup_code = format!(
        "/// Auto-selected PTX based on GPU compute capability\npub fn get_pom_ptx(cc: (u32, u32)) -> &'static str {{\n    match cc {{\n{}        _ => include_str!(concat!(env!(\"OUT_DIR\"), \"/pom_mine_sm61.ptx\")),\n    }}\n}}",
        match_arms
    );
    std::fs::write(format!("{out_dir}/pom_ptx.rs"), lookup_code)
        .unwrap_or_else(|e| panic!("Failed to write pom_ptx.rs: {e}"));
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let format = format_description::parse_borrowed::<2>("[year repr:last_two][month][day][hour][minute]")?;
    let dt = OffsetDateTime::now_utc().format(&format)?;
    println!("cargo:rustc-env=PACKAGE_COMPILE_TIME={}", dt);

    println!("cargo:rerun-if-changed=proto");
    println!("cargo:rerun-if-changed=src/keccakf1600_x86-64.s");
    println!("cargo:rerun-if-changed=metal/pom_mine.metal");
    println!("cargo:rerun-if-changed=build.rs");
    tonic_build::configure()
        .build_server(false)
        .compile(
            &["proto/rpc.proto", "proto/p2p.proto", "proto/messages.proto"],
            &["proto"],
        )?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    // ── Backend build steps ──────────────────────────────────────────────────
    //
    // Backends are additive per OS: Windows/Linux compile `cuda` + `vulkan`
    // together (one binary covers an NVIDIA + AMD rig), macOS/iOS compile `metal`, Android
    // compiles `vulkan`.
    //
    // CUDA PTX build — same target_os condition as `src/lib.rs`'s `pom_gpu` module cfg (NOT the
    // `cuda` Cargo feature: nothing in src/ checks `cfg(feature = "cuda")`, and `pom_gpu.rs` is
    // compiled unconditionally on desktop, so gating the PTX build behind the feature left a
    // plain `cargo build` unable to satisfy `pom_gpu.rs`'s `include!(.../pom_ptx.rs)`). The tuned
    // kernel + multi-arch selector come from keryx-pascal-miner; the metal fork's old single-arch
    // PTX build is superseded.
    let vulkan_enabled = env::var_os("CARGO_FEATURE_VULKAN").is_some();

    if target_os != "macos" && target_os != "ios" && target_os != "android" {
        let out_dir = env::var("OUT_DIR")?;
        build_cuda_ptx(&out_dir)?;
    }

    // Vulkan shader build is owned by the `keryx-vulkan` workspace crate's own build.rs (it
    // invokes glslc on shaders/*.comp). We only need to surface the feature so the crate is
    // compiled in at all (Cargo.toml makes `keryx-vulkan` an optional dependency activated by
    // `vulkan`). Nothing to do here beyond the feature existing — recorded for clarity.
    if vulkan_enabled {
        // The keryx-vulkan crate (workspace member) compiles its own shaders. This branch is a
        // no-op at the top-level build.rs; kept as a documented hook in case a future shader the
        // main crate needs directly is added.
    }

    // Keccak-f1600 assembly: x86_64 only. On ARM64 (Apple Silicon, macOS + iOS) the Rust
    // keccak crate is used instead (activated as a dependency in Cargo.toml for non-x86_64).
    if target_arch == "x86_64" && target_os != "windows" && target_os != "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64.s").compile("libkeccak.a");
    }
    if target_arch == "x86_64" && target_os == "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64-osx.s").compile("libkeccak.a");
    }
    Ok(())
}
