use std::env;
use std::path::PathBuf;

/// Every NVIDIA compute capability the miner's own PoM kernel targets (see the top-level
/// `build.rs`), from Pascal through Blackwell. The inference kernels here used to compile for
/// only ONE arch (whatever `bindgen_cuda` autodetected via `nvidia-smi`, or `CUDA_COMPUTE_CAP`) —
/// PTX JITs forward onto newer cards but NOT backward onto older ones, so a build done on/for an
/// Ampere+ card silently couldn't run OPoI inference on Pascal/Volta/Turing hardware even though
/// PoM mining itself (properly multi-arch) worked fine there. Fixed by building a fatbin per
/// kernel file instead: one file embedding real SASS for every listed arch plus a PTX-only entry
/// for the newest, so `cuModuleLoadData` (which accepts a fatbin exactly like it accepts raw PTX
/// text — see `candle-core`'s `get_or_load_func`) picks the right native code at runtime with no
/// per-arch dispatch logic needed on the Rust side.
const KERNEL_ARCHS: &[&str] = &["61", "70", "75", "80", "86", "89", "90", "100", "120"];

/// The 11 kernel source files `candle-kernels` compiles to PTX (matches the `Id` enum in
/// `src/lib.rs` 1:1 by uppercased file stem: `affine.cu` -> `AFFINE`, etc.). Hardcoded rather than
/// globbed (as `bindgen_cuda::Builder::default()`'s auto-discovery did) so the `moe/` subdirectory
/// — built separately below as a static lib, not PTX — is never accidentally swept in.
const KERNEL_FILES: &[&str] = &[
    "affine", "binary", "cast", "conv", "fill", "indexing", "quantized", "reduce", "sort",
    "ternary", "unary",
];

/// Blackwell (sm_100/sm_120) needs nvcc 12.8+ — an older toolkit (e.g. a CUDA 12.2 build
/// container predating Blackwell) hard-errors on `-gencode=arch=compute_100,...` with
/// "nvcc fatal: Value 'sm_100' is not defined", aborting the fatbin for EVERY arch in the same
/// invocation, not just the unsupported one. Detected once via `nvcc --version` so older
/// toolchains still produce a working (Pascal-through-Hopper) fatbin instead of failing outright.
fn nvcc_supports_blackwell(nvcc: &str) -> bool {
    let Ok(out) = std::process::Command::new(nvcc).arg("--version").output() else { return false };
    let text = String::from_utf8_lossy(&out.stdout);
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

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build one multi-arch fatbin per kernel file (see KERNEL_ARCHS doc above).
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let has_blackwell = nvcc_supports_blackwell(&nvcc);
    let kernel_archs: Vec<&str> =
        KERNEL_ARCHS.iter().copied().filter(|a| has_blackwell || (*a != "100" && *a != "120")).collect();
    let highest_arch = kernel_archs.last().expect("kernel_archs is non-empty");

    let mut lookup_code = String::new();
    for name in KERNEL_FILES {
        let src = format!("src/{name}.cu");
        println!("cargo:rerun-if-changed={src}");
        let fatbin_path = out_dir.join(format!("{name}.fatbin"));

        let mut cmd = std::process::Command::new(&nvcc);
        cmd.args(["-fatbin", "-O3", "--expt-relaxed-constexpr", "-std=c++17"]);
        for arch in &kernel_archs {
            cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        }
        // PTX-only entry for the newest listed arch, so the fatbin JITs forward onto any future
        // GPU generation released after this binary, exactly like a single-arch PTX build always
        // could — the fatbin approach only ADDS native SASS for the older/current archs, it never
        // narrows forward compatibility.
        cmd.arg(format!("-gencode=arch=compute_{highest_arch},code=compute_{highest_arch}"));
        cmd.args(["-o", fatbin_path.to_str().expect("valid OUT_DIR path"), &src]);

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        let out = cmd.output().unwrap_or_else(|e| panic!("nvcc ({nvcc}) failed to run: {e}"));
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            panic!("nvcc failed to build fatbin for {src}:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
        }

        lookup_code.push_str(&format!(
            "pub const {}: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{}.fatbin\"));\n",
            name.to_uppercase(),
            name
        ));
    }
    std::fs::write(&ptx_path, lookup_code).unwrap_or_else(|e| panic!("Failed to write ptx.rs: {e}"));

    let mut moe_builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Build for FFI binding (must use custom bindgen_cuda, which supports simutanously build PTX and lib)
    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
            // nvcc's default host-compiler CRT on MSVC is the static runtime (/MT). Everything
            // else in this workspace links the dynamic runtime (/MD) -- Rust's own MSVC target
            // default, and CMake's default for llama-cpp-sys-2's Release build. Mixing the two
            // in one binary trips MSVC's linker (LNK2038 "RuntimeLibrary mismatch") once both the
            // `cuda` and `vulkan` features are built together, since libmoe.a and
            // llama-cpp-sys-2's static libs both end up in the same link. Force /MD to match.
            moe_builder = moe_builder.arg("-Xcompiler").arg("/MD");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    let moe_builder = moe_builder.kernel_paths(vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
    ]);
    moe_builder.build_lib(out_dir.join("libmoe.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
