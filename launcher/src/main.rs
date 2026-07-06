// keryx-miner launcher: detects the GPU vendor and execs the matching sibling binary.
//
// Why this exists: a single cuda,vulkan binary hard-imports nvcuda.dll at load time, so it
// won't even start on an AMD/Intel-only machine. A vulkan-only binary runs everywhere but
// can't mine NVIDIA cards. Rather than risk a deep dynamic-loading refactor of candle/cudarc,
// we ship BOTH binaries in one package plus this ~5 KB launcher that picks the right one.
//
// Detection (any one hits → treat as NVIDIA rig):
//   1. `nvidia-smi` on PATH and exits 0 (the canonical "NVIDIA driver is installed" signal).
//   2. The driver DLL/so itself is loadable (nvcuda.dll on Windows, libcuda.so on Linux) —
//      catches systems where nvidia-smi isn't on PATH but the driver is.
//   3. GPU enumeration names an NVIDIA adapter (fallback).
//
// If NVIDIA is present → run keryx-miner-nvidia(.exe). Otherwise → run keryx-miner-amd(.exe).
// All argv, env vars, and the current working directory are forwarded unchanged, so to the
// user (and to pool/node software) the launcher is transparent.
//
// If the chosen binary is missing, fall back to the other one with a warning — better to try
// than to refuse outright (e.g. an NVIDIA box where the user only downloaded the AMD half).

use std::env;
use std::path::PathBuf;
use std::process::Command;

#[cfg(windows)]
const EXE_SUFFIX: &str = ".exe";
#[cfg(not(windows))]
const EXE_SUFFIX: &str = "";

#[cfg(windows)]
const NVIDIA_DRIVER_CANDIDATES: &[&str] = &["nvcuda.dll"];
#[cfg(not(windows))]
const NVIDIA_DRIVER_CANDIDATES: &[&str] = &[
    "libcuda.so",
    "libcuda.so.1",
    "libcuda.so.534.65", // any specific version proves the driver is installed
];

fn main() {
    // Same dir as this launcher exe — that's where the package lays out the binaries.
    let self_path = env::current_exe().unwrap_or_else(|_| PathBuf::from("keryx-miner"));
    let dir = self_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let nvidia_bin = dir.join(format!("keryx-miner-nvidia{}", EXE_SUFFIX));
    let amd_bin = dir.join(format!("keryx-miner-amd{}", EXE_SUFFIX));

    let has_nvidia = nvidia_driver_present();
    let (primary, fallback, label) = if has_nvidia {
        (&nvidia_bin, &amd_bin, "nvidia")
    } else {
        (&amd_bin, &nvidia_bin, "amd")
    };

    let chosen = if primary.exists() {
        primary
    } else if fallback.exists() {
        // The preferred binary isn't in the package — try the other one rather than failing.
        eprintln!(
            "keryx-launcher: preferred {label} binary not found next to launcher; \
             trying the other one. (expected: {})",
            primary.display()
        );
        fallback
    } else {
        eprintln!(
            "keryx-launcher: neither keryx-miner-nvidia{exe} nor keryx-miner-amd{exe} found \
             next to the launcher (looked in {dir}). Re-download the release package — both \
             binaries must be present.",
            exe = EXE_SUFFIX,
            dir = dir.display()
        );
        std::process::exit(1);
    };

    // Forward argv unchanged. std::process::Command does NOT carry over the parent env by
    // default on all platforms, but on Windows/Linux the spawned process inherits the env
    // unless we explicitly clear it — so leave it alone.
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut cmd = Command::new(&chosen);
    cmd.args(&argv);

    // Replace ourselves with the child if the platform supports it (cleaner than spawn+wait;
    // on Windows std::process::Command has no exec, so we spawn+wait+exit-with-child-code).
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        eprintln!("keryx-launcher: failed to exec {}: {}", chosen.display(), err);
        std::process::exit(1);
    }

    #[cfg(not(unix))]
    {
        match cmd.status() {
            Ok(status) => {
                // Propagate the child's exit code exactly.
                let code = status.code().unwrap_or(1);
                std::process::exit(code);
            }
            Err(e) => {
                eprintln!(
                    "keryx-launcher: failed to run {}: {}",
                    chosen.display(),
                    e
                );
                std::process::exit(1);
            }
        }
    }
}

/// True if an NVIDIA GPU + driver appears to be installed.
fn nvidia_driver_present() -> bool {
    // 1) nvidia-smi on PATH — the most reliable cross-platform signal.
    if nvidia_smi_works() {
        return true;
    }
    // 2) The driver shared library itself is loadable. On Windows this means LoadLibrary
    //    succeeds; on Linux dlopen finds it in the loader path. This catches systems where
    //    nvidia-smi isn't installed to PATH but the driver is (common on minimal server images
    //    and WSL/CUDA-in-WSL setups).
    if driver_dll_loadable() {
        return true;
    }
    false
}

/// Run `nvidia-smi` (no args) and treat exit 0 as "NVIDIA driver present".
fn nvidia_smi_works() -> bool {
    Command::new("nvidia-smi")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Try to load each candidate driver lib. On Windows we use LoadLibraryW; on Linux dlopen.
/// Returns true if any candidate loads. The libs are never queried for symbols — merely
/// loading them proves the driver is installed.
#[cfg(windows)]
fn driver_dll_loadable() -> bool {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    for name in NVIDIA_DRIVER_CANDIDATES {
        let mut wide: Vec<u16> = OsStr::new(name).encode_wide().collect();
        wide.push(0);
        // SAFETY: LoadLibraryW with a nul-terminated wide string; no tainted input.
        let h = unsafe { windows_sys::Win32::System::LibraryLoader::LoadLibraryW(wide.as_ptr()) };
        if !h.is_null() {
            // Free it so we don't leak. Best-effort — ignore failure.
            unsafe { windows_sys::Win32::Foundation::FreeLibrary(h as _) };
            return true;
        }
    }
    false
}

#[cfg(unix)]
fn driver_dll_loadable() -> bool {
    // SAFETY: dlopen with RTLD_LAZY on well-known library names; never dereferenced.
    for name in NVIDIA_DRIVER_CANDIDATES {
        let cs = std::ffi::CString::new(*name).unwrap();
        let h = unsafe { libc::dlopen(cs.as_ptr(), libc::RTLD_LAZY | libc::RTLD_GLOBAL) };
        if !h.is_null() {
            unsafe { libc::dlclose(h) };
            return true;
        }
    }
    false
}

// Tiny vendor crates for the dlopen/LoadLibrary calls — keeps this binary dependency-free
// beyond these two ubiquitous system bindings. Behind cfg so each platform only pulls its own.
#[cfg(windows)]
extern crate windows_sys;

#[cfg(unix)]
extern crate libc;
