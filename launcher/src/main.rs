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

    // Decide which binary to run based on the GPU makeup of this machine:
    //   - NVIDIA-only rig   -> nvidia binary (CUDA + Vulkan, tuned sm_xx kernels).
    //   - AMD/Intel-only    -> amd binary (Vulkan only — runs anywhere, no CUDA deps).
    //   - Heterogeneous     -> amd binary too. The CUDA build hard-links nvcuda.dll and its
    //     Vulkan desktop dispatch is cfg'd out, so it would mine only the NVIDIA card and leave
    //     the AMD/Intel card idle. The amd binary runs every card through Vulkan instead — yes,
    //     the NVIDIA card loses its sm_xx CUDA kernel (Vulkan shader instead), but every card
    //     actually mines, which is the better trade on a mixed rig. (If you want per-vendor
    //     kernels on a mixed rig, run two single-vendor processes with CUDA_VISIBLE_DEVICES /
    //     VK_DEVICE_FILTER.)
    //
    // Detection: enumerate display adapters via the OS and classify each as NVIDIA or not.
    // We only need a count, not full Vulkan init — keeps the launcher fast and dependency-free.
    let (nvidia_count, other_count) = count_gpus_by_vendor();
    let has_nvidia = nvidia_count > 0;
    let has_other = other_count > 0;
    let heterogeneous = has_nvidia && has_other;
    let use_nvidia_binary = has_nvidia && !heterogeneous;

    let (primary, fallback, label) = if use_nvidia_binary {
        (&nvidia_bin, &amd_bin, "nvidia")
    } else {
        (&amd_bin, &nvidia_bin, "amd")
    };

    // One-line decision log so users can see why the launcher picked what it did (helpful on a
    // mixed rig where the NVIDIA binary might feel like the "obvious" choice but isn't).
    if heterogeneous {
        eprintln!(
            "keryx-launcher: heterogeneous rig detected ({} NVIDIA + {} AMD/Intel GPU(s)) — \
             running the amd (Vulkan) binary so every card mines.",
            nvidia_count, other_count
        );
    }

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

/// Count installed GPUs by vendor class: `(nvidia_count, other_count)`.
///
/// Used to detect heterogeneous rigs (NVIDIA + AMD/Intel in one box). On Windows we use
/// `wmic path win32_VideoController get name` (ubiquitous, no deps). On Linux we parse
/// `/sys/class/drm/card*/device/vendor` (PCI vendor IDs: 0x10de = NVIDIA, 0x1002 = AMD,
/// 0x8086 = Intel). Falls back to `(nvidia_driver_present() as u32, 0)` if enumeration fails
/// — i.e. a pure NVIDIA-or-not decision, never a wrong heterogeneous classification.
fn count_gpus_by_vendor() -> (u32, u32) {
    #[cfg(windows)]
    {
        let (n, o) = count_gpus_wmic();
        if n + o > 0 {
            return (n, o);
        }
    }
    #[cfg(unix)]
    {
        let (n, o) = count_gpus_linux();
        if n + o > 0 {
            return (n, o);
        }
    }
    // Enumeration failed entirely — fall back to the simple "is NVIDIA driver present?" check.
    // This never produces a heterogeneous classification, so it can't accidentally pick the
    // wrong binary; it just degrades to the pre-heterogeneous-handling behaviour.
    (nvidia_driver_present() as u32, 0)
}

/// Windows: enumerate display adapters via CIM and classify by name.
///
/// Tries `Get-CimInstance Win32_VideoController` (PowerShell, on every modern Windows) first,
/// then falls back to `wmic` (the legacy CLI, removed from Windows 11 24H2+ but present on
/// older builds and Server). Both return the same `Name` strings — we classify by substring.
#[cfg(windows)]
fn count_gpus_wmic() -> (u32, u32) {
    // Try PowerShell (modern path), then legacy wmic.
    let names: Option<Vec<String>> = powershell_adapter_names().or_else(wmic_adapter_names);
    let Some(names) = names else { return (0, 0) };

    let mut nvidia = 0u32;
    let mut other = 0u32;
    for raw in names {
        let l = raw.trim().to_ascii_lowercase();
        // Skip the "name" header and blank lines.
        if l.is_empty() || l == "name" {
            continue;
        }
        // Basic display adapters (Microsoft Basic Render, Remote Desktop, etc.) are not GPUs we
        // would ever mine on — exclude them so they don't pollute the "other" count and trigger
        // a false heterogeneous detection. Virtual display adapters (Parsec IDR, SpaceDesk,
        // Duet, headless HDMI plugs' EDID emulators, virtual monitors) fall in the same bucket —
        // they have no compute capability at all.
        const VIRTUAL_OR_FAKE: &[&str] = &[
            "microsoft",
            "basic render",
            "remote desktop",
            "parsec",
            "spacedesk",
            "duet display",
            "virtual",
            "emulated",
            // EDID / dummy-plug adapters and other software-only renderers don't expose a real
            // GPU name; they usually carry these substrings.
            "dummy",
            "hdmi plug",
        ];
        if VIRTUAL_OR_FAKE.iter().any(|needle| l.contains(needle)) {
            continue;
        }
        if l.contains("nvidia") || l.contains("geforce") || l.contains("quadro") || l.contains("tesla") {
            nvidia += 1;
        } else {
            other += 1;
        }
    }
    (nvidia, other)
}

/// PowerShell: `Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name`.
/// Returns one adapter name per line. Returns None if PowerShell isn't available (very rare).
#[cfg(windows)]
fn powershell_adapter_names() -> Option<Vec<String>> {
    let out = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(|s| s.to_string())
            .collect(),
    )
}

/// Legacy `wmic path win32_VideoController get name` — same output, pre-PowerShell era.
/// Removed from Windows 11 24H2+, still on Server and older builds.
#[cfg(windows)]
fn wmic_adapter_names() -> Option<Vec<String>> {
    let out = Command::new("wmic")
        .args(["path", "win32_VideoController", "get", "name"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(|s| s.to_string())
            .collect(),
    )
}

/// Linux: read PCI vendor IDs from /sys/class/drm/card*/device/vendor.
#[cfg(unix)]
fn count_gpus_linux() -> (u32, u32) {
    let mut nvidia = 0u32;
    let mut other = 0u32;
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else { return (0, 0) };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        // Only real GPUs: card0, card1, ... (not card0-DP-1 render nodes etc.)
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let vendor_path = entry.path().join("device").join("vendor");
        let Ok(vendor) = std::fs::read_to_string(&vendor_path) else { continue };
        match vendor.trim() {
            "0x10de" => nvidia += 1,
            "0x1002" | "0x8086" | "0x10003" => other += 1, // AMD / Intel / VMware (don't mine, but count as non-NVIDIA)
            _ => other += 1, // unknown vendor — treat as non-NVIDIA rather than ignore
        }
    }
    (nvidia, other)
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
