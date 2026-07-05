#!/usr/bin/env bash
#
# build-linux.sh -- one-click Linux build for keryx-miner (unified fork).
#
# Always builds CUDA (unconditional on desktop -- see Cargo.toml; there is no CPU-only build).
# Additionally builds the desktop Vulkan backend + in-process llama.cpp inference engine
# (--features vulkan) when the required toolchain is detected: cmake, glslc (part of a Vulkan
# SDK / shaderc -- see https://vulkan.lunarg.com/, or the LunarG apt repo), and libclang (for the
# llama.cpp FFI bindgen step). Unlike Windows, Linux does NOT need CMAKE_GENERATOR=Ninja -- the
# default Makefiles generator builds llama.cpp's vulkan-shaders-gen subproject fine here (a
# Windows-only MAX_PATH/ExternalProject quirk -- see IMPLEMENTATION_NOTES.md).
#
# Usage:
#   ./build-linux.sh                 # auto-detect, build the best available feature set
#   ./build-linux.sh cuda            # force CUDA-only
#   ./build-linux.sh cuda,vulkan     # force the full heterogeneous-rig build (fails loudly if
#                                    #   the Vulkan toolchain isn't actually present)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"

echo "============================================================"
echo " keryx-miner-unified -- Linux build"
echo "============================================================"

# --- nvcc is required unconditionally (candle-core's cuda feature is hardcoded on desktop) ---
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[build] ERROR: nvcc not found on PATH." >&2
  echo "        Install the CUDA Toolkit -- this fork's desktop build requires CUDA" >&2
  echo "        unconditionally on Linux/Windows (there is no CPU-only build)." >&2
  exit 1
fi

# --- bindgen_cuda (candle-kernels' own build script) shells out to nvidia-smi to autodetect the
#     GPU's compute capability. Without a driver (e.g. building in a container with no GPU
#     passthrough) that call fails outright, so set a sane default explicitly. ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[build] nvidia-smi not found -- no GPU driver visible in this environment."
  echo "        Defaulting CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-86} (Ampere) to skip"
  echo "        bindgen_cuda's autodetect. Override with CUDA_COMPUTE_CAP=<xy> for other hardware"
  echo "        (e.g. 61 for Pascal/P40, 89 for Ada)."
  export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-86}"
fi

# --- Explicit feature override wins outright ---
if [[ $# -ge 1 ]]; then
  FEATURES="$1"
  echo "[build] Features forced via argument: $FEATURES"
else
  # --- Otherwise auto-detect the Vulkan toolchain ---
  have_cmake=0;  command -v cmake  >/dev/null 2>&1 && have_cmake=1
  have_glslc=0;  command -v glslc  >/dev/null 2>&1 && have_glslc=1
  have_clang=0
  if command -v clang >/dev/null 2>&1 || ldconfig -p 2>/dev/null | grep -q 'libclang\.so'; then
    have_clang=1
  fi

  if [[ $have_cmake -eq 1 && $have_glslc -eq 1 && $have_clang -eq 1 ]]; then
    FEATURES="cuda,vulkan"
    echo "[build] Vulkan toolchain found -- building --features $FEATURES"
  else
    FEATURES="cuda"
    echo "[build] Vulkan toolchain incomplete -- building CUDA-only. To enable Vulkan, install:"
    [[ $have_cmake -eq 0 ]] && echo "          - cmake (e.g. apt install cmake)"
    [[ $have_glslc -eq 0 ]] && echo "          - a Vulkan SDK providing glslc, e.g. via LunarG's apt repo:" \
      && echo "              curl -sSL https://packages.lunarg.com/lunarg-signing-key-pub.asc -o /etc/apt/trusted.gpg.d/lunarg.asc" \
      && echo "              curl -sSL https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list -o /etc/apt/sources.list.d/lunarg-vulkan-jammy.list" \
      && echo "              apt update && apt install vulkan-sdk"
    [[ $have_clang -eq 0 ]] && echo "          - clang/libclang (e.g. apt install clang libclang-dev)"
  fi
fi

echo "[build] Running: cargo build --release --features $FEATURES"
cargo build --release --features "$FEATURES"

echo "[build] Success: target/release/keryx-miner"
