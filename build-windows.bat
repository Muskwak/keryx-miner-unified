@echo off
REM build-windows.bat -- one-click Windows build for keryx-miner (unified fork).
REM
REM Always builds CUDA (unconditional on desktop -- see Cargo.toml). Additionally builds the
REM desktop Vulkan backend + in-process llama.cpp inference engine (--features vulkan) when the
REM required toolchain is detected:
REM   - LLVM (libclang.dll, for the llama.cpp FFI bindgen step) -- set LIBCLANG_PATH, or this
REM     script auto-detects a few common install locations.
REM   - Ninja on PATH -- the default Visual Studio CMake generator fails to build llama.cpp's
REM     vulkan-shaders-gen subproject on Windows (a Windows MAX_PATH / ExternalProject quirk);
REM     Ninja avoids it. See IMPLEMENTATION_NOTES.md for details.
REM   - cmake and the Vulkan SDK (glslc) on PATH.
REM
REM Run it from anywhere; it cd's to its own directory first.
REM   build-windows.bat            -> auto-detect, build the best available feature set
REM   build-windows.bat cuda       -> force CUDA-only
REM   build-windows.bat cuda,vulkan -> force the full heterogeneous-rig build (fails loudly if
REM                                    the Vulkan toolchain isn't actually present)

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  keryx-miner-unified -- Windows build
echo ============================================================

REM --- nvcc is required unconditionally (candle-core's cuda feature is hardcoded on desktop) ---
where nvcc >nul 2>nul
if errorlevel 1 (
    echo [build] ERROR: nvcc not found on PATH.
    echo         Install the CUDA Toolkit -- this fork's desktop build requires CUDA
    echo         unconditionally on Windows/Linux ^(there is no CPU-only build^).
    exit /b 1
)

REM --- Explicit feature override wins outright ---
if not "%~1"=="" (
    set "FEATURES=%~1"
    echo [build] Features forced via argument: !FEATURES!
    goto :run_build
)

REM --- Otherwise auto-detect the Vulkan toolchain ---
set "LIBCLANG_CANDIDATE="
if defined LIBCLANG_PATH (
    if exist "%LIBCLANG_PATH%\libclang.dll" set "LIBCLANG_CANDIDATE=%LIBCLANG_PATH%"
)
if not defined LIBCLANG_CANDIDATE (
    for %%P in (
        "C:\Program Files\LLVM\bin"
        "%USERPROFILE%\llvm\bin"
        "%USERPROFILE%\scoop\apps\llvm\current\bin"
    ) do (
        if exist "%%~P\libclang.dll" set "LIBCLANG_CANDIDATE=%%~P"
    )
)

set "HAVE_NINJA=0"
where ninja >nul 2>nul && set "HAVE_NINJA=1"

set "HAVE_CMAKE=0"
where cmake >nul 2>nul && set "HAVE_CMAKE=1"

set "HAVE_GLSLC=0"
where glslc >nul 2>nul && set "HAVE_GLSLC=1"

if defined LIBCLANG_CANDIDATE if "%HAVE_NINJA%"=="1" if "%HAVE_CMAKE%"=="1" if "%HAVE_GLSLC%"=="1" (
    set "LIBCLANG_PATH=%LIBCLANG_CANDIDATE%"
    set "CMAKE_GENERATOR=Ninja"
    set "FEATURES=cuda,vulkan"
    echo [build] Vulkan toolchain found ^(libclang: !LIBCLANG_PATH!^) -- building --features !FEATURES!
) else (
    set "FEATURES=cuda"
    echo [build] Vulkan toolchain incomplete -- building CUDA-only. To enable Vulkan, install:
    if not defined LIBCLANG_CANDIDATE echo           - LLVM ^(for libclang.dll^) and set LIBCLANG_PATH, e.g.: winget install LLVM.LLVM
    if "%HAVE_NINJA%"=="0"  echo           - Ninja on PATH, e.g.: winget install Ninja-build.Ninja
    if "%HAVE_CMAKE%"=="0"  echo           - CMake on PATH, e.g.: winget install Kitware.CMake
    if "%HAVE_GLSLC%"=="0"  echo           - the Vulkan SDK ^(for glslc^): https://vulkan.lunarg.com/
)

:run_build

REM --- The vulkan feature pulls llama.cpp's vendored CMake build, which nests deep enough
REM     (target\release\build\llama-cpp-sys-2-<hash>\out\build\ggml\...\vulkan-shaders-gen-build\
REM     CMakeFiles\CMakeScratch\TryCompile-<id>\testCXXCompiler.cxx) to blow past Windows'
REM     260-char MAX_PATH under a plain `target\release` -- cl.exe fails with
REM     "fatal error C1083: Cannot open compiler generated file: ''". Use a short
REM     CARGO_TARGET_DIR to dodge it, unless the caller already set one. See
REM     IMPLEMENTATION_NOTES.md for the full diagnosis. ---
echo !FEATURES! | findstr /C:"vulkan" >nul
if not errorlevel 1 (
    if not defined CARGO_TARGET_DIR (
        set "CARGO_TARGET_DIR=%SystemDrive%\kmu-build"
        echo [build] vulkan feature detected -- using a short CARGO_TARGET_DIR=!CARGO_TARGET_DIR!
        echo         to avoid a Windows MAX_PATH failure in llama.cpp's vendored CMake subbuild.
        echo         Override by setting CARGO_TARGET_DIR yourself before running this script.
    )
)

echo [build] Running: cargo build --release --features !FEATURES!
cargo build --release --features !FEATURES!
if errorlevel 1 (
    echo [build] FAILED
    exit /b 1
)

if defined CARGO_TARGET_DIR (
    echo [build] Success: !CARGO_TARGET_DIR!\release\keryx-miner.exe
) else (
    echo [build] Success: target\release\keryx-miner.exe
)
endlocal
