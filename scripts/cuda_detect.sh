#!/usr/bin/env bash
# cuda_detect.sh — quick CUDA availability probe for lfm25-native-cpp.
#
# Detects whether the runtime has everything required to BUILD and RUN the
# CUDA backend:
#   * a CUDA toolkit (nvcc)          -> CUDA_TOOLKIT
#   * a usable NVIDIA GPU + driver   -> CUDA_DEVICE / CUDA_DRIVER
#   * a GPU matching the build arch  -> CUDA_COMPATIBLE (best-effort)
#
# Exit codes:
#   0  CUDA backend is usable (toolkit + driver + device found)
#   1  toolkit present but no GPU/driver visible
#   2  no CUDA toolkit found
#   3  could not determine (unsupported environment)
#
# This script is intentionally self-contained and meant to be run by humans
# and by build/CI tooling. It prints a single human-readable summary and also
# exports the findings as shell variables (CUDA_TOOLKIT, CUDA_DEVICE,
# CUDA_DRIVER, CUDA_VERSION) so it can be sourced by other scripts.
#
# Environment overrides:
#   CMAKE_CUDA_COMPILER   full path to nvcc (skips auto-detection)
#   CUDA_ARCHITECTURES    sm_XX to check device compute capability against
#                         (default: 86, matching the tested RTX 3060)
set -uo pipefail

CUDA_TOOLKIT=""
CUDA_VERSION=""
CUDA_DEVICE=""
CUDA_DRIVER=""
CUDA_COMPATIBLE=""

# ---------------------------------------------------------------------------
# Windows POSIX layer detection (MINGW/MSYS/CYGWIN or WSL2).
# ---------------------------------------------------------------------------
WIN_ENV=""
case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*) WIN_ENV="msys" ;;
    *)
        if [[ -r /proc/version ]] && grep -qi microsoft /proc/version 2>/dev/null; then
            WIN_ENV="wsl"
        fi
        ;;
esac

detect_unix() {
    if [[ -n "${CMAKE_CUDA_COMPILER:-}" ]]; then
        NVCC="$CMAKE_CUDA_COMPILER"
    elif command -v nvcc >/dev/null 2>&1; then
        NVCC="$(command -v nvcc)"
    else
        NVCC=""
    fi

    if [[ -n "$NVCC" ]]; then
        CUDA_TOOLKIT="$NVCC"
        CUDA_VERSION="$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | awk '{print $2}')"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
        CUDA_DEVICE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    fi
}

detect_windows() {
    if [[ -n "${CMAKE_CUDA_COMPILER:-}" ]]; then
        NVCC="${CMAKE_CUDA_COMPILER//\//\\}"
    else
        NVCC="$(cmd.exe //c "where nvcc.exe" 2>/dev/null | tr -d '\r' | head -1 || true)"
    fi

    if [[ -n "$NVCC" && -f "$NVCC" ]]; then
        CUDA_TOOLKIT="$NVCC"
        # Invoke nvcc through cmd.exe using PATH lookup (avoids MSYS path
        # mangling of the full Windows path with backslashes).
        CUDA_VERSION="$(cmd.exe //c "nvcc --version" 2>/dev/null | tr -d '\r' | grep -o 'release [0-9.]*' | awk '{print $2}' | head -1 || true)"
    fi

    local smi
    smi="$(cmd.exe //c 'where nvidia-smi.exe' 2>/dev/null | tr -d '\r' | head -1 || true)"
    if [[ -n "$smi" && -f "$smi" ]]; then
        CUDA_DRIVER="$(cmd.exe //c "nvidia-smi --query-gpu=driver_version --format=csv,noheader" 2>/dev/null | tr -d '\r' | head -1 || true)"
        CUDA_DEVICE="$(cmd.exe //c "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null | tr -d '\r' | head -1 || true)"
    fi
}

if [[ -n "$WIN_ENV" ]]; then
    detect_windows
else
    detect_unix
fi

# Best-effort compute-capability compatibility check (Unix only: nvidia-smi
# exposes cuda_max_cap; Windows falls back to the configured arch assumption).
if [[ -z "$WIN_ENV" && -n "$(command -v nvidia-smi 2>/dev/null)" ]]; then
    MAX_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
    TARGET="${CUDA_ARCHITECTURES:-86}"
    if [[ -n "$MAX_CAP" && "$MAX_CAP" -ge "${TARGET//sm_/}" ]]; then
        CUDA_COMPATIBLE="yes"
    else
        CUDA_COMPATIBLE="no"
    fi
elif [[ -n "$WIN_ENV" && -n "$CUDA_DEVICE" ]]; then
    # Assume compatible when a device is present on Windows (toolkit arch check
    # is performed at build time by cmake, not here).
    CUDA_COMPATIBLE="assumed"
fi

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
echo "cuda_detect: CUDA toolkit : ${CUDA_TOOLKIT:-<none>}"
echo "cuda_detect: CUDA version : ${CUDA_VERSION:-<unknown>}"
echo "cuda_detect: GPU device   : ${CUDA_DEVICE:-<none>}"
echo "cuda_detect: GPU driver   : ${CUDA_DRIVER:-<none>}"
echo "cuda_detect: compatible   : ${CUDA_COMPATIBLE:-<unknown>}"

if [[ -z "$CUDA_TOOLKIT" ]]; then
    echo "cuda_detect: RESULT: no CUDA toolkit -> CUDA backend NOT buildable"
    exit 2
fi
if [[ -z "$CUDA_DEVICE" ]]; then
    echo "cuda_detect: RESULT: toolkit present but no GPU/driver -> buildable, not runnable"
    exit 1
fi
echo "cuda_detect: RESULT: CUDA backend usable"
exit 0
