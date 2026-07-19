#!/usr/bin/env bash
# Cross-platform CUDA build + test driver for lfm25-native-cpp.
#
# On a Windows POSIX layer (WSL2, Git Bash / MSYS2, Cygwin) the script:
#   * generates a .bat file that locates vcvars64.bat, locates nvcc.exe,
#     activates the MSVC environment, then runs cmake configure+build
#   * executes that .bat via cmd.exe so the MSVC environment is active
#   * uses the Ninja generator (the Visual Studio generator fails with
#     "No CUDA toolset found" when nvcc is not registered as a VS extension)
#   * passes -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler so newer MSVC
#     Build Tools (VS 18 / MSVC 19.50) still compile .cu files under
#     nvcc 12.x/13.x (nvcc officially only supports MSVC 2019-2022)
#   * runs each test .exe directly (ctest can hang on Windows)
#
# On Linux/macOS the script assumes nvcc and the C++ compiler are already
# on PATH.
#
# Environment overrides:
#   BUILD_DIR               output directory (default: ./build-cuda)
#   BUILD_TYPE              Release|Debug|RelWithDebInfo (default: Release)
#   CUDA_ARCHITECTURES      CMAKE_CUDA_ARCHITECTURES (default: 86 on Windows
#                           to match the tested RTX 3060, native on Unix)
#   VCVARSALL               full path to vcvars64.bat (Windows; auto-detected)
#   CMAKE_CUDA_COMPILER     full path to nvcc.exe  (Windows; auto-detected)
#   RUN_TESTS               ON|OFF (default: ON)
#   NVCC_ALLOW_UNSUPPORTED  ON|OFF (default: ON)
#
# Usage:
#   ./scripts/cuda_build_test.sh                configure + build + test
#   RUN_TESTS=OFF ./scripts/cuda_build_test.sh  skip tests
#   CUDA_ARCHITECTURES=89 ./scripts/cuda_build_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-cuda}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
RUN_TESTS="${RUN_TESTS:-ON}"
NVCC_ALLOW_UNSUPPORTED="${NVCC_ALLOW_UNSUPPORTED:-ON}"

# ---------------------------------------------------------------------------
# Detect a Windows POSIX layer (WSL2, Git Bash / MSYS2, Cygwin). Native
# Windows bash reports as MINGW*/MSYS*/CYGWIN* in `uname -s`; WSL2 reports
# as Linux but exposes Microsoft in /proc/version.
# ---------------------------------------------------------------------------
WIN_ENV=""
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        WIN_ENV="msys"
        ;;
    *)
        if [[ -r /proc/version ]] && grep -qi microsoft /proc/version 2>/dev/null; then
            WIN_ENV="wsl"
        fi
        ;;
esac

if [[ -n "$WIN_ENV" ]]; then
    CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-86}"
    case "$WIN_ENV" in
        msys) PATHCONV="cygpath" ;;
        wsl)  PATHCONV="wslpath" ;;
    esac

    if ! command -v cmd.exe >/dev/null 2>&1; then
        echo "cuda_build_test: error: cmd.exe not found on PATH." >&2
        exit 1
    fi

    ROOT_WIN=$($PATHCONV -m "$ROOT")
    BUILD_WIN=$($PATHCONV -m "$BUILD_DIR")

    # Pass user-supplied overrides into the .bat environment.
    VCVARSALL_ENV="${VCVARSALL:-}"
    NVCC_ENV="${CMAKE_CUDA_COMPILER:-}"
    UNSUPPORTED_FLAG=""
    if [[ "$NVCC_ALLOW_UNSUPPORTED" == "ON" ]]; then
        UNSUPPORTED_FLAG='-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler'
    fi

    # Resolve a Windows-accessible temp directory for the .bat file. The
    # default `mktemp` location in WSL (/tmp) maps to a UNC path that
    # cmd.exe cannot execute .bat files from; we need a real Windows drive.
    WINTEMP=$(cmd.exe /c "echo %TEMP%" 2>/dev/null | tr -d '\r' | head -1)
    if [[ -z "$WINTEMP" ]]; then
        echo "cuda_build_test: error: could not resolve Windows TEMP dir." >&2
        exit 1
    fi
    WINTEMP_POSIX=$($PATHCONV -u "$WINTEMP")
    TMPBAT="$WINTEMP_POSIX/cuda_build_test_$$.bat"
    trap 'rm -f "$TMPBAT"' EXIT
    cat > "$TMPBAT" <<EOF
@echo off
setlocal enabledelayedexpansion

rem ---- 1. Locate vcvars64.bat -------------------------------------------
set "VCVARSALL=$VCVARSALL_ENV"
if "!VCVARSALL!"=="" (
    set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq delims=" %%i in (\`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath\`) do set "VSINSTALL=%%i"
        if exist "!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat" set "VCVARSALL=!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat"
    )
)
if "!VCVARSALL!"=="" (
    for %%P in (
        "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    ) do if exist %%~P set "VCVARSALL=%%~P"
)
if "!VCVARSALL!"=="" (
    echo cuda_build_test: error: vcvars64.bat not found. 1>&2
    echo   Install Visual Studio 2022 Build Tools ^(C++ workload^) or 1>&2
    echo   set VCVARSALL to the full path of vcvars64.bat. 1>&2
    exit /b 1
)

echo cuda_build_test: MSVC = !VCVARSALL!
call "!VCVARSALL!" x64
if errorlevel 1 exit /b 1

rem ---- 2. Locate nvcc.exe ------------------------------------------------
set "NVCC_EXE=$NVCC_ENV"
if not "!NVCC_EXE!"=="" goto :nvcc_done
where nvcc.exe >nul 2>nul
if errorlevel 1 goto :nvcc_search
for /f "usebackq delims=" %%i in (\`where nvcc.exe\`) do set "NVCC_EXE=%%i"
goto :nvcc_done
:nvcc_search
for %%P in (
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe"
) do if exist %%~P set "NVCC_EXE=%%~P"
:nvcc_done
if "!NVCC_EXE!"=="" (
    echo cuda_build_test: error: nvcc.exe not found on PATH or CUDA Toolkit 1>&2
    echo   install location. Set CMAKE_CUDA_COMPILER to the full path of nvcc.exe. 1>&2
    exit /b 1
)

rem ---- 3. Locate ninja ---------------------------------------------------
where ninja >nul 2>nul
if errorlevel 1 (
    echo cuda_build_test: error: ninja not found on PATH. 1>&2
    echo   Install Ninja ^(pip install ninja^) and ensure it is on PATH. 1>&2
    exit /b 1
)

rem ---- 4. Configure + build ---------------------------------------------
echo cuda_build_test: configuring into $BUILD_WIN (Ninja, $BUILD_TYPE, sm_$CUDA_ARCHITECTURES)
cmake -S "$ROOT_WIN" -B "$BUILD_WIN" -G Ninja -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES -DLFM_BUILD_TESTS=ON -DCMAKE_CUDA_COMPILER="!NVCC_EXE!" $UNSUPPORTED_FLAG
if errorlevel 1 exit /b 1
echo cuda_build_test: building
cmake --build "$BUILD_WIN" --parallel
if errorlevel 1 exit /b 1
endlocal
EOF
    # Convert the POSIX .bat path back to Windows form for cmd.exe.
    TMPBAT_WIN=$($PATHCONV -w "$TMPBAT")
    cmd.exe /c "$TMPBAT_WIN"

    if [[ "$RUN_TESTS" == "ON" ]]; then
        echo "cuda_build_test: running tests"
        # Run each test executable directly. On Windows, ctest can hang when
        # a target waits on stdin; iterating the .exe files is reliable and
        # also produces per-test PASS/FAIL lines.
        failed=0
        shopt -s nullglob
        for test_exe in "$BUILD_DIR"/*_test.exe; do
            name=$(basename "$test_exe")
            if "$test_exe" >/dev/null 2>&1; then
                echo "  ok   $name"
            else
                echo "  FAIL $name"
                failed=1
            fi
        done
        shopt -u nullglob
        if [[ "$failed" -ne 0 ]]; then
            echo "cuda_build_test: one or more tests failed" >&2
            exit 1
        fi
    fi
    echo "cuda_build_test: build complete (Windows/$WIN_ENV, Ninja, sm_$CUDA_ARCHITECTURES)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Native Unix: assume nvcc and the C++ compiler are already on PATH.
# ---------------------------------------------------------------------------
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-native}"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "cuda_build_test: error: nvcc not found on PATH." >&2
    exit 1
fi

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    GENERATOR_ARGS=(-G Ninja)
else
    echo "cuda_build_test: warning: ninja not found, using default generator."
fi

cmake -S "$ROOT" -B "$BUILD_DIR" "${GENERATOR_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
    -DLFM_BUILD_TESTS=ON

cmake --build "$BUILD_DIR" --parallel

if [[ "$RUN_TESTS" == "ON" ]]; then
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

echo "cuda_build_test: build complete (Unix, $CUDA_ARCHITECTURES)"
