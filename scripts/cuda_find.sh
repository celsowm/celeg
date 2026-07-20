#!/usr/bin/env bash
# cuda_find.sh — enumerate installed CUDA toolkits and pick the right one for the
#                 detected GPU. Cross-platform: Linux/macOS and Windows
#                 (Git Bash / MSYS / CyGWIN / WSL).
#
# Why this exists:
#   A machine can have several CUDA toolkits installed (e.g. the distro's
#   CUDA 12.0 on PATH plus a manually installed CUDA 13.2). The default `nvcc`
#   may be too old for the local GPU (a Blackwell RTX 5090 needs sm_120, which
#   only CUDA 12.8+ can compile). This script finds every toolkit, reads each
#   nvcc's real architecture support, compares it to the GPU's compute
#   capability, and prints the toolkit(s) that can actually build & run here.
#
# Output:
#   * a human-readable table of every detected toolkit
#   * the detected GPU + compute capability
#   * the recommended toolkit and the CMAKE_CUDA_ARCHITECTURES value
#   * a copy/paste `export` block (also written to cuda_env.sh when run with
#     --write-env)
#
# Exit codes:
#   0  at least one toolkit can target the detected GPU
#   1  toolkits found but none supports the GPU
#   2  no CUDA toolkit found at all
#   3  unsupported / undetermined environment (no gpu detected either)
#
# Overrides:
#   CMAKE_CUDA_COMPILER   full path to a specific nvcc to evaluate
#   CUDA_ARCHITECTURES    force the target sm_XX (skips GPU auto-detection)
set -uo pipefail

WRITE_ENV=0
for arg in "$@"; do
    case "$arg" in
        --write-env) WRITE_ENV=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
    esac
done

# ---------------------------------------------------------------------------
# OS layer
# ---------------------------------------------------------------------------
OS="$(uname -s 2>/dev/null || echo Unknown)"
case "$OS" in
    Linux*)  PLAT="linux" ;;
    Darwin*) PLAT="macos" ;;
    MINGW*|MSYS*|CYGWIN*) PLAT="windows" ;;
    *)
        if [[ -r /proc/version ]] && grep -qi microsoft /proc/version 2>/dev/null; then
            PLAT="linux"   # WSL2
        else
            PLAT="unknown"
        fi
        ;;
esac

# Run a Windows command portably (Git Bash / MSYS).
win_cmd() { cmd.exe //c "$*" 2>/dev/null | tr -d '\r'; }

# ---------------------------------------------------------------------------
# Map a compute capability (e.g. "12.0") to the minimum CUDA version that
# supports it, and the sm_ token.
# ---------------------------------------------------------------------------
# Returns (echoes) the sm_XX code for a "major.minor" capability.
cap_to_sm() {
    local maj="${1%%.*}" min="${1#*.}"
    printf 'sm_%d%d' "$maj" "$min"
}

# Minimum CUDA release (major.minor) that introduced support for an sm_ code.
min_cuda_for_sm() {
    case "$1" in
        sm_70|sm_72)            echo "9.0" ;;
        sm_75)                  echo "10.0" ;;
        sm_80|sm_86)            echo "11.0" ;;
        sm_87)                  echo "11.1" ;;
        sm_90|sm_90a)           echo "11.8" ;;
        sm_100|sm_100a)         echo "12.8" ;;  # Blackwell (GB100/GB200/RTX 50*)
        sm_120|sm_120a)         echo "12.8" ;;  # Blackwell consumer (RTX 5090)
        *)                      echo "0.0" ;;
    esac
}

# ---------------------------------------------------------------------------
# Enumerate candidate toolkit roots.
# ---------------------------------------------------------------------------
declare -a CANDIDATE_BINS=()

if [[ "$PLAT" == "windows" ]]; then
    # Common Windows install locations.
    for base in /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA \
                /c/ProgramData/NVIDIA\ Corporation/CUDA/v* ; do
        for d in "$base"/v*.* "$base"; do
            [[ -x "$d/bin/nvcc.exe" ]] && CANDIDATE_BINS+=("$d/bin/nvcc.exe")
        done
    done
    # Also whatever `where nvcc` finds.
    while IFS= read -r p; do
        [[ -n "$p" ]] && CANDIDATE_BINS+=("$p")
    done < <(win_cmd 'where nvcc.exe' | sed 's#\\#/#g')
else
    # Linux/macOS: /usr/local/cuda*, the alternatives symlink, and PATH.
    for d in /usr/local/cuda /usr/local/cuda-* ; do
        [[ -x "$d/bin/nvcc" ]] && CANDIDATE_BINS+=("$d/bin/nvcc")
    done
    shopt -s nullglob
    for d in /opt/cuda* ; do
        [[ -x "$d/bin/nvcc" ]] && CANDIDATE_BINS+=("$d/bin/nvcc")
    done
    shopt -u nullglob
    if command -v nvcc >/dev/null 2>&1; then
        CANDIDATE_BINS+=("$(command -v nvcc)")
    fi
fi

# De-duplicate while preserving order.
declare -a NVCC_BINS=()
seen=""
for b in "${CANDIDATE_BINS[@]:-}"; do
    [[ -z "$b" ]] && continue
    if [[ ";$seen;" != *";$b;"* ]]; then
        NVCC_BINS+=("$b"); seen="$seen$b;"
    fi
done

# Explicit override wins and replaces the list.
if [[ -n "${CMAKE_CUDA_COMPILER:-}" ]]; then
    NVCC_BINS=("$CMAKE_CUDA_COMPILER")
fi

# ---------------------------------------------------------------------------
# Detect GPU + compute capability.
# ---------------------------------------------------------------------------
GPU_NAME=""
GPU_CAP=""      # "major.minor"
GPU_SM=""       # "sm_XX"

detect_gpu() {
    if [[ "$PLAT" == "windows" ]]; then
        GPU_NAME="$(win_cmd 'nvidia-smi --query-gpu=name --format=csv,noheader' | head -1)"
        GPU_CAP="$(win_cmd 'nvidia-smi --query-gpu=compute_cap --format=csv,noheader' | head -1 | tr -d ' ')"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
        GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    fi
    if [[ -n "$GPU_CAP" ]]; then
        GPU_SM="$(cap_to_sm "$GPU_CAP")"
    fi
}

if [[ -z "${CUDA_ARCHITECTURES:-}" ]]; then
    detect_gpu
else
    GPU_SM="${CUDA_ARCHITECTURES}"
    GPU_CAP="${CUDA_ARCHITECTURES#sm_}"
    GPU_CAP="${GPU_CAP%a}"
    GPU_CAP="${GPU_CAP:0:1}.${GPU_CAP:1:1}"
    GPU_NAME="${GPU_NAME:-<forced via CUDA_ARCHITECTURES>}"
fi

# ---------------------------------------------------------------------------
# Probe each toolkit: version + whether it can target the GPU's sm_.
# ---------------------------------------------------------------------------
declare -a TK_PATH=() TK_VER=() TK_OK=() TK_REASON=()
RECOMMENDED=""
BEST_VER=""

nvcc_version() {
    # $1 = nvcc path. Prints "major.minor".
    local out
    if [[ "$PLAT" == "windows" && "$1" == *"nvcc.exe" ]]; then
        out="$(win_cmd "nvcc --version" | grep -o 'release [0-9.]*' | awk '{print $2}' | head -1)"
    else
        out="$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | awk '{print $2}' | head -1)"
    fi
    echo "$out"
}

# Query nvcc for the highest real architecture it knows (used only as a
# fallback when we cannot read the GPU capability).
nvcc_max_sm() {
    local nvcc="$1" out max_sm=""
    if [[ "$PLAT" == "windows" && "$nvcc" == *"nvcc.exe" ]]; then
        out="$(win_cmd "nvcc --help" 2>/dev/null)"
    else
        out="$(nvcc --help 2>/dev/null)"
    fi
    # Find the highest sm_XXX mentioned in the --gpu-architecture help text.
    max_sm="$(echo "$out" | grep -oE 'sm_[0-9]{2,3}' | sed 's/sm_//' | sort -n | tail -1)"
    if [[ -n "$max_sm" ]]; then
        printf 'sm_%s' "$max_sm"
    else
        echo ""
    fi
}

for nvcc in "${NVCC_BINS[@]:-}"; do
    [[ -z "$nvcc" ]] && continue
    ver="$(nvcc_version "$nvcc")"
    TK_PATH+=("$nvcc")
    TK_VER+=("$ver")

    if [[ -z "$ver" ]]; then
        TK_OK+=("no"); TK_REASON+=("nvcc unreadable"); continue
    fi

    if [[ -n "$GPU_SM" ]]; then
        # Does this toolkit's minimum required CUDA meet/exceed the GPU's need?
        need="$(min_cuda_for_sm "$GPU_SM")"
        # Compare version tuples "a.b".
        ok=0
        if [[ "$need" == "0.0" ]]; then
            ok=1   # unknown arch: let nvcc try
        else
            # Normalize to comparable integers a*100+b.
            need_n=$(( ${need%%.*} * 100 + ${need#*.} ))
            ver_n=$(( ${ver%%.*} * 100 + ${ver#*.} ))
            if [[ "$ver_n" -ge "$need_n" ]]; then ok=1; fi
        fi
        if [[ "$ok" -eq 1 ]]; then
            TK_OK+=("yes"); TK_REASON+=("supports $GPU_SM")
            if [[ -z "$RECOMMENDED" ]]; then RECOMMENDED="$nvcc"; BEST_VER="$ver"; fi
        else
            TK_OK+=("no"); TK_REASON+=("needs CUDA>=$need for $GPU_SM (has $ver)")
        fi
    else
        # No GPU info: report the toolkit's max known arch as informational.
        msm="$(nvcc_max_sm "$nvcc")"
        TK_OK+=("?"); TK_REASON+=("no GPU detected; max arch $msm")
    fi
done

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
echo "cuda_find: platform        : $PLAT ($OS)"
echo "cuda_find: GPU device      : ${GPU_NAME:-<none detected>}"
echo "cuda_find: compute cap     : ${GPU_CAP:-<unknown>}  (${GPU_SM:-<unknown>})"
echo
printf '%-4s %-28s %-10s %-5s %s\n' "USE" "TOOLKIT" "VERSION" "OK" "NOTE"
printf '%-4s %-28s %-10s %-5s %s\n' "---" "----------------------------" "----------" "-----" "--------------------"
for i in "${!TK_PATH[@]}"; do
    use="   "
    if [[ "${TK_PATH[$i]}" == "$RECOMMENDED" ]]; then use=">>>"; fi
    short="${TK_PATH[$i]}"
    # Shorten the printed path for readability.
    printf '%-4s %-28s %-10s %-5s %s\n' "$use" "${short:0:28}" "${TK_VER[$i]:-?}" "${TK_OK[$i]}" "${TK_REASON[$i]}"
done

echo
if [[ ${#TK_PATH[@]} -eq 0 ]]; then
    echo "cuda_find: RESULT: no CUDA toolkit found."
    exit 2
fi

if [[ -n "$RECOMMENDED" ]]; then
    echo "cuda_find: RESULT: use '$RECOMMENDED' (CUDA ${BEST_VER}) for this GPU."
    echo
    echo "# Copy/paste to build against the recommended toolkit:"
    if [[ "$PLAT" == "windows" ]]; then
        rec_win="${RECOMMENDED//\//\\}"
        echo "    set \"PATH=$(dirname "$rec_win") ;%PATH%\""
        echo "    cmake -S . -B build -DCMAKE_CUDA_COMPILER=\"$rec_win\" -DCMAKE_CUDA_ARCHITECTURES=${GPU_SM#sm_}"
    else
        echo "    export PATH=\"$(dirname "$RECOMMENDED"):\$PATH\""
        echo "    cmake -S . -B build-cuda -DCMAKE_CUDA_ARCHITECTURES=${GPU_SM#sm_}"
    fi
    if [[ "$WRITE_ENV" -eq 1 ]]; then
        {
            echo "# Generated by cuda_find.sh — source me before building."
            if [[ "$PLAT" == "windows" ]]; then
                echo "set \"PATH=$(dirname "${RECOMMENDED//\//\\}");%PATH%\""
            else
                echo "export PATH=\"$(dirname "$RECOMMENDED"):\$PATH\""
                echo "export CUDACXX=\"$RECOMMENDED\""
            fi
            echo "export CMAKE_CUDA_ARCHITECTURES=${GPU_SM#sm_}"
        } > cuda_env.sh
        echo
        echo "cuda_find: wrote cuda_env.sh (source it before building)."
    fi
    exit 0
else
    echo "cuda_find: RESULT: ${#TK_PATH[@]} toolkit(s) found, but none supports ${GPU_SM:-<unknown arch>}."
    echo "cuda_find:        upgrade the CUDA toolkit to >= $(min_cuda_for_sm "${GPU_SM:-sm_120}") for this GPU."
    exit 1
fi
