#!/usr/bin/env bash
set -euo pipefail
mkdir -p benchmark-artifacts/results
{
  echo "sha=$GITHUB_SHA"
  echo "nproc=$(nproc)"
  uname -a
  lscpu
  cmake --version
  g++ --version
} | tee benchmark-artifacts/system.txt

python - <<'PY' | tee benchmark-artifacts/ephemeral-restoration.log
from pathlib import Path
import subprocess

deleted = subprocess.check_output([
    "git", "log", "--all", "--diff-filter=D", "--format=%H", "--", "src/config.cpp"
], text=True).splitlines()[0]
legacy = subprocess.check_output(["git", "rev-parse", f"{deleted}^"], text=True).strip()
print(f"deletion_commit={deleted}\nrestoration_source={legacy}")
files = {
    "include/lfm/config.hpp": "include/lfm/model/config/config.hpp",
    "include/lfm/model_shape.hpp": "include/lfm/model/config/shape.hpp",
    "include/lfm/model_variant.hpp": "include/lfm/model/config/variant.hpp",
    "include/lfm/runtime_types.hpp": "include/lfm/model/execution/runtime_types.hpp",
    "include/lfm/execution_plan.hpp": "include/lfm/model/execution/plan.hpp",
    "include/lfm/quantization.hpp": "include/lfm/model/weights/quantization.hpp",
    "include/lfm/reference.hpp": "include/lfm/model/reference.hpp",
    "src/config.cpp": "src/model/config/config.cpp",
    "src/model_shape.cpp": "src/model/config/shape.cpp",
    "src/model_variant.cpp": "src/model/config/variant.cpp",
    "src/runtime_types.cpp": "src/model/execution/runtime_types.cpp",
    "src/execution_plan.cpp": "src/model/execution/plan.cpp",
    "src/quantization.cpp": "src/model/weights/quantization.cpp",
    "src/reference.cpp": "src/model/reference.cpp",
}
repl = {
    '"lfm/config.hpp"': '"lfm/model/config/config.hpp"',
    '"lfm/model_shape.hpp"': '"lfm/model/config/shape.hpp"',
    '"lfm/model_variant.hpp"': '"lfm/model/config/variant.hpp"',
    '"lfm/runtime_types.hpp"': '"lfm/model/execution/runtime_types.hpp"',
    '"lfm/execution_plan.hpp"': '"lfm/model/execution/plan.hpp"',
    '"lfm/quantization.hpp"': '"lfm/model/weights/quantization.hpp"',
    '"lfm/reference.hpp"': '"lfm/model/reference.hpp"',
    '"lfm/json.hpp"': '"lfm/checkpoint/formats/json.hpp"',
    '"lfm/policy.hpp"': '"lfm/runtime/concurrency/policy_types.hpp"',
}
for old, new in files.items():
    target = Path(new)
    if target.exists() and target.stat().st_size:
        print(f"kept {new}")
        continue
    text = subprocess.check_output(["git", "show", f"{legacy}:{old}"]).decode("utf-8")
    for before, after in repl.items():
        text = text.replace(before, after)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"restored {old} -> {new}")
PY

set -o pipefail
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLFM_ENABLE_CUDA=OFF \
  -DLFM_ENABLE_SERVE=OFF -DLFM_BUILD_TESTS=OFF \
  2>&1 | tee benchmark-artifacts/configure-baseline.log
cmake --build build-cpu --target lfm25-bench lfm25-cpu-prefill-benchmark \
  -j "$(nproc)" 2>&1 | tee benchmark-artifacts/build-baseline.log

cmake -S . -B build-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLFM_ENABLE_CUDA=OFF \
  -DLFM_ENABLE_SERVE=OFF -DLFM_BUILD_TESTS=OFF \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -march=native" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native" \
  2>&1 | tee benchmark-artifacts/configure-native.log
cmake --build build-native --target lfm25-bench lfm25-cpu-prefill-benchmark \
  -j "$(nproc)" 2>&1 | tee benchmark-artifacts/build-native.log

python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
lfm = Path(snapshot_download("LiquidAI/LFM2.5-230M", revision=os.environ["LFM_REVISION"]))
q40 = Path(hf_hub_download("LiquidAI/LFM2.5-230M-GGUF", "LFM2.5-230M-Q4_0.gguf"))
q4km = Path(hf_hub_download("LiquidAI/LFM2.5-230M-GGUF", "LFM2.5-230M-Q4_K_M.gguf"))
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as f:
    f.write(f"LFM_MODEL_DIR={lfm}\n")
    f.write(f"LFM_MODEL_FILE={lfm / 'model.safetensors'}\n")
    f.write(f"GGUF_Q40={q40}\n")
    f.write(f"GGUF_Q4KM={q4km}\n")
Path("benchmark-artifacts/model-paths.txt").write_text(
    f"LFM_MODEL_DIR={lfm}\nLFM_MODEL_FILE={lfm / 'model.safetensors'}\nGGUF_Q40={q40}\nGGUF_Q4KM={q4km}\n"
)
PY
source <(cat benchmark-artifacts/model-paths.txt)

git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git .externals/llama.cpp
git -C .externals/llama.cpp checkout --detach "$LLAMA_REVISION"
cmake -S .externals/llama.cpp -B .externals/llama.cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF \
  2>&1 | tee benchmark-artifacts/configure-llama.log
cmake --build .externals/llama.cpp/build --target llama-bench -j "$(nproc)" \
  2>&1 | tee benchmark-artifacts/build-llama.log

LFM=$(find build-cpu -type f -name lfm25-bench -perm -111 | head -1)
LFMN=$(find build-native -type f -name lfm25-bench -perm -111 | head -1)
PF=$(find build-cpu -type f -name lfm25-cpu-prefill-benchmark -perm -111 | head -1)
PFN=$(find build-native -type f -name lfm25-cpu-prefill-benchmark -perm -111 | head -1)
LLAMA=$(find .externals/llama.cpp/build -type f -name llama-bench -perm -111 | head -1)

for t in 1 2 4; do
  "$LFM" --model "$LFM_MODEL_DIR" -p 512 -n 128 -r 5 --warmup 1 -t "$t" --cpu-q4-group 32 -o json > "benchmark-artifacts/results/lfm-baseline-t$t.json"
  "$LFMN" --model "$LFM_MODEL_DIR" -p 512 -n 128 -r 5 --warmup 1 -t "$t" --cpu-q4-group 32 -o json > "benchmark-artifacts/results/lfm-native-t$t.json"
  "$LLAMA" -m "$GGUF_Q40" -p 512 -n 128 -r 5 -o json -t "$t" -b 256 -ub 256 > "benchmark-artifacts/results/llama-q4_0-t$t.json"
  "$LLAMA" -m "$GGUF_Q4KM" -p 512 -n 128 -r 5 -o json -t "$t" -b 256 -ub 256 > "benchmark-artifacts/results/llama-q4_k_m-t$t.json"
  "$PF" "$LFM_MODEL_FILE" 512 256 32 bf16 auto "$t" > "benchmark-artifacts/results/lfm-profile-t$t.txt"
  "$PFN" "$LFM_MODEL_FILE" 512 256 32 bf16 auto "$t" > "benchmark-artifacts/results/lfm-native-profile-t$t.txt"
done

python - <<'PY'
import json, re
from pathlib import Path
d = Path("benchmark-artifacts/results")
rows = []
for p in d.glob("*.json"):
    data = json.loads(p.read_text())
    pp = next(r for r in data if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0)
    tg = next(r for r in data if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0)
    t = int(re.search(r"-t(\d+)\.json", p.name).group(1))
    rows.append((t, p.name.rsplit("-t", 1)[0], float(pp["avg_ts"]), float(tg["avg_ts"])))
lines = ["# CPU benchmark", "", "| threads | engine | pp512 | tg128 |", "|---:|---|---:|---:|"]
lines += [f"| {t} | {e} | {pp:.2f} | {tg:.2f} |" for t, e, pp, tg in sorted(rows)]
for p in sorted(d.glob("*.txt")):
    lines += ["", f"## {p.name}", "```", p.read_text().strip(), "```"]
Path("benchmark-artifacts/summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
