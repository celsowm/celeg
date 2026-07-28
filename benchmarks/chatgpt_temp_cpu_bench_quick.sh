#!/usr/bin/env bash
set -euo pipefail
sed \
  -e 's/for t in 1 2 4; do/for t in 2; do/' \
  -e 's/-p 512 -n 128 -r 5/-p 128 -n 32 -r 2/g' \
  -e 's/ 512 256 32 bf16/ 128 128 32 bf16/g' \
  -e 's/pp512/pp128/g' \
  -e 's/tg128/tg32/g' \
  benchmarks/chatgpt_temp_cpu_bench.sh > /tmp/chatgpt_quick_bench.sh
bash /tmp/chatgpt_quick_bench.sh
