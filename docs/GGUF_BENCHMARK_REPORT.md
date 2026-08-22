# GGUF benchmark sweep: celeg vs llama.cpp

Every `.gguf` file present in the local Hugging Face hub cache (`*imatrix*` calibration files excluded) was benchmarked against upstream llama.cpp, on CPU and, where available, GPU. Tokens/sec are the `avg_ts` llama-bench/celeg-bench report over 5 timed repetitions plus one discarded warmup, 512 synthetic prefill tokens and 128 decode steps, batch/ubatch size 256, BF16 KV cache on both engines. This is an exploratory survey across every cached quant, not the strict same-file release gate in `benchmarks/compare_llama.py`. Each engine is recorded independently: a file only one of them can load still contributes that engine's numbers, with the ratio columns left blank.

## This machine

```
CPU:    Intel(R) Core(TM) i9-14900KF
Threads used: 32
GPU:    NVIDIA GeForce RTX 5090
OS:     Linux-7.0.0-29-generic-x86_64-with-glibc2.39
```

## Methodology notes

- On CPU, celeg executes GGUF blocks in place with native dot kernels for every quantization the type registry marks `cpu_native_dot`; anything else is dequantized on load and repacked into groupwise Q4. See `docs/QUANTIZATION_SUPPORT_MATRIX.md`.
- On GPU, celeg ran with `--weight-mode auto`. **`auto` requantizes every GGUF quantization to int8 on load**, so under `auto` the GPU columns measure an int8 weight path and are largely insensitive to the source quantization -- only `native` keeps Q4_K/Q6_K blocks packed on the device.
- llama.cpp runs `-ngl 99` on GPU rows (every layer offloaded) and `-ngl 0` on CPU rows.
- celeg commit: `d79e3d51f5e3c409a851d175813499c25754cfed`
- llama.cpp commit: `dc72703fc69698b1ea68ece8d2dd8a96e6a4e1fe`

## Load failures

| backend | engine | reason | files |
|---|---|---|---|
| CPU | llama.cpp | load-failed | `lizzy-7b-q4_k_m.gguf`, `lizzy-7b-q5_k_m.gguf`, `lizzy-7b-q6_k.gguf`, `lizzy-7b-q8_0.gguf`, +1 more |
| GPU | llama.cpp | load-failed | `lizzy-7b-q4_k_m.gguf`, `lizzy-7b-q5_k_m.gguf`, `lizzy-7b-q6_k.gguf`, `lizzy-7b-q8_0.gguf`, +1 more |

## LiquidAI/LFM2.5-230M-GGUF

### CPU

| quant file | size (GB) | llama.cpp prefill | celeg prefill | prefill | llama.cpp decode | celeg decode | decode | status |
|---|---|---|---|---|---|---|---|---|
| `LFM2.5-230M-Q4_K_M.gguf` | 0.14 | 2429.7 | 1125.6 | 0.46x | 114.4 | 106.8 | 0.93x | ok |

### GPU

| quant file | size (GB) | llama.cpp prefill | celeg prefill | prefill | llama.cpp decode | celeg decode | decode | status |
|---|---|---|---|---|---|---|---|---|
| `LFM2.5-230M-Q4_K_M.gguf` | 0.14 | 47304.0 | 204188.7 | 4.32x | 2011.2 | 1158.3 | 0.58x | ok |

## LiquidAI/LFM2.5-350M-GGUF

### CPU

| quant file | size (GB) | llama.cpp prefill | celeg prefill | prefill | llama.cpp decode | celeg decode | decode | status |
|---|---|---|---|---|---|---|---|---|
| `LFM2.5-350M-BF16.gguf` | 0.66 | 1164.1 | 317.4 | 0.27x | 41.9 | 93.9 | 2.24x | ok |
| `LFM2.5-350M-F16.gguf` | 0.66 | 1034.1 | 318.1 | 0.31x | 42.4 | 94.5 | 2.23x | ok |
| `LFM2.5-350M-Q4_0.gguf` | 0.20 | 1808.6 | 331.3 | 0.18x | 89.1 | 87.6 | 0.98x | ok |
| `LFM2.5-350M-Q4_K_M.gguf` | 0.21 | 1516.4 | 613.4 | 0.40x | 85.5 | 86.5 | 1.01x | ok |
| `LFM2.5-350M-Q5_K_M.gguf` | 0.24 | 858.6 | 325.8 | 0.38x | 77.1 | 82.6 | 1.07x | ok |
| `LFM2.5-350M-Q6_K.gguf` | 0.27 | 1007.6 | 399.9 | 0.40x | 72.8 | 79.9 | 1.10x | ok |
| `LFM2.5-350M-Q8_0.gguf` | 0.35 | 1013.4 | 366.8 | 0.36x | 64.4 | 70.4 | 1.09x | ok |

### GPU

| quant file | size (GB) | llama.cpp prefill | celeg prefill | prefill | llama.cpp decode | celeg decode | decode | status |
|---|---|---|---|---|---|---|---|---|
| `LFM2.5-350M-BF16.gguf` | 0.66 | 11857.1 | 14231.9 | 1.20x | 1275.3 | 1060.0 | 0.83x | ok |
| `LFM2.5-350M-F16.gguf` | 0.66 | 22415.6 | 14520.2 | 0.65x | 1272.7 | 1051.4 | 0.83x | ok |
| `LFM2.5-350M-Q4_0.gguf` | 0.20 | 42308.9 | 151341.5 | 3.58x | 1832.0 | 1072.1 | 0.59x | ok |
| `LFM2.5-350M-Q4_K_M.gguf` | 0.21 | 39094.5 | 151744.5 | 3.88x | 1639.8 | 1007.5 | 0.61x | ok |
| `LFM2.5-350M-Q5_K_M.gguf` | 0.24 | 38649.6 | 151235.8 | 3.91x | 1619.1 | 1063.5 | 0.66x | ok |
| `LFM2.5-350M-Q6_K.gguf` | 0.27 | 35540.3 | 150238.3 | 4.23x | 1649.3 | 1007.7 | 0.61x | ok |
| `LFM2.5-350M-Q8_0.gguf` | 0.35 | 42611.3 | 154069.1 | 3.62x | 1569.7 | 1071.9 | 0.68x | ok |

## bartowski/Nanbeige_Nanbeige4.2-3B-GGUF

### CPU

| quant file | size (GB) | llama.cpp prefill | celeg prefill | prefill | llama.cpp decode | celeg decode | decode | status |
|---|---|---|---|---|---|---|---|---|
| `Nanbeige_Nanbeige4.2-3B-IQ2_M.gguf` | 1.74 | 28.2 | 19.3 | 0.68x | 11.0 | 7.8 | 0.72x | ok |
| `Nanbeige_Nanbeige4.2-3B-IQ3_M.gguf` | 2.02 | 19.9 | 19.0 | 0.96x | 9.4 | 7.5 | 0.79x | ok |
| `Nanbeige_Nanbeige4.2-3B-IQ3_XS.gguf` | 1.89 | 19.4 | 17.3 | 0.89x | 10.0 | 7.6 | 0.76x | ok |
| `Nanbeige_Nanbeige4.2-3B-IQ3_XXS.gguf` | 1.76 | 24.0 | 17.2 | 0.72x | 10.6 | 7.7 | 0.73x | ok |
| `Nanbeige_Nanbeige4.2-3B-IQ4_NL.gguf` | 2.37 | 79.0 | 17.0 | 0.22x | 8.7 | 7.5 | 0.86x | ok |
| `Nanbeige_Nanbeige4.2-3B-IQ4_XS.gguf` | 2.27 | 43.4 | 17.1 | 0.39x | 8.9 | 7.5 | 0.84x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q2_K.gguf` | 1.72 | 52.3 | 14.1 | 0.27x | 11.2 | 8.8 | 0.79x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q2_K_L.gguf` | 2.19 | 52.3 | 14.1 | 0.27x | 10.8 | 8.6 | 0.79x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q3_K_L.gguf` | 2.17 | 43.7 | 14.3 | 0.33x | 9.1 | 7.7 | 0.85x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q3_K_M.gguf` | 2.09 | 48.9 | 16.0 | 0.33x | 9.5 | 8.0 | 0.84x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q3_K_S.gguf` | 1.93 | 40.7 | 12.3 | 0.30x | 10.1 | 8.2 | 0.82x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q3_K_XL.gguf` | 2.59 | 43.4 | 14.3 | 0.33x | 8.8 | 7.5 | 0.85x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q4_0.gguf` | 2.37 | 75.4 | 16.8 | 0.22x | 8.7 | 7.8 | 0.89x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q4_1.gguf` | 2.56 | 41.6 | 14.9 | 0.36x | 7.9 | 7.3 | 0.92x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q4_K_L.gguf` | 2.85 | 61.4 | 26.6 | 0.43x | 7.9 | 7.2 | 0.91x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q4_K_M.gguf` | 2.50 | 60.9 | 26.6 | 0.44x | 8.1 | 7.4 | 0.90x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q4_K_S.gguf` | 2.38 | 70.5 | 31.9 | 0.45x | 8.6 | 7.7 | 0.90x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q5_K_L.gguf` | 3.16 | 42.6 | 17.0 | 0.40x | 7.1 | 6.5 | 0.92x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q5_K_M.gguf` | 2.87 | 42.7 | 17.0 | 0.40x | 7.2 | 6.6 | 0.92x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q5_K_S.gguf` | 2.76 | 40.9 | 16.2 | 0.40x | 7.5 | 6.8 | 0.91x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q6_K.gguf` | 3.35 | 47.7 | 19.7 | 0.41x | 6.2 | 5.8 | 0.93x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q6_K_L.gguf` | 3.58 | 47.6 | 19.7 | 0.41x | 6.1 | 5.7 | 0.93x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q8_0.gguf` | 4.13 | 48.2 | 18.4 | 0.38x | 5.2 | 4.9 | 0.93x | ok |
| `Nanbeige_Nanbeige4.2-3B-bf16.gguf` | 7.77 | 54.2 | 17.1 | 0.32x | 2.9 | 7.9 | 2.74x | ok |

### GPU

| quant file | size (GB) | llama.cpp prefill | celeg prefill | prefill | llama.cpp decode | celeg decode | decode | status |
|---|---|---|---|---|---|---|---|---|
| `Nanbeige_Nanbeige4.2-3B-IQ2_M.gguf` | 1.74 | 2005.1 | 3141.8 | 1.57x | 295.2 | 134.3 | 0.46x | ok |
| `Nanbeige_Nanbeige4.2-3B-IQ3_M.gguf` | 2.02 | 2024.4 | 3201.2 | 1.58x | 289.2 | 134.6 | 0.47x | ok |
| `Nanbeige_Nanbeige4.2-3B-IQ3_XS.gguf` | 1.89 | 2050.8 | 3148.5 | 1.54x | 300.3 | 134.5 | 0.45x | ok |
| `Nanbeige_Nanbeige4.2-3B-IQ3_XXS.gguf` | 1.76 | 2060.5 | 3206.6 | 1.56x | 318.3 | 134.7 | 0.42x | ok |
| `Nanbeige_Nanbeige4.2-3B-IQ4_NL.gguf` | 2.37 | 2105.2 | 3176.6 | 1.51x | 272.2 | 134.6 | 0.49x | ok |
| `Nanbeige_Nanbeige4.2-3B-IQ4_XS.gguf` | 2.27 | 2114.6 | 3108.7 | 1.47x | 273.6 | 134.4 | 0.49x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q2_K.gguf` | 1.72 | 1958.3 | 3152.4 | 1.61x | 297.3 | 134.5 | 0.45x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q2_K_L.gguf` | 2.19 | 1956.9 | 3170.4 | 1.62x | 288.9 | 134.2 | 0.46x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q3_K_L.gguf` | 2.17 | 2044.3 | 3220.2 | 1.58x | 266.1 | 134.7 | 0.51x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q3_K_M.gguf` | 2.09 | 2051.4 | 3133.1 | 1.53x | 272.8 | 134.1 | 0.49x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q3_K_S.gguf` | 1.93 | 1991.1 | 3169.2 | 1.59x | 277.2 | 134.2 | 0.48x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q3_K_XL.gguf` | 2.59 | 2037.0 | 3217.9 | 1.58x | 260.2 | 134.6 | 0.52x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q4_0.gguf` | 2.37 | 2123.6 | 3104.3 | 1.46x | 275.7 | 134.3 | 0.49x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q4_1.gguf` | 2.56 | 2090.7 | 3113.5 | 1.49x | 260.8 | 134.4 | 0.52x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q4_K_L.gguf` | 2.85 | 2048.1 | 3156.8 | 1.54x | 250.8 | 132.4 | 0.53x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q4_K_M.gguf` | 2.50 | 2063.7 | 3135.2 | 1.52x | 255.7 | 134.4 | 0.53x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q4_K_S.gguf` | 2.38 | 2085.0 | 3088.1 | 1.48x | 265.8 | 130.4 | 0.49x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q5_K_L.gguf` | 3.16 | 2011.3 | 3127.5 | 1.55x | 229.8 | 134.5 | 0.59x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q5_K_M.gguf` | 2.87 | 2028.9 | 3209.8 | 1.58x | 233.8 | 134.7 | 0.58x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q5_K_S.gguf` | 2.76 | 2064.8 | 3104.3 | 1.50x | 240.6 | 134.4 | 0.56x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q6_K.gguf` | 3.35 | 2015.8 | 3213.8 | 1.59x | 206.2 | 130.5 | 0.63x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q6_K_L.gguf` | 3.58 | 2017.3 | 3186.5 | 1.58x | 203.3 | 130.7 | 0.64x | ok |
| `Nanbeige_Nanbeige4.2-3B-Q8_0.gguf` | 4.13 | 2099.8 | 3086.3 | 1.47x | 181.3 | 134.4 | 0.74x | ok |
| `Nanbeige_Nanbeige4.2-3B-bf16.gguf` | 7.77 | 958.8 | 687.9 | 0.72x | 109.9 | 133.3 | 1.21x | ok |

## flwrlabs/Lizzy-7B-GGUF

### CPU

| quant file | size (GB) | llama.cpp prefill | celeg prefill | prefill | llama.cpp decode | celeg decode | decode | status |
|---|---|---|---|---|---|---|---|---|
| `lizzy-7b-q4_k_m.gguf` | 4.16 | load-failed | 31.5 | -- | load-failed | 6.9 | -- | partial (llama.cpp) |
| `lizzy-7b-q5_k_m.gguf` | 4.85 | load-failed | 15.9 | -- | load-failed | 6.1 | -- | partial (llama.cpp) |
| `lizzy-7b-q6_k.gguf` | 5.58 | load-failed | 19.3 | -- | load-failed | 5.4 | -- | partial (llama.cpp) |
| `lizzy-7b-q8_0.gguf` | 7.23 | load-failed | 18.0 | -- | load-failed | 4.5 | -- | partial (llama.cpp) |
| `lizzy-final.gguf` | 13.60 | load-failed | 16.8 | -- | load-failed | 7.2 | -- | partial (llama.cpp) |

### GPU

| quant file | size (GB) | llama.cpp prefill | celeg prefill | prefill | llama.cpp decode | celeg decode | decode | status |
|---|---|---|---|---|---|---|---|---|
| `lizzy-7b-q4_k_m.gguf` | 4.16 | load-failed | 4508.4 | -- | load-failed | 137.4 | -- | partial (llama.cpp) |
| `lizzy-7b-q5_k_m.gguf` | 4.85 | load-failed | 4577.0 | -- | load-failed | 142.3 | -- | partial (llama.cpp) |
| `lizzy-7b-q6_k.gguf` | 5.58 | load-failed | 4487.7 | -- | load-failed | 137.3 | -- | partial (llama.cpp) |
| `lizzy-7b-q8_0.gguf` | 7.23 | load-failed | 4497.6 | -- | load-failed | 141.6 | -- | partial (llama.cpp) |
| `lizzy-final.gguf` | 13.60 | load-failed | 705.5 | -- | load-failed | 141.7 | -- | partial (llama.cpp) |

## openbmb/MiniCPM5-1B-GGUF

### CPU

| quant file | size (GB) | llama.cpp prefill | celeg prefill | prefill | llama.cpp decode | celeg decode | decode | status |
|---|---|---|---|---|---|---|---|---|
| `MiniCPM5-1B-F16.gguf` | 2.02 | 418.5 | 139.3 | 0.33x | 19.4 | 42.3 | 2.18x | ok |
| `MiniCPM5-1B-Q4_K_M.gguf` | 0.64 | 570.7 | 250.7 | 0.44x | 43.4 | 39.0 | 0.90x | ok |
| `MiniCPM5-1B-Q8_0.gguf` | 1.07 | 416.5 | 151.9 | 0.36x | 31.3 | 30.5 | 0.98x | ok |

### GPU

| quant file | size (GB) | llama.cpp prefill | celeg prefill | prefill | llama.cpp decode | celeg decode | decode | status |
|---|---|---|---|---|---|---|---|---|
| `MiniCPM5-1B-F16.gguf` | 2.02 | 7506.4 | 4690.5 | 0.62x | 611.5 | 425.3 | 0.70x | ok |
| `MiniCPM5-1B-Q4_K_M.gguf` | 0.64 | 11194.6 | 11869.5 | 1.06x | 918.3 | 406.3 | 0.44x | ok |
| `MiniCPM5-1B-Q8_0.gguf` | 1.07 | 11624.6 | 11891.2 | 1.02x | 801.9 | 422.4 | 0.53x | ok |

## Raw data

Per-run JSON lives under `benchmarks/results/sweep/` (gitignored); the full machine-readable summary for this run is `benchmarks/results/sweep/summary.json`.

