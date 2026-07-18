# Parity workflow

Use BF16 strict mode, classic cuBLAS, graph disabled and fast math disabled for
the initial reference comparison. INT8, INT4 and segmented/automatic attention are optimization
modes and should be evaluated only after the BF16 baseline is understood.

## 1. Compare tokenizer IDs

```bash
./build/lfm25-run \
  --model ./model/LFM2.5-230M \
  --raw \
  --prompt '<|startoftext|><|im_start|>user
Explique CUDA.<|im_end|>
<|im_start|>assistant
' \
  --tokens-only
```

## 2. Compare batched and legacy C++ prefill

```bash
./build/lfm25-run --model ./model/LFM2.5-230M \
  --prompt 'Explique CUDA.' --max-new-tokens 0 --no-cuda-graph \
  --weight-mode bf16 --dump-logits batched.f32

./build/lfm25-run --model ./model/LFM2.5-230M \
  --prompt 'Explique CUDA.' --max-new-tokens 0 --no-cuda-graph \
  --weight-mode bf16 --legacy-prefill --dump-logits legacy.f32

./build/lfm25-compare-logits legacy.f32 batched.f32
```

## 3. Compare the official BF16 reference

Export the final-position logits from the official BF16 model as exactly 65,536
contiguous little-endian float32 values, then run:

```bash
./build/lfm25-compare-logits reference.f32 batched.f32
```

## 4. Measure quantization drift

```bash
MODEL=./model/LFM2.5-230M \
PROMPT='Explique CUDA.' \
./scripts/compare_weight_modes.sh
```

For W8A16 and W4A16, track maximum/mean absolute error, RMSE, top-1 agreement and top-10
overlap. Also compare complete greedy or seeded sequences over a prompt suite;
one logit vector is not sufficient to characterize quality.

## Recommended progression

1. token IDs;
2. one-token and short-prompt BF16 logits;
3. batched versus legacy BF16 prefill;
4. official versus strict BF16;
5. graph versus non-graph BF16 decode;
6. each BF16 fast/fused option independently;
7. BF16 versus W8A16 logits and generated sequences;
8. BF16 versus W4A16 logits and generated sequences;
9. online versus segmented attention at several context lengths;
10. automatic attention around the configured threshold.
