# Persistent sessions

A session stores the mutable inference state, not model weights:

- current token position;
- RNG state;
- repetition-penalty seen-token bitmap;
- current logits;
- K/V cache for all six attention layers;
- recurrent state for all eight short-convolution layers.

```bash
# Create and save after generation
./build/lfm25-run \
  --model ./model/LFM2.5-230M \
  --prompt "Explique CUDA." \
  --max-new-tokens 32 \
  --save-session cuda.session

# Continue later without re-prefilling the original prompt
./build/lfm25-run \
  --model ./model/LFM2.5-230M \
  --load-session cuda.session \
  --max-new-tokens 32
```

The same feature is available through `lfm25_save_session` and
`lfm25_load_session`.

## Compatibility checks

Loading rejects a session when:

- the magic or format version is unsupported;
- BF16 and INT8 KV modes differ;
- the saved position exceeds the current model context;
- compiled layer/head/vocabulary dimensions differ;
- the file is truncated or has trailing data.

Weight mode does not need to match, because the session stores already-computed
mutable state. Switching weight mode can nevertheless change continuation
logits after the first restored decode step, so matching weight and numerical
options is recommended for reproducibility.

## Size

Session size is proportional to the used context, not `max_context`. INT8 KV
sessions include one FP32 scale per token and KV head for K and V.

Sessions are local binary artifacts. The format is versioned but is not intended
as a portable interchange format across different architectures or model
families.
