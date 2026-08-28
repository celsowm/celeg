# Celeg C API

The public C ABI is declared in `include/celeg/api.h`. It is intended for C,
Rust, Zig, Node native addons, and other FFI consumers. This is a breaking
rename: the exported `celeg_*` symbols replace the former project and runtime
names.

## Initialization

Every options structure must be initialized before use. The `struct_size`
field supports size validation at the ABI boundary.

```c
#include "celeg/api.h"

celeg_cpu_model_options model;
celeg_cpu_model_options_init(&model);

celeg_cpu_backend_options backend;
celeg_cpu_backend_options_init(&backend);

celeg_engine_options engine;
celeg_engine_options_init(&engine, "cpu", &backend, sizeof(backend));

celeg_request_options request;
celeg_request_options_init(&request);
```

## Single-model API

`celeg_model_create` constructs a CPU model from a checkpoint path. Use
`celeg_model_prefill`, `celeg_model_decode`, and
`celeg_model_copy_logits` for direct inference. Metrics and errors are
available through `celeg_model_get_metrics` and
`celeg_model_last_error`.

```c
celeg_model* model = celeg_model_create("model.safetensors", &model_options);
if (!model) {
    /* celeg_model_last_error(NULL) contains the global error. */
}
celeg_model_destroy(model);
```

## Engine API

`celeg_engine` provides request submission, polling, cancellation, status
inspection, and explicit stepping. Creating a `celeg_engine` starts its
selected scheduler; destroying it stops the scheduler. The backend is selected
by the backend ID, and backend-owned options are passed as an opaque payload.

```c
celeg_engine* engine = celeg_engine_create("model.safetensors", &engine_options);
celeg_request_id request_id = 0;
const int32_t prompt[] = {1, 2, 3};

celeg_engine_submit(engine, prompt, 3, &request, &request_id);
int progressed = 0;
celeg_engine_step(engine, &progressed);
celeg_engine_destroy(engine);
```

## Metal backend

On Apple Silicon, select `metal` through `celeg_engine_options` and initialize
the backend payload with `celeg_metal_backend_options_init`. The Metal options
include the KV page size, request limits, and prefix-cache capacity. The
currently validated cached fixtures are LFM2.5-350M Safetensors, its Q4_K_M
GGUF variant, and a one-token LFM2.5-8B-A1B MoE smoke with demand-loaded
experts.

```c
celeg_metal_backend_options metal;
celeg_metal_backend_options_init(&metal);
metal.engine.max_active_requests = 2;
metal.engine.kv_page_tokens = 16;
metal.engine.prefix_cache = 1;

celeg_engine_options engine;
celeg_engine_options_init(&engine, "metal", &metal, sizeof(metal));
```

## Tokenizer API

`celeg_tokenizer_create`, `celeg_tokenizer_encode`, and
`celeg_tokenizer_decode` expose tokenizer operations with caller-provided
buffers. The `required` output reports the required element count when a
buffer is absent or too small.

## ABI conventions

- `celeg_status` reports success, invalid arguments, runtime errors, buffer
  sizing failures, missing resources, and unavailable backends.
- `celeg_backend_capabilities` reports the capabilities of a compiled backend.
- Handles are opaque and must be destroyed with their matching destroy
  function.
- The API uses the upstream LFM2/LFM2.5 checkpoint formats; renaming Celeg
  does not change model IDs, repository IDs, or checkpoint serialization.
