set(CELEG_METAL_BACKEND_SOURCES
    src/backend/metal/runtime/device.mm
    src/backend/metal/model/compiler.cpp
    src/backend/metal/model/resources.cpp
    src/backend/metal/model/lifecycle.cpp
    src/backend/metal/model/runtime/linear_bindings.cpp
)

set(CELEG_METAL_MODEL_SOURCES
    src/backend/metal/model/runtime/attention.mm
    src/backend/metal/model/runtime/execution.mm
    src/backend/metal/model/runtime/feed_forward.mm
    src/backend/metal/model/runtime/initialization.mm
    src/backend/metal/model/runtime/moe.mm
    src/backend/metal/model/runtime/pipelines.mm
    src/backend/metal/model/runtime/recurrent.mm
    src/backend/metal/model/runtime/resources.mm
    src/backend/metal/model/runtime/runtime.mm
    src/backend/metal/model/runtime/state.mm
)

set(CELEG_METAL_SHADER_SOURCES
    src/backend/metal/kernels/probe.metal
    src/backend/metal/kernels/inference.metal
    src/backend/metal/kernels/inference/common.metal
    src/backend/metal/kernels/inference/vector.metal
    src/backend/metal/kernels/inference/state.metal
    src/backend/metal/kernels/inference/projection.metal
    src/backend/metal/kernels/inference/batch.metal
    src/backend/metal/kernels/inference/pair.metal
    src/backend/metal/kernels/tensor.metal
)

set(CELEG_METAL_APP_SOURCES
    apps/benchmark/metal/inference.cpp
)
