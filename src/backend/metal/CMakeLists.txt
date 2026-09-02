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
    src/backend/metal/kernels/inference/kv_store.metal
    src/backend/metal/kernels/inference/feed_forward.metal
    src/backend/metal/kernels/inference/normalization.metal
    src/backend/metal/kernels/inference/convolution.metal
    src/backend/metal/kernels/inference/relative_bias.metal
    src/backend/metal/kernels/inference/position.metal
    src/backend/metal/kernels/inference/no_position.metal
    src/backend/metal/kernels/inference/qk_position.metal
    src/backend/metal/kernels/inference/qk_norm.metal
    src/backend/metal/kernels/inference/mrope_batch.metal
    src/backend/metal/kernels/inference/attention_output.metal
    src/backend/metal/kernels/inference/attention_gate.metal
    src/backend/metal/kernels/tensor.metal
    src/backend/metal/kernels/tensor_q4k_static_stage128.metal
    src/backend/metal/kernels/tensor_q4k_relaxed.metal
)

set(_CELEG_METAL_KV_STORE_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/kv_store.metal")
file(READ "${_CELEG_METAL_KV_STORE_SOURCE_FILE}"
    CELEG_METAL_KV_STORE_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_KV_STORE_SOURCE_FILE}")

set(_CELEG_METAL_FEED_FORWARD_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/feed_forward.metal")
file(READ "${_CELEG_METAL_FEED_FORWARD_SOURCE_FILE}"
    CELEG_METAL_FEED_FORWARD_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_FEED_FORWARD_SOURCE_FILE}")

set(_CELEG_METAL_NORMALIZATION_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/normalization.metal")
file(READ "${_CELEG_METAL_NORMALIZATION_SOURCE_FILE}"
    CELEG_METAL_NORMALIZATION_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_NORMALIZATION_SOURCE_FILE}")

set(_CELEG_METAL_CONVOLUTION_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/convolution.metal")
file(READ "${_CELEG_METAL_CONVOLUTION_SOURCE_FILE}"
    CELEG_METAL_CONVOLUTION_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_CONVOLUTION_SOURCE_FILE}")

set(_CELEG_METAL_RELATIVE_BIAS_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/relative_bias.metal")
file(READ "${_CELEG_METAL_RELATIVE_BIAS_SOURCE_FILE}"
    CELEG_METAL_RELATIVE_BIAS_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_RELATIVE_BIAS_SOURCE_FILE}")

set(_CELEG_METAL_POSITION_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/position.metal")
file(READ "${_CELEG_METAL_POSITION_SOURCE_FILE}"
    CELEG_METAL_POSITION_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_POSITION_SOURCE_FILE}")

set(_CELEG_METAL_NO_POSITION_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/no_position.metal")
file(READ "${_CELEG_METAL_NO_POSITION_SOURCE_FILE}"
    CELEG_METAL_NO_POSITION_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_NO_POSITION_SOURCE_FILE}")

set(_CELEG_METAL_QK_POSITION_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/qk_position.metal")
file(READ "${_CELEG_METAL_QK_POSITION_SOURCE_FILE}"
    CELEG_METAL_QK_POSITION_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_QK_POSITION_SOURCE_FILE}")

set(_CELEG_METAL_QK_NORM_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/qk_norm.metal")
file(READ "${_CELEG_METAL_QK_NORM_SOURCE_FILE}"
    CELEG_METAL_QK_NORM_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_QK_NORM_SOURCE_FILE}")

set(_CELEG_METAL_MROPE_BATCH_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/mrope_batch.metal")
file(READ "${_CELEG_METAL_MROPE_BATCH_SOURCE_FILE}"
    CELEG_METAL_MROPE_BATCH_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_MROPE_BATCH_SOURCE_FILE}")

set(_CELEG_METAL_ATTENTION_OUTPUT_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/attention_output.metal")
file(READ "${_CELEG_METAL_ATTENTION_OUTPUT_SOURCE_FILE}"
    CELEG_METAL_ATTENTION_OUTPUT_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_ATTENTION_OUTPUT_SOURCE_FILE}")

set(_CELEG_METAL_ATTENTION_GATE_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/inference/attention_gate.metal")
file(READ "${_CELEG_METAL_ATTENTION_GATE_SOURCE_FILE}"
    CELEG_METAL_ATTENTION_GATE_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_ATTENTION_GATE_SOURCE_FILE}")

set(_CELEG_METAL_TENSOR_Q4K_STATIC_STAGE128_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/tensor_q4k_static_stage128.metal")
file(READ "${_CELEG_METAL_TENSOR_Q4K_STATIC_STAGE128_SOURCE_FILE}"
    CELEG_METAL_TENSOR_Q4K_STATIC_STAGE128_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_TENSOR_Q4K_STATIC_STAGE128_SOURCE_FILE}")

set(_CELEG_METAL_TENSOR_Q4K_RELAXED_SOURCE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/kernels/tensor_q4k_relaxed.metal")
file(READ "${_CELEG_METAL_TENSOR_Q4K_RELAXED_SOURCE_FILE}"
    CELEG_METAL_TENSOR_Q4K_RELAXED_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_CELEG_METAL_TENSOR_Q4K_RELAXED_SOURCE_FILE}")

set(CELEG_METAL_APP_SOURCES
    apps/benchmark/metal/inference.cpp
)
