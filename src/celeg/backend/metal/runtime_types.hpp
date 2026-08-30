#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace celeg {

enum class MetalWeightMode : std::int32_t {
    Bf16 = 0,
};

enum class MetalKvCacheMode : std::int32_t {
    Bf16 = 0,
};

enum class MetalStorageMode : std::int32_t {
    Shared = 0,
    Private = 1,
};

struct MetalCapabilities {
    std::string device_name;
    std::string registry_id;
    std::size_t max_buffer_length = 0;
    std::size_t max_threads_per_threadgroup = 0;
    bool apple_gpu_family_8 = false;
    bool apple_gpu_family_9 = false;
    bool apple_gpu_family_10 = false;
    bool supports_shared_storage = true;
    bool supports_private_storage = true;
    bool runtime_shader_compilation = false;

    std::string summary() const;
};

struct MetalModelOptions {
    MetalWeightMode weight_mode = MetalWeightMode::Bf16;
    MetalKvCacheMode kv_cache_mode = MetalKvCacheMode::Bf16;
    int32_t storage_mode = static_cast<int32_t>(MetalStorageMode::Shared);
    int32_t kv_page_tokens = 16;
};

struct MetalEngineOptions {
    int32_t max_active_requests = 1;
    int32_t max_batched_tokens = 256;
    int32_t prefill_chunk_tokens = 256;
    int32_t kv_page_tokens = 16;
    bool prefix_cache = true;
    std::size_t prefix_cache_max_entries = 32;
};

struct MetalExecutionMetrics {
    double command_encoding_ms = 0.0;
    double command_wait_ms = 0.0;
    double gpu_execution_ms = 0.0;
    uint64_t command_buffers = 0;
    uint64_t dispatches = 0;
    std::size_t resident_weight_bytes = 0;
    std::size_t resident_state_bytes = 0;

    std::size_t resident_bytes() const {
        return resident_weight_bytes + resident_state_bytes;
    }
};

}
