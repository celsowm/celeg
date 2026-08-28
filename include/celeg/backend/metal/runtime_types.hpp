#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace celeg {

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
    int32_t weight_mode = 0;
    int32_t kv_cache_mode = 0;
    int32_t storage_mode = 0;
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

}
