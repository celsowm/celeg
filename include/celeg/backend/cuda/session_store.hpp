#pragma once

#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "celeg/backend/cuda/utils.cuh"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace celeg {

class SessionStore {
public:
    struct Header {
        std::array<char, 8> magic{{'C', 'E', 'L', 'E', 'G', 'S', 'S', '5'}};
        uint32_t version = 5;
        uint32_t kv_cache_mode = 0;
        int32_t position = 0;
        int32_t max_context = 0;
        int32_t layers = 0;
        int32_t vocab = 0;
        int32_t attention_layers = 0;
        uint64_t rng_state = 0;
        uint64_t model_identity_hash = 0;
    };

    static constexpr std::array<char, 8> kMagic{
        {'C', 'E', 'L', 'E', 'G', 'S', 'S', '5'}};
    static constexpr uint32_t kVersion = 5;

    struct SessionState {
        const ExecutionTopology& shape;
        const CompiledModelProgram& program;
        const CheckpointDimensions& dims;
        int max_context = 0;
        int position = 0;
        KvCacheMode kv_cache_mode = KvCacheMode::Bf16;
        std::string model_identity;
        cudaStream_t stream = nullptr;
        DeviceBuffer<uint8_t>* seen_tokens = nullptr;
        DeviceBuffer<__nv_bfloat16>* logits = nullptr;
        DeviceBuffer<uint64_t>* rng_state = nullptr;
        struct LayerBuffers {
            bool is_attention = false;
            bool is_mamba = false;
            bool is_gated_delta = false;
            bool owns_kv_cache = false;
            __nv_bfloat16* key_cache_bf16 = nullptr;
            __nv_bfloat16* value_cache_bf16 = nullptr;
            int8_t* key_cache_int8 = nullptr;
            int8_t* value_cache_int8 = nullptr;
            float* key_cache_scales = nullptr;
            float* value_cache_scales = nullptr;
            __nv_bfloat16* conv_state = nullptr;
            size_t conv_state_elements = 0;
            __nv_bfloat16* ssm_state = nullptr;
            size_t ssm_state_elements = 0;
            __nv_bfloat16* recurrent_state = nullptr;
            size_t recurrent_state_elements = 0;
        };
        std::vector<LayerBuffers> layer_buffers;
    };

    static void save(const std::string& path, SessionState& state);

    static void load(const std::string& path, SessionState& state);

    struct PrefixSnapshot {
        int position = 0;
        std::vector<uint8_t> seen_tokens;
        std::vector<uint16_t> logits_bf16;
        std::vector<uint16_t> conv_state_bf16;
        std::vector<uint16_t> mamba_state_bf16;
        std::vector<uint16_t> gated_delta_state_bf16;
    };

    static PrefixSnapshot export_prefix(const SessionState& state);

    static void restore_prefix(const PrefixSnapshot& snapshot,
                               SessionState& state,
                               uint64_t request_seed);
};

}
