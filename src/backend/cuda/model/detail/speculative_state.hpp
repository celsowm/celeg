#pragma once

#include "backend/cuda/runtime_types.hpp"
#include "backend/cuda/utils.cuh"

#include <array>
#include <cstdint>
#include <vector>

namespace celeg {

struct CudaSpeculativeLayerState {
    DeviceBuffer<__nv_bfloat16> conv_state;
    DeviceBuffer<__nv_bfloat16> ssm_state;
    DeviceBuffer<__nv_bfloat16> recurrent_state;
};

struct CudaSpeculativeState {
    bool valid = false;
    int position = 0;
    std::array<int32_t, 3> next_rope_position{0, 0, 0};
    SessionPhase phase = SessionPhase::Empty;
    bool active_segmented_attention = false;
    bool mtp_candidate_ready = false;
    RuntimeMetrics metrics;
    DeviceBuffer<uint8_t> seen_tokens;
    DeviceBuffer<__nv_bfloat16> logits;
    DeviceBuffer<__nv_bfloat16> mtp_logits;
    DeviceBuffer<int32_t> mtp_candidate;
    DeviceBuffer<int32_t> mtp_target_candidate;
    DeviceBuffer<uint64_t> rng_state;
    DeviceBuffer<int32_t> position_device;
    DeviceBuffer<int32_t> mrope_position_device;
    std::vector<CudaSpeculativeLayerState> layers;
    std::vector<CudaSpeculativeLayerState> mtp_layers;

    void discard() noexcept { valid = false; }
};

}
