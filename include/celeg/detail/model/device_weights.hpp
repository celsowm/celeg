#pragma once

// Ownership of device-resident weight storage.
//
// `LinearWeight` is a non-owning view; `DeviceWeight` is the allocation that
// backs one. Kept apart from `linear_weights.hpp` so that consumers of the
// view (kernels, dispatchers, layer bindings) do not pull in `DeviceBuffer`
// and the CUDA runtime allocation machinery.

#include "celeg/detail/model/linear_weights.hpp"
#include "celeg/backend/cuda/utils.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace celeg {

struct DeviceWeight {
    DeviceBuffer<__nv_bfloat16> bf16_storage;
    DeviceBuffer<int8_t> int8_storage;
    DeviceBuffer<uint8_t> int4_storage;
    DeviceBuffer<uint8_t> gguf_expert_storage;
    std::vector<DeviceBuffer<uint8_t>> gguf_segment_storage;
    DeviceBuffer<float> scales_storage;
    std::vector<int64_t> shape;
    LinearWeight linear;
};

using WeightMap = std::unordered_map<std::string, DeviceWeight>;

} // namespace celeg
