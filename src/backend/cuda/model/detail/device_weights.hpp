#pragma once


#include "linear_weights.hpp"
#include "backend/cuda/utils.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace celeg {

struct DeviceWeight {
    DeviceWeight() : DeviceWeight(cuda_current_memory_kind()) {}
    explicit DeviceWeight(CudaMemoryKind memory_kind)
        : bf16_storage(memory_kind),
          int8_storage(memory_kind),
          int4_storage(memory_kind),
          gguf_expert_storage(memory_kind),
          scales_storage(memory_kind),
          fp8_storage(memory_kind),
          nvfp4_packed_storage(memory_kind),
          nvfp4_block_scale_storage(memory_kind) {}

    DeviceBuffer<__nv_bfloat16> bf16_storage;
    DeviceBuffer<int8_t> int8_storage;
    DeviceBuffer<uint8_t> int4_storage;
    DeviceBuffer<uint8_t> gguf_expert_storage;
    std::vector<DeviceBuffer<uint8_t>> gguf_segment_storage;
    DeviceBuffer<float> scales_storage;
    DeviceBuffer<uint8_t> fp8_storage;
    DeviceBuffer<uint8_t> nvfp4_packed_storage;
    DeviceBuffer<uint8_t> nvfp4_block_scale_storage;
    std::vector<int64_t> shape;
    LinearWeight linear;
};

using WeightMap = std::unordered_map<std::string, DeviceWeight>;

}
