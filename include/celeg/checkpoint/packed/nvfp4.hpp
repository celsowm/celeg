#pragma once

#include "celeg/checkpoint/weight_repository.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

// NVFP4 (e2m1) weight: 2 values packed per byte, one UE4M3 scale per
// 16-element block along the row, plus a per-tensor fp32 "global scale" --
// the compressed-tensors "nvfp4-pack-quantized" format. On disk: a
// "<name>_packed" sidecar (uint8, [rows, cols/2]), a "<name>_scale" sidecar
// (F8_E4M3, [rows, cols/16], row-major -- not yet swizzled for cuBLASLt),
// and a "<name>_global_scale" scalar (F32). If a sibling
// "<module>.input_global_scale" scalar exists (module = name with a
// trailing ".weight" stripped), it calibrates the dynamic activation
// quantization; absent it, activations use a global scale of 1.0.
inline constexpr int kNvfp4PackedBlockSize = 16;

struct PackedNvfp4Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<uint8_t> packed;       // [rows, cols/2]
    std::vector<uint8_t> block_scales; // raw e4m3 bits, [rows, cols/block_size]
    float global_scale = 1.0f;
    float input_global_scale = 1.0f;

    void validate() const;
};

bool has_packed_nvfp4_matrix(const IWeightRepository& repository,
                             std::string_view name);

PackedNvfp4Matrix load_packed_nvfp4_matrix(
    const IWeightRepository& repository, std::string_view name,
    const std::vector<int64_t>& expected_shape);

}
