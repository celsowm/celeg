#pragma once

#include "celeg/checkpoint/weight_repository.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

// FP8 E4M3 weight with a per-channel (per-output-row) fp32 dequant scale --
// the compressed-tensors "float-quantized" format. On disk the weight
// tensor keeps its ordinary name but with dtype F8_E4M3, and is accompanied
// by a "<name>_scale" sidecar of shape [rows, 1].
struct PackedFp8Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<uint8_t> values;  // raw e4m3 bit patterns, row-major
    std::vector<float> scales;    // one per row

    void validate() const;
};

bool has_packed_fp8_matrix(const IWeightRepository& repository,
                           std::string_view name);

PackedFp8Matrix load_packed_fp8_matrix(
    const IWeightRepository& repository, std::string_view name,
    const std::vector<int64_t>& expected_shape);

}
