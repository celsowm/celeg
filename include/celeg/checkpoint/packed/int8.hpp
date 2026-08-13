#pragma once

#include "celeg/checkpoint/weight_repository.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace celeg {

// The compressed-tensors `pack-quantized` INT8 representation used by the
// official Nanbeige GPTQ checkpoint. Values are signed INT8, with one BF16
// scale per output row; the packed source stores four unsigned bytes per I32.
struct PackedInt8Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<int8_t> values;
    std::vector<float> scales;

    void validate() const;
};

bool has_packed_int8_matrix(const IWeightRepository& repository,
                            std::string_view name);

PackedInt8Matrix load_packed_int8_matrix(
    const IWeightRepository& repository, std::string_view name,
    const std::vector<int64_t>& expected_shape);

} // namespace celeg
