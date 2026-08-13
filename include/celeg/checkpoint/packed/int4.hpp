#pragma once

#include "celeg/checkpoint/weight_repository.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

// The compressed-tensors `pack-quantized` INT4 representation.  The packed
// I32 payload is byte-addressable here: two signed 4-bit values occupy each
// byte, low nibble first.  Scales are BF16 and are stored per output row and
// per group of `group_size` input columns.
struct PackedInt4Matrix {
    int rows = 0;
    int cols = 0;
    int group_size = 0;
    std::vector<uint8_t> values;
    std::vector<float> scales;

    size_t packed_cols() const { return (static_cast<size_t>(cols) + 1) / 2; }
    size_t groups_per_row() const {
        return (static_cast<size_t>(cols) + static_cast<size_t>(group_size) - 1) /
            static_cast<size_t>(group_size);
    }
    void validate() const;
};

bool has_packed_int4_matrix(const IWeightRepository& repository,
                            std::string_view name);

PackedInt4Matrix load_packed_int4_matrix(
    const IWeightRepository& repository, std::string_view name,
    const std::vector<int64_t>& expected_shape);

std::vector<float> dequantize_packed_int4(const PackedInt4Matrix& matrix);

} // namespace celeg
