#pragma once

#include "celeg/checkpoint/weight_repository.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace celeg {

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

}
