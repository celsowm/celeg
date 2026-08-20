#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <stdexcept>
#include <vector>

namespace celeg::cuda_loader_detail {

inline size_t checked_element_count(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("weight shape is empty");
    }
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("weight shape has non-positive dimension");
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

}
