#include "celeg/attention/merge_semantics.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require_close(float actual, float expected, float tolerance = 1.0e-6f) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error("attention merge semantics mismatch");
    }
}

}

int main() {
    try {
        using namespace celeg::attention_semantics;

        const auto first = merge_pair(
            -std::numeric_limits<float>::infinity(), 0.0f,
            2.0f, 3.0f);
        require_close(first.maximum, 2.0f);
        require_close(first.denominator, 3.0f);
        require_close(first.destination_scale, 0.0f);
        require_close(first.source_scale, 1.0f);
        require_close(merge_accumulate(0.0f, 5.0f, first), 5.0f);

        const auto second = merge_pair(2.0f, 3.0f, 4.0f, 2.0f);
        const float expected_old = std::exp(-2.0f);
        require_close(second.maximum, 4.0f);
        require_close(second.destination_scale, expected_old);
        require_close(second.source_scale, 1.0f);
        require_close(second.denominator, 3.0f * expected_old + 2.0f);
        require_close(
            merge_accumulate(5.0f, 7.0f, second),
            5.0f * expected_old + 7.0f);

        require_close(partial_rescale(1.5f, 4.0f), std::exp(-2.5f));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
