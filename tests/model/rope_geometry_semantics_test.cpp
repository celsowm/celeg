#include "celeg/model/rope_geometry.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    try {
        using namespace celeg;
        using namespace celeg::rope_geometry;

        if (rotary_pairs(0) != 0 || rotary_pairs(2) != 1 ||
            rotary_pairs(64) != 32) {
            throw std::runtime_error("RoPE pair-count semantics failed");
        }

        const auto split = pair_components(
            3, 8, RopePairingKind::SplitHalf);
        if (split.first != 3 || split.second != 11) {
            throw std::runtime_error("split-half RoPE pairing semantics failed");
        }

        const auto adjacent = pair_components(
            3, 8, RopePairingKind::AdjacentPairs);
        if (adjacent.first != 6 || adjacent.second != 7) {
            throw std::runtime_error("adjacent RoPE pairing semantics failed");
        }

        if (mrope_axis_for_pair(0, 2, 3, true) != 0 ||
            mrope_axis_for_pair(1, 2, 3, true) != 1 ||
            mrope_axis_for_pair(2, 2, 3, true) != 2 ||
            mrope_axis_for_pair(4, 2, 3, true) != 1) {
            throw std::runtime_error("interleaved MRoPE axis semantics failed");
        }

        if (mrope_axis_for_pair(0, 2, 3, false) != 0 ||
            mrope_axis_for_pair(1, 2, 3, false) != 0 ||
            mrope_axis_for_pair(2, 2, 3, false) != 1 ||
            mrope_axis_for_pair(4, 2, 3, false) != 1 ||
            mrope_axis_for_pair(5, 2, 3, false) != 2) {
            throw std::runtime_error("sectioned MRoPE axis semantics failed");
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
