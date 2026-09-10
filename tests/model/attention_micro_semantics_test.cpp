#include "celeg/attention/micro_semantics.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        if (celeg::attention_semantics::gqa_kv_head(0, 8, 2) != 0 ||
            celeg::attention_semantics::gqa_kv_head(3, 8, 2) != 0 ||
            celeg::attention_semantics::gqa_kv_head(4, 8, 2) != 1 ||
            celeg::attention_semantics::gqa_kv_head(7, 8, 2) != 1) {
            throw std::runtime_error("GQA head mapping semantics failed");
        }
        if (celeg::attention_semantics::sequence_length_from_query_position(0) != 1 ||
            celeg::attention_semantics::sequence_length_from_query_position(31) != 32 ||
            celeg::attention_semantics::query_position_from_sequence_length(1) != 0 ||
            celeg::attention_semantics::query_position_from_sequence_length(32) != 31) {
            throw std::runtime_error("attention position semantics failed");
        }
        const float scale = celeg::attention_semantics::attention_scale(64);
        if (std::abs(scale - 0.125f) > 1.0e-7f) {
            throw std::runtime_error("attention scale semantics failed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
