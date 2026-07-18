#include "lfm/runtime_types.hpp"

#include <cmath>
#include <stdexcept>

namespace lfm {

void GenerationConfig::validate() const {
    if (!std::isfinite(temperature) || temperature < 0.0f) {
        throw std::invalid_argument("temperature must be finite and non-negative");
    }
    if (top_k <= 0 || top_k > LfmConfig::max_top_k) {
        throw std::invalid_argument("top_k must be between 1 and 128");
    }
    if (!std::isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f) {
        throw std::invalid_argument("top_p must be in (0, 1]");
    }
    if (!std::isfinite(repetition_penalty) || repetition_penalty < 1.0f) {
        throw std::invalid_argument("repetition_penalty must be finite and at least 1");
    }
    if (seed == 0) {
        throw std::invalid_argument("seed must be non-zero");
    }
}

} // namespace lfm
