#include "celeg/model/runtime_types.hpp"

#include <cmath>
#include <stdexcept>

namespace celeg {

double RuntimeMetrics::prefill_tokens_per_second() const {
    return last_prefill_ms > 0.0
        ? static_cast<double>(prefill_tokens) * 1000.0 / last_prefill_ms
        : 0.0;
}

double RuntimeMetrics::decode_tokens_per_second() const {
    return cumulative_decode_ms > 0.0
        ? static_cast<double>(decoded_tokens) * 1000.0 / cumulative_decode_ms
        : 0.0;
}

void GenerationConfig::validate() const {
    if (!std::isfinite(temperature) || temperature < 0.0f) {
        throw std::invalid_argument("temperature must be finite and non-negative");
    }
    if (top_k <= 0 || top_k > kMaxTopK) {
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

} // namespace celeg
