#include "lfm/backend/cpu/kernels.hpp"

#include <stdexcept>

namespace lfm {

void cpu_conv_decode(const float* projected_bcx, const float* weight,
                     float* state, float* output, int hidden,
                     int cache_length, int position) {
    if (!projected_bcx || !weight || !state || !output || hidden <= 0 ||
        cache_length <= 0 || position < 0) {
        throw std::invalid_argument("invalid ShortConv arguments");
    }
    const int cursor = position % cache_length;
    for (int channel = 0; channel < hidden; ++channel) {
        const float b = projected_bcx[channel];
        const float c = projected_bcx[hidden + channel];
        const float x = projected_bcx[2 * hidden + channel];
        state[static_cast<size_t>(cursor) * hidden + channel] = b * x;
        float conv = 0.0f;
        const size_t weight_offset = static_cast<size_t>(channel) * cache_length;
        for (int tap = 0; tap < cache_length; ++tap) {
            const int slot = (cursor + 1 + tap) % cache_length;
            conv += state[static_cast<size_t>(slot) * hidden + channel] * weight[weight_offset + tap];
        }
        output[channel] = c * conv;
    }
}

} // namespace lfm
