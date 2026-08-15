#include "celeg/backend/cpu/kernels.hpp"

#include <stdexcept>

namespace celeg {

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
        for (int tap = 0; tap < cache_length; ++tap) {
            const int slot = (cursor + 1 + tap) % cache_length;
            conv += state[static_cast<size_t>(slot) * hidden + channel] *
                weight[static_cast<size_t>(tap) * hidden + channel];
        }
        output[channel] = c * conv;
    }
}

void cpu_conv_prefill(const float* projected_bcx, const float* weight,
                      float* state, float* output, size_t rows, int hidden,
                      int cache_length, int base_position,
                      CpuThreadPool& thread_pool) {
    if (!projected_bcx || !weight || !state || !output || rows == 0 ||
        hidden <= 0 || cache_length <= 0 || base_position < 0) {
        throw std::invalid_argument("invalid ShortConv prefill arguments");
    }
    const size_t channel_grain = std::max<size_t>(1,
        static_cast<size_t>(hidden) / std::max<size_t>(1, thread_pool.size() * 4));
    thread_pool.parallel_for(0, static_cast<size_t>(hidden), channel_grain,
        [&](size_t begin, size_t end) {
        for (size_t channel = begin; channel < end; ++channel) {
            for (size_t row = 0; row < rows; ++row) {
                const float* projected = projected_bcx + row * 3ULL * hidden;
                const int position = base_position + static_cast<int>(row);
                const int cursor = position % cache_length;
                state[static_cast<size_t>(cursor) * hidden + channel] =
                    projected[channel] * projected[2 * hidden + channel];
                float conv = 0.0f;
                for (int tap = 0; tap < cache_length; ++tap) {
                    const int slot = (cursor + 1 + tap) % cache_length;
                    conv += state[static_cast<size_t>(slot) * hidden + channel] *
                        weight[static_cast<size_t>(tap) * hidden + channel];
                }
                output[row * static_cast<size_t>(hidden) + channel] =
                    projected[hidden + channel] * conv;
            }
        }
    });
}

}
