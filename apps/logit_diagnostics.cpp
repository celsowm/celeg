#include "celeg/app/logit_diagnostics.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace celeg::app {

void dump_logits_file(const std::string& path, const std::vector<float>& logits) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create logits file: " + path);
    out.write(reinterpret_cast<const char*>(logits.data()),
              static_cast<std::streamsize>(logits.size() * sizeof(float)));
    if (!out) throw std::runtime_error("failed writing logits file: " + path);
}

void print_top_logits(const std::vector<float>& logits, int count) {
    count = std::min(count, static_cast<int>(logits.size()));
    std::vector<int32_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + count, indices.end(),
        [&](int32_t a, int32_t b) {
            if (logits[static_cast<size_t>(a)] != logits[static_cast<size_t>(b)]) {
                return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)];
            }
            return a < b;
        });
    for (int i = 0; i < count; ++i) {
        const int32_t token = indices[static_cast<size_t>(i)];
        std::cerr << "top[" << i << "] token=" << token
                  << " logit=" << logits[static_cast<size_t>(token)] << '\n';
    }
}

}
