#include "lfm/backend/cpu/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    const auto bytes = input.tellg();
    if (bytes < 0 || static_cast<size_t>(bytes) % sizeof(T) != 0) {
        throw std::runtime_error("invalid binary length: " + path.string());
    }
    std::vector<T> result(static_cast<size_t>(bytes) / sizeof(T));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(result.data()), bytes);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return result;
}

std::vector<size_t> top_indices(const std::vector<float>& values, size_t count) {
    std::vector<size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    count = std::min(count, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + static_cast<ptrdiff_t>(count),
                      indices.end(), [&](size_t a, size_t b) {
                          return values[a] > values[b];
                      });
    indices.resize(count);
    return indices;
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage: lfm25-cpu-compare-reference MODEL.safetensors REFERENCE_DIR [KV] [ISA] [THREADS]\n";
            return 2;
        }
        const std::filesystem::path reference(argv[2]);
        const auto tokens = read_binary<int32_t>(reference / "tokens.i32");
        const auto expected = read_binary<float>(reference / "prefill_logits.f32");
        if (tokens.empty()) throw std::runtime_error("reference token sequence is empty");
        lfm::CpuModelOptions options;
        options.kv_cache_mode = lfm::parse_cpu_kv_cache_mode(argc > 3 ? argv[3] : "fp32");
        options.isa = lfm::parse_cpu_isa(argc > 4 ? argv[4] : "scalar");
        options.threads = argc > 5 ? static_cast<size_t>(std::stoul(argv[5])) : 1;
        options.use_pack_cache = true;
        lfm::GenerationConfig generation;
        generation.temperature = 0.0f;
        generation.top_k = 1;
        lfm::CpuModel model(argv[1], static_cast<int>(tokens.size() + 8),
                            options, generation);
        if (expected.size() != model.diagnostics().copy_logits().size()) {
            throw std::runtime_error("reference logits have the wrong vocabulary size");
        }
        model.session().prefill(tokens);
        const auto actual = model.diagnostics().copy_logits();
        double absolute_sum = 0.0;
        double squared_sum = 0.0;
        double dot = 0.0;
        double actual_norm = 0.0;
        double expected_norm = 0.0;
        float maximum = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i) {
            const float difference = std::abs(actual[i] - expected[i]);
            maximum = std::max(maximum, difference);
            absolute_sum += difference;
            squared_sum += static_cast<double>(difference) * difference;
            dot += static_cast<double>(actual[i]) * expected[i];
            actual_norm += static_cast<double>(actual[i]) * actual[i];
            expected_norm += static_cast<double>(expected[i]) * expected[i];
        }
        const auto actual_top = top_indices(actual, 10);
        const auto expected_top = top_indices(expected, 10);
        size_t intersection = 0;
        for (size_t index : actual_top) {
            if (std::find(expected_top.begin(), expected_top.end(), index) != expected_top.end()) {
                ++intersection;
            }
        }
        const double cosine = dot / std::sqrt(actual_norm * expected_norm);
        std::cout << std::fixed << std::setprecision(8)
                  << "tokens=" << tokens.size() << '\n'
                  << "max_abs_error=" << maximum << '\n'
                  << "mean_abs_error=" << absolute_sum / actual.size() << '\n'
                  << "rmse=" << std::sqrt(squared_sum / actual.size()) << '\n'
                  << "cosine_similarity=" << cosine << '\n'
                  << "top1_equal=" << (actual_top.front() == expected_top.front()) << '\n'
                  << "top10_intersection=" << intersection << '\n'
                  << "actual_top1=" << actual_top.front() << '\n'
                  << "expected_top1=" << expected_top.front() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
