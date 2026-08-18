#include "celeg/backend/cpu/model.hpp"

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
#include <string_view>
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

void print_tokens(std::string_view label, const std::vector<int32_t>& tokens) {
    std::cout << label << '=';
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << tokens[index];
    }
    std::cout << '\n';
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage: celeg-cpu-compare-reference MODEL REFERENCE_DIR [KV] [ISA] [THREADS]\n";
            return 2;
        }
        const std::filesystem::path reference(argv[2]);
        const auto tokens = read_binary<int32_t>(reference / "tokens.i32");
        const auto expected = read_binary<float>(reference / "prefill_logits.f32");
        const std::filesystem::path generated_path = reference / "generated_tokens.i32";
        const std::vector<int32_t> expected_generation =
            std::filesystem::exists(generated_path)
                ? read_binary<int32_t>(generated_path)
                : std::vector<int32_t>{};
        if (tokens.empty()) throw std::runtime_error("reference token sequence is empty");
        celeg::CpuModelOptions options;
        options.kv_cache_mode = celeg::parse_cpu_kv_cache_mode(argc > 3 ? argv[3] : "fp32");
        options.isa = celeg::parse_cpu_isa(argc > 4 ? argv[4] : "scalar");
        options.threads = argc > 5 ? static_cast<size_t>(std::stoul(argv[5])) : 1;
        options.use_pack_cache = true;
        celeg::GenerationConfig generation;
        generation.temperature = 0.0f;
        generation.top_k = 1;
        generation.top_p = 1.0f;
        generation.repetition_penalty = 1.0f;
        celeg::CpuModel model(argv[1],
                             static_cast<int>(tokens.size() + expected_generation.size() + 8),
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
        const bool top1_equal = actual_top.front() == expected_top.front();
        const double cosine = dot / std::sqrt(actual_norm * expected_norm);

        std::vector<int32_t> actual_generation;
        actual_generation.reserve(expected_generation.size());
        size_t first_generation_mismatch = expected_generation.size();
        for (size_t index = 0; index < expected_generation.size(); ++index) {
            const int32_t token = model.session().decode();
            actual_generation.push_back(token);
            if (first_generation_mismatch == expected_generation.size() &&
                token != expected_generation[index]) {
                first_generation_mismatch = index;
            }
        }
        const bool generation_equal =
            first_generation_mismatch == expected_generation.size();

        std::cout << std::fixed << std::setprecision(8)
                  << "tokens=" << tokens.size() << '\n'
                  << "max_abs_error=" << maximum << '\n'
                  << "mean_abs_error=" << absolute_sum / actual.size() << '\n'
                  << "rmse=" << std::sqrt(squared_sum / actual.size()) << '\n'
                  << "cosine_similarity=" << cosine << '\n'
                  << "top1_equal=" << top1_equal << '\n'
                  << "top10_intersection=" << intersection << '\n'
                  << "actual_top1=" << actual_top.front() << '\n'
                  << "expected_top1=" << expected_top.front() << '\n'
                  << "generated_token_count=" << expected_generation.size() << '\n'
                  << "generation_equal=" << generation_equal << '\n';
        if (!generation_equal) {
            std::cout << "first_generation_mismatch=" << first_generation_mismatch << '\n';
        }
        if (!expected_generation.empty()) {
            print_tokens("expected_generation", expected_generation);
            print_tokens("actual_generation", actual_generation);
        }
        return top1_equal && generation_equal ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
