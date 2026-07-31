#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32(const std::string& path) {
    const auto bytes = std::filesystem::file_size(path);
    if (bytes == 0 || bytes % sizeof(float) != 0) {
        throw std::runtime_error("invalid raw float32 file size: " + path);
    }
    std::vector<float> values(static_cast<size_t>(bytes / sizeof(float)));
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open: " + path);
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("failed reading: " + path);
    return values;
}

std::vector<int32_t> top_indices(const std::vector<float>& values, int count) {
    count = std::min(count, static_cast<int>(values.size()));
    std::vector<int32_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + count, indices.end(),
        [&](int32_t a, int32_t b) {
            if (values[static_cast<size_t>(a)] != values[static_cast<size_t>(b)]) {
                return values[static_cast<size_t>(a)] > values[static_cast<size_t>(b)];
            }
            return a < b;
        });
    indices.resize(static_cast<size_t>(count));
    return indices;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: celeg-compare-logits reference.f32 candidate.f32\n";
            return 2;
        }
        const std::vector<float> reference = read_f32(argv[1]);
        const std::vector<float> candidate = read_f32(argv[2]);
        if (reference.size() != candidate.size()) {
            throw std::runtime_error("logit files have different element counts");
        }

        double sum_abs = 0.0;
        double sum_sq = 0.0;
        float max_abs = 0.0f;
        size_t max_index = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            const float error = std::fabs(reference[i] - candidate[i]);
            sum_abs += error;
            sum_sq += static_cast<double>(error) * error;
            if (error > max_abs) {
                max_abs = error;
                max_index = i;
            }
        }

        const int top_count = std::min<int>(10, static_cast<int>(reference.size()));
        const auto ref_top = top_indices(reference, top_count);
        const auto candidate_top = top_indices(candidate, top_count);
        int overlap = 0;
        for (int32_t token : ref_top) {
            if (std::find(candidate_top.begin(), candidate_top.end(), token) !=
                candidate_top.end()) {
                ++overlap;
            }
        }

        std::cout << "elements=" << reference.size() << '\n'
                  << "reference_top1=" << ref_top.front() << '\n'
                  << "candidate_top1=" << candidate_top.front() << '\n'
                  << "top1_match=" << (ref_top.front() == candidate_top.front() ? "yes" : "no") << '\n'
                  << "top" << top_count << "_overlap=" << overlap << "/" << top_count << '\n'
                  << "max_abs=" << max_abs << " at=" << max_index << '\n'
                  << "mean_abs=" << (sum_abs / reference.size()) << '\n'
                  << "rmse=" << std::sqrt(sum_sq / reference.size()) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
