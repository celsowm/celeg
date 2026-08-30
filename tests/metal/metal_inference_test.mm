#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/metal/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double cosine(std::span<const float> left, std::span<const float> right) {
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
        dot += static_cast<double>(left[index]) * right[index];
        left_norm += static_cast<double>(left[index]) * left[index];
        right_norm += static_cast<double>(right[index]) * right[index];
    }
    return dot / std::sqrt(left_norm * right_norm);
}

double rmse(std::span<const float> left, std::span<const float> right) {
    double sum = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
        const double difference = static_cast<double>(left[index]) - right[index];
        sum += difference * difference;
    }
    return std::sqrt(sum / static_cast<double>(left.size()));
}

int top_index(std::span<const float> values) {
    return static_cast<int>(std::distance(values.begin(),
        std::max_element(values.begin(), values.end())));
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: metal_inference_test MODEL");
        std::vector<int32_t> tokens{1, 36309};
        tokens.reserve(512);
        for (int32_t index = 2; index < 512; ++index) {
            tokens.push_back((1 + index * 36309) % 65536);
        }
        celeg::CpuModel cpu(argv[1], 640);
        celeg::MetalModel metal(argv[1], 640);
        celeg::MetalModel tokenized(argv[1], 640);
        auto cpu_session = cpu.session();
        auto metal_session = metal.session();
        auto tokenized_session = tokenized.session();
        celeg::GenerationConfig generation;
        generation.temperature = 0.0f;
        generation.top_k = 1;
        cpu_session.set_generation_config(generation);
        metal_session.set_generation_config(generation);
        for (const size_t count : std::vector<size_t>{1, 2, 8, 32, 512}) {
            const std::vector<int32_t> prefix(tokens.begin(), tokens.begin() + count);
            cpu_session.prefill(prefix);
            metal_session.prefill(prefix);
            const std::vector<float> cpu_logits = cpu.diagnostics().copy_logits();
            const std::vector<float> metal_logits = metal_session.copy_logits();
            tokenized_session.prefill({prefix.front()});
            for (size_t index = 1; index < prefix.size(); ++index) {
                tokenized_session.eval_token(prefix[index]);
            }
            const std::vector<float> tokenized_logits = tokenized_session.copy_logits();
            if (cpu_logits.size() != metal_logits.size()) {
                throw std::runtime_error("CPU and Metal vocabulary sizes differ");
            }
            const double batch_similarity = cosine(metal_logits, tokenized_logits);
            if (!(batch_similarity > 0.999999) ||
                top_index(metal_logits) != top_index(tokenized_logits)) {
                throw std::runtime_error("Metal batched prefill differs from tokenized execution");
            }
            const double similarity = cosine(cpu_logits, metal_logits);
            const double error = rmse(cpu_logits, metal_logits);
            const int cpu_top = top_index(cpu_logits);
            const int metal_top = top_index(metal_logits);
            std::cout << "tokens=" << count << " cosine=" << similarity << " rmse=" << error
                      << " cpu_top=" << cpu_top << " metal_top=" << metal_top << '\n';
            if (count == 2 && (!(similarity > 0.80) || cpu_top != metal_top)) {
                throw std::runtime_error("Metal logits failed CPU parity check");
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
