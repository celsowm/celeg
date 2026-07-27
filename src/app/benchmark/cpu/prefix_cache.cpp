#include "lfm/model/config/config.hpp"
#include "lfm/backend/cpu/concurrent.hpp"
#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/text/tokenizer.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage: lfm25-cpu-prefix-cache-benchmark MODEL_DIR PROMPT [REPEATS] [SUFFIX]\n";
            return 2;
        }
        const std::filesystem::path model_dir(argv[1]);
        const std::string prompt(argv[2]);
        const int repeats = argc > 3 ? std::stoi(argv[3]) : 4;
        const std::string suffix = argc > 4 ? argv[4] : "";
        if (repeats < 2) throw std::invalid_argument("repeats must be at least two");
        const lfm::detail::ModelBootstrap bootstrap =
            lfm::detail::load_model_bootstrap(model_dir);
        const auto& config = bootstrap.config;
        lfm::BpeTokenizer tokenizer((model_dir / "tokenizer.json").string());
        const auto base = tokenizer.encode(
            tokenizer.format_chat(
                std::vector<lfm::ChatMessage>{{lfm::ChatRole::User, prompt}}),
            false);
        const auto longer = suffix.empty() ? base : tokenizer.encode(
            tokenizer.format_chat(
                std::vector<lfm::ChatMessage>{{lfm::ChatRole::User, prompt + suffix}}),
            false);
        lfm::CpuModelOptions model_options;
        model_options.kv_cache_mode = lfm::CpuKvCacheMode::Bf16;
        model_options.numa_mode = lfm::CpuNumaMode::Local;
        lfm::CpuConcurrentEngineOptions engine_options;
        engine_options.max_active_requests = 1;
        engine_options.prefix_cache = true;
        lfm::CpuConcurrentEngine engine(
            (model_dir / "model.safetensors").string(),
            std::min(config.max_position_embeddings,
                     static_cast<int>(longer.size()) + 16),
            model_options, engine_options);
        lfm::CpuRequestOptions request_options;
        request_options.max_new_tokens = 1;
        request_options.eos_token_id = config.eos_token_id;
        request_options.generation.temperature = 0.0f;
        request_options.generation.top_k = 1;
        for (int run = 0; run < repeats; ++run) {
            const auto& tokens = run == 0 ? base : longer;
            const auto id = engine.submit(tokens, request_options);
            bool finished = false;
            while (!finished) {
                engine.step();
                (void)engine.poll(id, 8, &finished);
            }
        }
        const auto metrics = engine.metrics();
        std::cout << std::fixed << std::setprecision(3)
                  << "runs=" << repeats << '\n'
                  << "base_tokens=" << base.size() << '\n'
                  << "longer_tokens=" << longer.size() << '\n'
                  << "prefix_cache_hits=" << metrics.prefix_cache_hits << '\n'
                  << "prefix_cache_partial_hits=" << metrics.prefix_cache_partial_hits << '\n'
                  << "prefix_reused_tokens=" << metrics.prefix_reused_tokens << '\n'
                  << "prefix_cow_pages=" << metrics.prefix_cow_pages << '\n'
                  << "prefix_cow_bytes=" << metrics.prefix_cow_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
