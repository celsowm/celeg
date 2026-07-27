#include "lfm/runtime/concurrency.hpp"
#include "lfm/text/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int parse_positive(const char* text, const char* name) {
    const int value = std::atoi(text);
    if (value <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
    return value;
}

void complete(lfm::ConcurrentEngine& engine,
              lfm::ConcurrentEngine::RequestId id) {
    for (;;) {
        engine.step();
        const lfm::PollResult result = engine.poll(id, 0);
        if (result.finished) {
            if (result.status == lfm::RequestStatus::Failed) {
                throw std::runtime_error(result.error);
            }
            return;
        }
    }
}
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL_SAFETENSORS TOKENIZER_JSON PROMPT [repeats=4] [max_new=1] [suffix]\n";
        return 2;
    }
    try {
        const int repeats = argc > 4 ? parse_positive(argv[4], "repeats") : 4;
        const int max_new = argc > 5 ? parse_positive(argv[5], "max_new") : 1;
        lfm::BpeTokenizer tokenizer(argv[2]);
        std::vector<int32_t> prompt = tokenizer.encode(argv[3], true);
        std::vector<int32_t> extended_prompt = prompt;
        if (argc > 6 && argv[6][0] != '\0') {
            std::vector<int32_t> suffix = tokenizer.encode(argv[6], false);
            extended_prompt.insert(extended_prompt.end(), suffix.begin(), suffix.end());
        }

        // v0.0.12+ supports arbitrary prompt lengths through partial-page
        // copy-on-write, so no artificial page alignment is required.
        constexpr int page_tokens = 16;

        lfm::ModelOptions model;
        model.fast_attention = true;
        model.fused_projections = true;
        model.fused_residuals = true;
        model.cuda_graph = false;

        lfm::ConcurrentEngineOptions options;
        options.max_active_requests = 1;
        options.max_batched_tokens = 256;
        options.prefill_chunk_tokens = 256;
        options.page_tokens = page_tokens;
        options.worker_thread = false;
        options.prefix_cache = true;
        options.prefix_cache_entries = static_cast<size_t>(repeats + 1);

        const int context = static_cast<int>(
            std::max(prompt.size(), extended_prompt.size())) + max_new + 4;
        lfm::ConcurrentEngine engine(argv[1], context, model, options);

        for (int i = 0; i < repeats; ++i) {
            lfm::ConcurrentRequestOptions request;
            request.max_new_tokens = max_new;
            request.eos_token = -1;
            request.generation.seed = static_cast<uint64_t>(i + 1);
            const bool use_suffix = extended_prompt.size() > prompt.size() && i > 0;
            const std::vector<int32_t>& submitted_prompt =
                use_suffix ? extended_prompt : prompt;
            const auto begin = std::chrono::steady_clock::now();
            const auto id = engine.submit(submitted_prompt, request);
            complete(engine, id);
            const auto end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(
                                  end - begin).count();
            const auto metrics = engine.metrics();
            std::cout << "iteration=" << i
                      << " elapsed_ms=" << ms
                      << " hits=" << metrics.prefix_cache_hits
                      << " misses=" << metrics.prefix_cache_misses
                      << " inserts=" << metrics.prefix_cache_inserts
                      << " partial_hits=" << metrics.prefix_cache_partial_hits
                      << " reused_tokens=" << metrics.prefix_reused_tokens
                      << " prompt_tokens=" << submitted_prompt.size()
                      << " cow_pages=" << metrics.prefix_cow_pages
                      << " cow_bytes_saved=" << metrics.prefix_cow_bytes_saved
                      << " radix_nodes=" << metrics.prefix_radix_nodes
                      << " radix_lookups=" << metrics.prefix_radix_lookups
                      << " pages=" << metrics.logical_pages_used
                      << '/' << metrics.logical_pages_total << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
