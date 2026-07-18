#include "lfm/concurrent.hpp"
#include "lfm/tokenizer.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {
int parse_positive(const char* text, const char* name) {
    const int value = std::atoi(text);
    if (value <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
    return value;
}
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL_SAFETENSORS TOKENIZER_JSON PROMPT [requests=8] [max_new=64]\n";
        return 2;
    }
    try {
        const int requests = argc > 4 ? parse_positive(argv[4], "requests") : 8;
        const int max_new = argc > 5 ? parse_positive(argv[5], "max_new") : 64;
        lfm::BpeTokenizer tokenizer(argv[2]);
        const std::vector<int32_t> prompt = tokenizer.encode(argv[3], true);

        lfm::ModelOptions model;
        model.fast_attention = true;
        model.fused_projections = true;
        model.fused_residuals = true;
        model.cuda_graph = true;

        lfm::ConcurrentEngineOptions engine_options;
        engine_options.max_active_requests = requests;
        engine_options.max_batched_tokens = std::max(256, requests * 2);
        engine_options.prefill_chunk_tokens = 256;
        engine_options.worker_thread = false;

        const int context = static_cast<int>(prompt.size()) + max_new + 8;
        lfm::ConcurrentEngine engine(argv[1], context, model, engine_options);
        std::vector<lfm::ConcurrentEngine::RequestId> ids;
        ids.reserve(static_cast<size_t>(requests));
        for (int i = 0; i < requests; ++i) {
            lfm::ConcurrentRequestOptions options;
            options.max_new_tokens = max_new;
            options.eos_token = tokenizer.eos_id();
            options.generation.seed = static_cast<uint64_t>(i + 1);
            ids.push_back(engine.submit(prompt, options));
        }

        const auto begin = std::chrono::steady_clock::now();
        int finished = 0;
        uint64_t returned_tokens = 0;
        std::vector<bool> done(static_cast<size_t>(requests), false);
        while (finished < requests) {
            engine.step();
            for (int i = 0; i < requests; ++i) {
                if (done[static_cast<size_t>(i)]) continue;
                lfm::PollResult result = engine.poll(ids[static_cast<size_t>(i)], 0);
                returned_tokens += result.tokens.size();
                if (result.finished) {
                    done[static_cast<size_t>(i)] = true;
                    ++finished;
                }
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - begin).count();
        const lfm::ConcurrentMetrics metrics = engine.metrics();
        std::cout << "requests=" << requests << '\n'
                  << "prompt_tokens_each=" << prompt.size() << '\n'
                  << "decoded_tokens=" << metrics.decoded_tokens << '\n'
                  << "elapsed_ms=" << elapsed_ms << '\n'
                  << "aggregate_tokens_per_second="
                  << (elapsed_ms > 0.0 ? metrics.decoded_tokens * 1000.0 / elapsed_ms : 0.0)
                  << '\n'
                  << "average_ttft_ms=" << metrics.average_ttft_ms() << '\n'
                  << "average_itl_ms=" << metrics.average_itl_ms() << '\n'
                  << "scheduler_steps=" << metrics.scheduler_steps << '\n'
                  << "packed_decode_steps=" << metrics.packed_decode_steps << '\n'
                  << "packed_decode_tokens=" << metrics.packed_decode_tokens << '\n'
                  << "lane_decode_tokens=" << metrics.lane_decode_tokens << '\n'
                  << "packed_fallback_batches=" << metrics.packed_fallback_batches << '\n'
                  << "maximum_packed_batch=" << metrics.maximum_packed_batch << '\n'
                  << "ragged_prefill_steps=" << metrics.ragged_prefill_steps << '\n'
                  << "ragged_prefill_tokens=" << metrics.ragged_prefill_tokens << '\n'
                  << "lane_prefill_tokens=" << metrics.lane_prefill_tokens << '\n'
                  << "maximum_ragged_prefill_batch="
                  << metrics.maximum_ragged_prefill_batch << '\n'
                  << "ragged_prefill_tokens_per_second="
                  << (metrics.cumulative_ragged_prefill_ms > 0.0
                          ? metrics.ragged_prefill_tokens * 1000.0 /
                                metrics.cumulative_ragged_prefill_ms
                          : 0.0)
                  << '\n'
                  << "packed_tokens_per_second="
                  << (metrics.cumulative_packed_decode_ms > 0.0
                          ? metrics.packed_decode_tokens * 1000.0 /
                                metrics.cumulative_packed_decode_ms
                          : 0.0)
                  << '\n'
                  << "physical_pages_current=" << metrics.logical_pages_used << '\n'
                  << "physical_pages_total=" << metrics.logical_pages_total << '\n'
                  << "physical_kv_bytes=" << metrics.physical_kv_bytes << '\n'
                  << "prefix_cache_hits=" << metrics.prefix_cache_hits << '\n'
                  << "prefix_cache_misses=" << metrics.prefix_cache_misses << '\n'
                  << "prefix_cache_inserts=" << metrics.prefix_cache_inserts << '\n'
                  << "prefix_cache_evictions=" << metrics.prefix_cache_evictions << '\n'
                  << "prefix_cache_partial_hits=" << metrics.prefix_cache_partial_hits << '\n'
                  << "prefix_reused_tokens=" << metrics.prefix_reused_tokens << '\n'
                  << "prefix_cow_pages=" << metrics.prefix_cow_pages << '\n'
                  << "prefix_radix_nodes=" << metrics.prefix_radix_nodes << '\n'
                  << "prefix_radix_lookups=" << metrics.prefix_radix_lookups << '\n'
                  << "prefix_cow_bytes_copied="
                  << metrics.prefix_cow_bytes_copied << '\n'
                  << "prefix_cow_bytes_saved="
                  << metrics.prefix_cow_bytes_saved << '\n'
                  << "direct_paged_prefill_tokens=" << metrics.direct_paged_prefill_tokens << '\n'
                  << "segmented_paged_decode_steps=" << metrics.segmented_paged_decode_steps << '\n'
                  << "segmented_paged_decode_tokens=" << metrics.segmented_paged_decode_tokens << '\n';
        (void) returned_tokens;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
