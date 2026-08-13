#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/backend/cpu/concurrent.hpp"
#include "celeg/text/chat_template.hpp"
#include "celeg/text/tokenizer.hpp"
#include "celeg/text/tokenizer_definition.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 6) {
            std::cerr << "usage: celeg-cpu-concurrent-benchmark MODEL_DIR PROMPT REQUESTS MAX_NEW THREADS [KV] [ISA]\n";
            return 2;
        }
        const std::filesystem::path model_dir(argv[1]);
        const std::string prompt(argv[2]);
        const int requests = std::stoi(argv[3]);
        const int max_new = std::stoi(argv[4]);
        const int threads = std::stoi(argv[5]);
        const std::string kv = argc > 6 ? argv[6] : "bf16";
        const std::string isa = argc > 7 ? argv[7] : "auto";
        if (requests <= 0 || max_new <= 0 || threads < 0) {
            throw std::invalid_argument("benchmark counts must be positive");
        }

        const celeg::detail::ModelBootstrap bootstrap =
            celeg::detail::load_model_bootstrap(model_dir);
        const auto& topology = bootstrap.model.topology;
        celeg::BpeTokenizer tokenizer(celeg::load_tokenizer_definition_json(
            (model_dir / "tokenizer.json").string()));
        const auto chat_template = celeg::resolve_chat_template(
            bootstrap.checkpoint.metadata, tokenizer);
        const std::vector<int32_t> tokens = tokenizer.encode(
            celeg::render_chat(
                std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, prompt}},
                chat_template),
            false);

        celeg::CpuModelOptions model_options;
        model_options.threads = static_cast<size_t>(threads);
        model_options.isa = celeg::parse_cpu_isa(isa);
        model_options.kv_cache_mode = celeg::parse_cpu_kv_cache_mode(kv);
        model_options.affinity = celeg::CpuAffinityPolicy::Compact;

        celeg::CpuConcurrentEngineOptions engine_options;
        engine_options.max_active_requests = static_cast<size_t>(requests);
        engine_options.max_batched_tokens = std::max(256, requests * 4);
        engine_options.max_prefill_batch = static_cast<size_t>(requests);
        engine_options.max_decode_batch = static_cast<size_t>(requests);

        celeg::CpuConcurrentEngine engine(
            (model_dir / "model.safetensors").string(),
            std::min(topology.dims.max_position_embeddings,
                     static_cast<int>(tokens.size()) + max_new + 8),
            model_options, engine_options);

        celeg::ConcurrentRequestOptions request_options;
        request_options.max_new_tokens = static_cast<size_t>(max_new);
        request_options.eos_tokens = bootstrap.model.topology.dims.token_policy.eos_token_ids;
        request_options.generation.temperature = 0.0f;
        request_options.generation.top_k = 1;

        std::vector<celeg::RequestId> ids;
        ids.reserve(static_cast<size_t>(requests));
        const auto started = std::chrono::steady_clock::now();
        for (int i = 0; i < requests; ++i) {
            request_options.generation.seed = static_cast<uint64_t>(i + 1);
            ids.push_back(engine.submit(tokens, request_options));
        }

        size_t finished = 0;
        std::vector<uint8_t> done(ids.size(), 0);
        while (finished < ids.size()) {
            if (!engine.step()) {
                std::this_thread::yield();
            }
            for (size_t i = 0; i < ids.size(); ++i) {
                if (done[i]) continue;
                const auto result = engine.poll(ids[i], 4096);
                if (result.finished) {
                    done[i] = 1;
                    ++finished;
                }
            }
        }
        const auto ended = std::chrono::steady_clock::now();
        const double wall_ms = std::chrono::duration<double, std::milli>(
            ended - started).count();
        const celeg::CpuConcurrentMetrics metrics = engine.metrics();

        std::cout << std::fixed << std::setprecision(3)
                  << "backend=" << engine.backend_description() << '\n'
                  << "requests=" << requests << '\n'
                  << "prompt_tokens=" << tokens.size() << '\n'
                  << "decode_tokens=" << metrics.decode_tokens << '\n'
                  << "wall_ms=" << wall_ms << '\n'
                  << "aggregate_tokens_per_second="
                  << (wall_ms > 0.0 ? metrics.decode_tokens * 1000.0 / wall_ms : 0.0) << '\n'
                  << "kernel_decode_tokens_per_second="
                  << metrics.decode_tokens_per_second() << '\n'
                  << "prefill_tokens_per_second="
                  << metrics.prefill_tokens_per_second() << '\n'
                  << "average_ttft_ms=" << metrics.average_ttft_ms() << '\n'
                  << "average_itl_ms=" << metrics.average_itl_ms() << '\n'
                  << "maximum_prefill_batch=" << metrics.maximum_prefill_batch << '\n'
                  << "maximum_decode_batch=" << metrics.maximum_decode_batch << '\n'
                  << "ragged_prefill_steps=" << metrics.ragged_prefill_steps << '\n'
                  << "packed_decode_steps=" << metrics.packed_decode_steps << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
