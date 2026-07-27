#include "lfm/backend/cpu/model.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 8) {
            std::cerr << "usage: lfm25-cpu-prefill-benchmark MODEL_SAFETENSORS "
                         "[TOKENS=1024] [CHUNK=256] [PAGE=32] "
                         "[KV=bf16] [ISA=auto] [THREADS=0]\n";
            return 2;
        }
        const std::string model = argv[1];
        const size_t token_count = argc > 2 ? std::stoull(argv[2]) : 1024;
        const size_t chunk = argc > 3 ? std::stoull(argv[3]) : 256;
        const size_t page = argc > 4 ? std::stoull(argv[4]) : 32;
        const std::string kv = argc > 5 ? argv[5] : "bf16";
        const std::string isa = argc > 6 ? argv[6] : "auto";
        const size_t threads = argc > 7 ? std::stoull(argv[7]) : 0;
        if (token_count == 0 || chunk == 0 || page == 0) {
            throw std::invalid_argument("tokens, chunk and page must be positive");
        }
        lfm::CpuModelOptions options;
        options.isa = lfm::parse_cpu_isa(isa);
        options.kv_cache_mode = lfm::parse_cpu_kv_cache_mode(kv);
        options.kv_page_tokens = page;
        options.prefill_chunk_tokens = chunk;
        options.prefill_chunk_threshold = 1;
        options.threads = threads;
        lfm::GenerationConfig generation;
        generation.temperature = 0.0f;
        generation.top_k = 1;
        lfm::CpuModel runtime(model, static_cast<int>(token_count + 16),
                              options, generation);
        std::vector<int32_t> tokens(token_count, 1);
        runtime.session().prefill(tokens);
        const auto metrics = runtime.diagnostics().runtime_metrics();
        const auto profile = runtime.diagnostics().prefill_profile();
        const auto memory = runtime.diagnostics().memory_stats();
        std::cout << std::fixed << std::setprecision(3)
                  << "backend=" << runtime.diagnostics().backend_description() << '\n'
                  << "prefill.tokens=" << metrics.prefill_tokens << '\n'
                  << "prefill.ms=" << metrics.last_prefill_ms << '\n'
                  << "prefill.tokens_per_second="
                  << metrics.prefill_tokens_per_second() << '\n'
                  << "prefill.profile.linear_ms=" << profile.linear_ms << '\n'
                  << "prefill.profile.attention_ms=" << profile.attention_ms << '\n'
                  << "prefill.profile.shortconv_ms=" << profile.shortconv_ms << '\n'
                  << "prefill.profile.total_ms=" << profile.total_ms << '\n'
                  << "kv.pages.used=" << memory.kv_pages_used << '\n'
                  << "kv.pages.total=" << memory.kv_pages_total << '\n'
                  << "kv.bytes=" << memory.kv_cache << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
