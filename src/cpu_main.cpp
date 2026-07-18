#include "lfm/config.hpp"
#include "lfm/cpu_isa.hpp"
#include "lfm/cpu_model.hpp"
#include "lfm/cpu_topology.hpp"
#include "lfm/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Args {
    std::string model_dir;
    std::string prompt;
    std::string system;
    std::string isa = "auto";
    std::string pack_cache;
    std::string affinity = "none";
    std::string kv_cache = "bf16";
    std::string numa = "disabled";
    int context = 4096;
    int max_new_tokens = 128;
    int threads = 0;
    int group_size = 32;
    int kv_page_tokens = 32;
    int prefill_chunk_tokens = 256;
    int prefill_chunk_threshold = 16;
    int attention_parallel_threshold = 256;
    int attention_page_tile = 4;
    int top_k = 50;
    float top_p = 1.0f;
    float temperature = 0.1f;
    float repetition_penalty = 1.05f;
    uint64_t seed = 1;
    bool raw_prompt = false;
    bool no_pack_cache = false;
    bool print_cpu = false;
    bool memory_report = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + key);
            return argv[i];
        };
        if (key == "--model") args.model_dir = value();
        else if (key == "--prompt") args.prompt = value();
        else if (key == "--system") args.system = value();
        else if (key == "--cpu-isa") args.isa = value();
        else if (key == "--cpu-pack-cache") args.pack_cache = value();
        else if (key == "--cpu-affinity") args.affinity = value();
        else if (key == "--cpu-kv-cache") args.kv_cache = value();
        else if (key == "--cpu-numa") args.numa = value();
        else if (key == "--context") args.context = std::stoi(value());
        else if (key == "--max-new-tokens") args.max_new_tokens = std::stoi(value());
        else if (key == "--threads") args.threads = std::stoi(value());
        else if (key == "--cpu-q4-group") args.group_size = std::stoi(value());
        else if (key == "--cpu-kv-page-tokens") args.kv_page_tokens = std::stoi(value());
        else if (key == "--cpu-prefill-chunk") args.prefill_chunk_tokens = std::stoi(value());
        else if (key == "--cpu-prefill-threshold") args.prefill_chunk_threshold = std::stoi(value());
        else if (key == "--cpu-attention-threshold") args.attention_parallel_threshold = std::stoi(value());
        else if (key == "--cpu-attention-page-tile") args.attention_page_tile = std::stoi(value());
        else if (key == "--top-k") args.top_k = std::stoi(value());
        else if (key == "--top-p") args.top_p = std::stof(value());
        else if (key == "--temperature") args.temperature = std::stof(value());
        else if (key == "--repetition-penalty") args.repetition_penalty = std::stof(value());
        else if (key == "--seed") args.seed = std::stoull(value());
        else if (key == "--raw-prompt") args.raw_prompt = true;
        else if (key == "--no-pack-cache") args.no_pack_cache = true;
        else if (key == "--print-cpu") args.print_cpu = true;
        else if (key == "--memory-report") args.memory_report = true;
        else if (key == "--help") {
            std::cout
                << "lfm25-cpu-run --model DIR --prompt TEXT [options]\n"
                << "  --cpu-isa auto|scalar|avx2|avx-vnni|avx512-vnni|neon\n"
                << "    (AMX/I8MM/SME2 remain diagnostic-only in v0.0.20)\n"
                << "  --cpu-q4-group 32|64 --threads N\n"
                << "  --cpu-affinity none|compact|scatter\n"
                << "  --cpu-kv-cache fp32|bf16 --cpu-kv-page-tokens N\n"
                << "  --cpu-prefill-chunk N --cpu-prefill-threshold N\n"
                << "  --cpu-pack-cache DIR | --no-pack-cache\n"
                << "  --context N --max-new-tokens N --memory-report\n"
                << "  --temperature F --top-k N --top-p F\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + key);
    }
    if (args.print_cpu && args.model_dir.empty()) return args;
    if (args.model_dir.empty()) throw std::runtime_error("--model is required");
    if (args.prompt.empty()) throw std::runtime_error("--prompt is required");
    if (args.context <= 0 || args.max_new_tokens < 0 || args.threads < 0 ||
        args.kv_page_tokens <= 0 || args.prefill_chunk_tokens <= 0 ||
        args.prefill_chunk_threshold <= 0 ||
        args.attention_parallel_threshold <= 0 || args.attention_page_tile <= 0 ||
        (args.group_size != 32 && args.group_size != 64)) {
        throw std::runtime_error("invalid CPU numeric argument");
    }
    return args;
}

std::string bytes(size_t count) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(count);
    int unit = 0;
    while (value >= 1024.0 && unit < 3) { value /= 1024.0; ++unit; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit ? 2 : 0) << value << ' ' << units[unit];
    return out.str();
}
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const lfm::CpuCapabilities caps = lfm::detect_cpu_capabilities();
        if (args.print_cpu) {
            std::cout << caps.summary() << '\n'
                      << lfm::detect_cpu_topology().summary() << '\n';
            if (args.model_dir.empty()) return 0;
        }
        const std::filesystem::path model(args.model_dir);
        const lfm::ModelConfig config = lfm::ModelConfig::load((model / "config.json").string());
        config.validate_compiled_backend();
        if (args.context > config.max_position_embeddings) {
            throw std::runtime_error("--context exceeds model maximum");
        }
        lfm::BpeTokenizer tokenizer((model / "tokenizer.json").string());
        const std::string text = args.raw_prompt
            ? args.prompt : tokenizer.format_chat(args.prompt, args.system);
        const std::vector<int32_t> input = tokenizer.encode(text, args.raw_prompt);
        if (static_cast<int>(input.size()) + args.max_new_tokens > args.context) {
            throw std::runtime_error("prompt plus output exceeds context");
        }
        lfm::CpuModelOptions options;
        options.isa = lfm::parse_cpu_isa(args.isa);
        options.weight_format = args.group_size == 64
            ? lfm::CpuWeightFormat::Q4Group64 : lfm::CpuWeightFormat::Q4Group32;
        options.threads = static_cast<size_t>(args.threads);
        options.affinity = lfm::parse_cpu_affinity(args.affinity);
        options.kv_cache_mode = lfm::parse_cpu_kv_cache_mode(args.kv_cache);
        options.kv_page_tokens = static_cast<size_t>(args.kv_page_tokens);
        options.prefill_chunk_tokens = static_cast<size_t>(args.prefill_chunk_tokens);
        options.prefill_chunk_threshold = static_cast<size_t>(args.prefill_chunk_threshold);
        options.attention_parallel_threshold = static_cast<size_t>(args.attention_parallel_threshold);
        options.attention_page_tile = static_cast<size_t>(args.attention_page_tile);
        options.numa_mode = lfm::parse_cpu_numa_mode(args.numa);
        options.use_pack_cache = !args.no_pack_cache;
        if (!args.pack_cache.empty()) options.pack_cache_directory = args.pack_cache;
        lfm::GenerationConfig generation;
        generation.temperature = args.temperature;
        generation.top_k = args.top_k;
        generation.top_p = args.top_p;
        generation.repetition_penalty = args.repetition_penalty;
        generation.seed = args.seed;
        lfm::CpuModel engine((model / "model.safetensors").string(), args.context,
                             options, generation);
        std::cerr << "backend=" << engine.backend_description() << '\n';
        if (!engine.pack_path().empty()) std::cerr << "cpu.pack_path=" << engine.pack_path() << '\n';
        if (args.memory_report) {
            const auto stats = engine.memory_stats();
            std::cerr << "memory.weights=" << bytes(stats.weights) << '\n'
                      << "memory.kv_cache=" << bytes(stats.kv_cache) << '\n'
                      << "memory.conv_state=" << bytes(stats.conv_state) << '\n'
                      << "memory.activations=" << bytes(stats.activations) << '\n'
                      << "memory.kv_pages_used=" << stats.kv_pages_used << '\n'
                      << "memory.kv_pages_total=" << stats.kv_pages_total << '\n'
                      << "memory.total=" << bytes(stats.total()) << '\n';
        }
        engine.prefill(input);
        std::string pending;
        for (int i = 0; i < args.max_new_tokens; ++i) {
            const int32_t token = engine.decode();
            if (token == config.eos_token_id) break;
            pending += tokenizer.decode({token}, true);
            std::cout << pending << std::flush;
            pending.clear();
        }
        std::cout << '\n';
        const lfm::RuntimeMetrics metrics = engine.runtime_metrics();
        std::cerr << std::fixed << std::setprecision(3)
                  << "runtime.prefill_tokens=" << metrics.prefill_tokens << '\n'
                  << "runtime.prefill_ms=" << metrics.last_prefill_ms << '\n'
                  << "runtime.prefill_tokens_per_second=" << metrics.prefill_tokens_per_second() << '\n'
                  << "runtime.decode_tokens=" << metrics.decoded_tokens << '\n'
                  << "runtime.decode_ms=" << metrics.cumulative_decode_ms << '\n'
                  << "runtime.decode_tokens_per_second=" << metrics.decode_tokens_per_second() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
