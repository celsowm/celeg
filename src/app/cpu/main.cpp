#include "celeg/backend/cpu/isa.hpp"
#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/topology.hpp"
#include "celeg/text/chat_template.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/runtime/request_types.hpp"
#include "celeg/checkpoint/downloader.hpp"
#include "celeg/text/tokenizer.hpp"
#include "celeg/runtime/context.hpp"

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
    std::string repo;
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
    bool group_size_explicit = false;
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
        else if (key == "--repo") args.repo = value();
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
        else if (key == "--cpu-q4-group") {
            args.group_size = std::stoi(value());
            args.group_size_explicit = true;
        }
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
                << "celeg-cpu-run [--model DIR | --repo REPO_ID] --prompt TEXT [options]\n"
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
    if (args.print_cpu && args.model_dir.empty() && args.repo.empty()) return args;
    if (args.model_dir.empty() && args.repo.empty())
        throw std::runtime_error("--model or --repo is required");
    if (!args.model_dir.empty() && !args.repo.empty())
        throw std::runtime_error("--model and --repo are mutually exclusive");
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
        const celeg::CpuCapabilities caps = celeg::detect_cpu_capabilities();
        if (args.print_cpu) {
            std::cout << caps.summary() << '\n'
                      << celeg::detect_cpu_topology().summary() << '\n';
            if (args.model_dir.empty() && args.repo.empty()) return 0;
        }
        std::string repo_id = args.repo;
        std::string quant_tag;
        if (!repo_id.empty()) {
            const size_t colon = repo_id.find(':');
            if (colon != std::string::npos) {
                quant_tag = repo_id.substr(colon + 1);
                repo_id = repo_id.substr(0, colon);
            }
        }
        const bool repo_is_gguf = !repo_id.empty() &&
            (repo_id.ends_with("-GGUF") || !quant_tag.empty());
        std::filesystem::path model;
        if (!args.repo.empty()) {
            model = repo_is_gguf
                ? celeg::resolve_hf_gguf(
                      repo_id, quant_tag.empty() ? "Q4_K_M" : quant_tag)
                : celeg::resolve_hf_model(args.repo);
        } else {
            model = std::filesystem::path(args.model_dir);
        }
        const auto runtime = celeg::create_builtin_runtime_context();
        const celeg::detail::ModelBootstrap bootstrap =
            celeg::detail::load_model_bootstrap(model, *runtime);
        const auto& topology = bootstrap.model.topology;
        if (bootstrap.checkpoint.tokenizer && args.group_size_explicit) {
            throw std::runtime_error(
                "--cpu-q4-group is only valid for Safetensors checkpoints");
        }
        if (args.context > topology.max_position_embeddings) {
            throw std::runtime_error("--context exceeds model maximum");
        }
        const auto chat_catalog = celeg::make_chat_profile_catalog();
        const auto& chat_template = chat_catalog.find(bootstrap.model.chat_profile_id);
        const auto& tokenizer_provider = celeg::select_tokenizer_provider(
            *runtime, bootstrap.checkpoint, model);
        const auto tokenizer_storage = tokenizer_provider.create(
            bootstrap.checkpoint, model);
        const celeg::BpeTokenizer& tokenizer = *tokenizer_storage;
        std::vector<celeg::ChatMessage> chat_messages;
        if (!args.system.empty()) {
            chat_messages.push_back({celeg::ChatRole::System, args.system});
        }
        chat_messages.push_back({celeg::ChatRole::User, args.prompt});
        const std::string text = args.raw_prompt
            ? args.prompt : celeg::render_chat(chat_messages, chat_template);
        const std::vector<int32_t> input = tokenizer.encode(text, args.raw_prompt);
        if (static_cast<int>(input.size()) + args.max_new_tokens > args.context) {
            throw std::runtime_error("prompt plus output exceeds context");
        }
        celeg::CpuModelOptions options;
        options.isa = celeg::parse_cpu_isa(args.isa);
        options.weight_format = args.group_size == 64
            ? celeg::CpuWeightFormat::Q4Group64 : celeg::CpuWeightFormat::Q4Group32;
        options.threads = static_cast<size_t>(args.threads);
        options.affinity = celeg::parse_cpu_affinity(args.affinity);
        options.kv_cache_mode = celeg::parse_cpu_kv_cache_mode(args.kv_cache);
        options.kv_page_tokens = static_cast<size_t>(args.kv_page_tokens);
        options.prefill_chunk_tokens = static_cast<size_t>(args.prefill_chunk_tokens);
        options.prefill_chunk_threshold = static_cast<size_t>(args.prefill_chunk_threshold);
        options.attention_parallel_threshold = static_cast<size_t>(args.attention_parallel_threshold);
        options.attention_page_tile = static_cast<size_t>(args.attention_page_tile);
        options.numa_mode = celeg::parse_cpu_numa_mode(args.numa);
        options.use_pack_cache = !args.no_pack_cache;
        if (!args.pack_cache.empty()) options.pack_cache_directory = args.pack_cache;
        celeg::GenerationConfig generation;
        generation.temperature = args.temperature;
        generation.top_k = args.top_k;
        generation.top_p = args.top_p;
        generation.repetition_penalty = args.repetition_penalty;
        generation.seed = args.seed;
        celeg::CpuModel engine(model.string(), args.context, options, generation);
        std::cerr << "backend=" << engine.diagnostics().backend_description() << '\n';
        if (!engine.diagnostics().pack_path().empty()) std::cerr << "cpu.pack_path=" << engine.diagnostics().pack_path() << '\n';
        if (args.memory_report) {
            const auto stats = engine.diagnostics().memory_stats();
            std::cerr << "memory.weights=" << bytes(stats.weights) << '\n'
                      << "memory.kv_cache=" << bytes(stats.kv_cache) << '\n'
                      << "memory.conv_state=" << bytes(stats.conv_state) << '\n'
                      << "memory.activations=" << bytes(stats.activations) << '\n'
                      << "memory.kv_pages_used=" << stats.kv_pages_used << '\n'
                      << "memory.kv_pages_total=" << stats.kv_pages_total << '\n'
                      << "memory.total=" << bytes(stats.total()) << '\n';
        }
        engine.session().prefill(input);
        std::string pending;
        for (int i = 0; i < args.max_new_tokens; ++i) {
            const int32_t token = engine.session().decode();
            if (celeg::is_stop_token(topology.eos_token_ids, token)) break;
            pending += tokenizer.decode({token}, true);
            std::cout << pending << std::flush;
            pending.clear();
        }
        std::cout << '\n';
        const celeg::RuntimeMetrics metrics = engine.diagnostics().runtime_metrics();
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
