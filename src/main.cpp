#include "lfm/config.hpp"
#include "lfm/chat_template.hpp"
#include "lfm/downloader.hpp"
#include "lfm/model.hpp"
#include "lfm/model_shape.hpp"
#include "lfm/model_variant.hpp"
#include "lfm/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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
    std::string dump_logits;
    std::string save_session;
    std::string load_session;
    int max_new_tokens = 128;
    int context = 4096;
    int print_top = 0;
    int top_k = 50;
    float temperature = 0.1f;
    float top_p = 1.0f;
    float repetition_penalty = 1.05f;
    uint64_t seed = 1;
    int benchmark_decode = 0;
    int benchmark_warmup = 8;
    int lt_workspace_mb = 64;
    int lt_heuristics = 8;
    std::string gemm_backend = "cublas";
    std::string weight_mode = "bf16";
    std::string kv_cache_mode = "bf16";
    std::string attention_mode = "single";
    int attention_chunk_tokens = 256;
    int attention_auto_threshold = 4096;
    bool raw_prompt = false;
    bool fused_residuals = false;
    bool fast_attention = false;
    bool fused_projections = false;
    bool print_config = false;
    bool tokens_only = false;
    bool no_cuda_graph = false;
    bool legacy_prefill = false;
    bool legacy_sampling = false;
    bool lt_autotune = false;
    bool memory_report = false;
    bool runtime_metrics = false;
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
        else if (key == "--max-new-tokens") args.max_new_tokens = std::stoi(value());
        else if (key == "--context") args.context = std::stoi(value());
        else if (key == "--dump-logits") args.dump_logits = value();
        else if (key == "--save-session") args.save_session = value();
        else if (key == "--load-session") args.load_session = value();
        else if (key == "--print-top") args.print_top = std::stoi(value());
        else if (key == "--temperature") args.temperature = std::stof(value());
        else if (key == "--top-k") args.top_k = std::stoi(value());
        else if (key == "--top-p") args.top_p = std::stof(value());
        else if (key == "--repetition-penalty") args.repetition_penalty = std::stof(value());
        else if (key == "--seed") args.seed = static_cast<uint64_t>(std::stoull(value()));
        else if (key == "--benchmark-decode") args.benchmark_decode = std::stoi(value());
        else if (key == "--benchmark-warmup") args.benchmark_warmup = std::stoi(value());
        else if (key == "--gemm-backend") args.gemm_backend = value();
        else if (key == "--lt-workspace-mb") args.lt_workspace_mb = std::stoi(value());
        else if (key == "--lt-heuristics") args.lt_heuristics = std::stoi(value());
        else if (key == "--weight-mode") args.weight_mode = value();
        else if (key == "--kv-cache") args.kv_cache_mode = value();
        else if (key == "--attention-mode") args.attention_mode = value();
        else if (key == "--attention-chunk-tokens") args.attention_chunk_tokens = std::stoi(value());
        else if (key == "--attention-auto-threshold") args.attention_auto_threshold = std::stoi(value());
        else if (key == "--raw") args.raw_prompt = true;
        else if (key == "--fused-residuals") args.fused_residuals = true;
        else if (key == "--fast-attention") args.fast_attention = true;
        else if (key == "--fused-projections") args.fused_projections = true;
        else if (key == "--print-config") args.print_config = true;
        else if (key == "--tokens-only") args.tokens_only = true;
        else if (key == "--no-cuda-graph") args.no_cuda_graph = true;
        else if (key == "--legacy-prefill") args.legacy_prefill = true;
        else if (key == "--legacy-sampling") args.legacy_sampling = true;
        else if (key == "--lt-autotune") args.lt_autotune = true;
        else if (key == "--memory-report") args.memory_report = true;
        else if (key == "--runtime-metrics") args.runtime_metrics = true;
        else if (key == "--segmented-attention") args.attention_mode = "segmented";
        else if (key == "--help") {
            std::cout
                << "lfm25-run [--model DIR | --repo REPO_ID] [--prompt TEXT] [--system TEXT]\n"
                << "  [--max-new-tokens N] [--context N] [--raw]\n"
                << "  [--fused-residuals] [--fast-attention] [--fused-projections]\n"
                << "  [--print-config] [--tokens-only] [--legacy-prefill]\n"
                << "  [--legacy-sampling] [--no-cuda-graph]\n"
                << "  [--gemm-backend cublas|cublaslt] [--lt-autotune]\n"
                << "  [--lt-workspace-mb N] [--lt-heuristics N]\n"
                << "  [--weight-mode bf16|int8|int4] [--kv-cache bf16|int8]\n"
                << "  [--attention-mode single|segmented|auto]\n"
                << "  [--attention-chunk-tokens N] [--attention-auto-threshold N]\n"
                << "  [--memory-report] [--benchmark-decode N]\n"
                << "  [--benchmark-warmup N]\n"
                << "  [--temperature F] [--top-k N] [--top-p F]\n"
                << "  [--repetition-penalty F] [--seed N]\n"
                << "  [--dump-logits FILE.f32] [--print-top N]\n"
                << "  [--save-session FILE] [--load-session FILE]\n"
                << "  [--runtime-metrics]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.model_dir.empty() && args.repo.empty())
        throw std::runtime_error("--model or --repo is required");
    if (!args.model_dir.empty() && !args.repo.empty())
        throw std::runtime_error("--model and --repo are mutually exclusive");
    if (!args.print_config && args.prompt.empty() && args.load_session.empty()) {
        throw std::runtime_error(
            "--prompt is required unless --print-config or --load-session is used");
    }
    if (args.context <= 0 || args.max_new_tokens < 0 || args.print_top < 0 ||
        args.top_k <= 0 || args.temperature < 0.0f || args.top_p <= 0.0f ||
        args.top_p > 1.0f || args.repetition_penalty < 1.0f || args.seed == 0 ||
        args.benchmark_decode < 0 || args.benchmark_warmup < 0 ||
        args.lt_workspace_mb < 0 || args.lt_heuristics <= 0 ||
        args.lt_heuristics > 64 || args.attention_chunk_tokens <= 0 ||
        args.attention_auto_threshold <= 0) {
        throw std::runtime_error("invalid numeric argument");
    }
    if (args.gemm_backend != "cublas" && args.gemm_backend != "cublaslt") {
        throw std::runtime_error("--gemm-backend must be cublas or cublaslt");
    }
    if (args.weight_mode != "bf16" && args.weight_mode != "int8" &&
        args.weight_mode != "int4") {
        throw std::runtime_error("--weight-mode must be bf16, int8 or int4");
    }
    if (args.kv_cache_mode != "bf16" && args.kv_cache_mode != "int8") {
        throw std::runtime_error("--kv-cache must be bf16 or int8");
    }
    if (args.attention_mode != "single" && args.attention_mode != "segmented" &&
        args.attention_mode != "auto") {
        throw std::runtime_error(
            "--attention-mode must be single, segmented or auto");
    }
    if (args.attention_mode != "single" && !args.fast_attention) {
        throw std::runtime_error(
            "segmented or automatic attention requires --fast-attention");
    }
    return args;
}

void dump_logits_file(const std::string& path, const std::vector<float>& logits) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create logits file: " + path);
    out.write(reinterpret_cast<const char*>(logits.data()),
              static_cast<std::streamsize>(logits.size() * sizeof(float)));
    if (!out) throw std::runtime_error("failed writing logits file: " + path);
}

void print_top_logits(const std::vector<float>& logits, int count) {
    count = std::min(count, static_cast<int>(logits.size()));
    std::vector<int32_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + count, indices.end(),
        [&](int32_t a, int32_t b) {
            if (logits[static_cast<size_t>(a)] != logits[static_cast<size_t>(b)]) {
                return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)];
            }
            return a < b;
        });
    for (int i = 0; i < count; ++i) {
        const int32_t token = indices[static_cast<size_t>(i)];
        std::cerr << "top[" << i << "] token=" << token
                  << " logit=" << logits[static_cast<size_t>(token)] << '\n';
    }
}

std::string format_bytes(size_t bytes) {
    static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 3) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 2)
        << value << ' ' << units[unit];
    return out.str();
}

void print_memory_stats(const lfm::ModelMemoryStats& stats) {
    std::cerr << "memory.weights=" << format_bytes(stats.weights) << '\n'
              << "memory.kv_cache=" << format_bytes(stats.kv_cache) << '\n'
              << "memory.conv_state=" << format_bytes(stats.conv_state) << '\n'
              << "memory.rope_tables=" << format_bytes(stats.rope_tables) << '\n'
              << "memory.activations=" << format_bytes(stats.activations) << '\n'
              << "memory.sampling=" << format_bytes(stats.sampling) << '\n'
              << "memory.matmul_workspace=" << format_bytes(stats.matmul_workspace) << '\n'
              << "memory.attention_workspace=" << format_bytes(stats.attention_workspace) << '\n'
              << "memory.total=" << format_bytes(stats.total()) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        std::filesystem::path model;
        if (!args.repo.empty()) {
            model = lfm::resolve_hf_model(args.repo);
        } else {
            model = std::filesystem::path(args.model_dir);
        }
        const lfm::ModelConfig config =
            lfm::ModelConfig::load((model / "config.json").string());
        const lfm::ModelShape shape = lfm::ModelShape::from_config(config);
        lfm::register_builtin_variants();
        const lfm::IModelVariant& variant =
            lfm::ModelVariantRegistry::instance().select(shape, config.repo_hint);
        (void)variant;
        if (args.context > config.max_position_embeddings) {
            throw std::runtime_error("--context exceeds max_position_embeddings in config.json");
        }
        if (args.print_config) {
            std::cout << config.summary() << '\n';
            return 0;
        }

        lfm::BpeTokenizer tokenizer((model / "tokenizer.json").string(),
            lfm::make_chat_template(variant.chat_template_kind()));
        if (tokenizer.bos_id() != config.bos_token_id ||
            tokenizer.eos_id() != config.eos_token_id) {
            throw std::runtime_error("tokenizer special IDs disagree with config.json");
        }

        std::vector<int32_t> input;
        if (args.load_session.empty()) {
            const std::string formatted = args.raw_prompt
                ? args.prompt : tokenizer.format_chat(args.prompt, args.system);
            // Chat formatting contains <|startoftext|>; raw prompts receive BOS automatically.
            input = tokenizer.encode(formatted, args.raw_prompt);
            if (args.tokens_only) {
                for (size_t i = 0; i < input.size(); ++i) {
                    if (i) std::cout << ' ';
                    std::cout << input[i];
                }
                std::cout << '\n';
                return 0;
            }
            if (static_cast<int>(input.size()) + args.max_new_tokens > args.context) {
                throw std::runtime_error("prompt plus output exceeds --context");
            }
        } else if (args.tokens_only) {
            throw std::runtime_error("--tokens-only cannot be used with --load-session");
        }

        lfm::ModelOptions model_options;
        model_options.fused_residuals = args.fused_residuals;
        model_options.fast_attention = args.fast_attention;
        model_options.fused_projections = args.fused_projections;
        model_options.fused_sampling = !args.legacy_sampling;
        model_options.cuda_graph = !args.no_cuda_graph;
        model_options.legacy_prefill = args.legacy_prefill;
        model_options.gemm_backend = args.gemm_backend == "cublaslt"
            ? lfm::GemmBackend::CublasLt
            : lfm::GemmBackend::Cublas;
        model_options.lt_workspace_bytes =
            static_cast<size_t>(args.lt_workspace_mb) * 1024ULL * 1024ULL;
        model_options.lt_heuristics = args.lt_heuristics;
        model_options.lt_autotune = args.lt_autotune;
        if (args.weight_mode == "int8") {
            model_options.weight_mode = lfm::WeightMode::Int8;
        } else if (args.weight_mode == "int4") {
            model_options.weight_mode = lfm::WeightMode::Int4;
        } else {
            model_options.weight_mode = lfm::WeightMode::Bf16;
        }
        model_options.kv_cache_mode = args.kv_cache_mode == "int8"
            ? lfm::KvCacheMode::Int8 : lfm::KvCacheMode::Bf16;
        if (args.attention_mode == "segmented") {
            model_options.attention_mode = lfm::AttentionMode::Segmented;
        } else if (args.attention_mode == "auto") {
            model_options.attention_mode = lfm::AttentionMode::Auto;
        } else {
            model_options.attention_mode = lfm::AttentionMode::Single;
        }
        model_options.attention_chunk_tokens = args.attention_chunk_tokens;
        model_options.attention_auto_threshold = args.attention_auto_threshold;
        lfm::GenerationConfig generation;
        generation.temperature = args.temperature;
        generation.top_k = args.top_k;
        generation.top_p = args.top_p;
        generation.repetition_penalty = args.repetition_penalty;
        generation.seed = args.seed;
        lfm::LfmModel engine(
            (model / "model.safetensors").string(), args.context,
            model_options, generation);
        if (args.memory_report) print_memory_stats(engine.memory_stats());

        if (!args.load_session.empty()) {
            engine.load_session(args.load_session);
            if (engine.position() + args.max_new_tokens > args.context) {
                throw std::runtime_error("loaded session plus output exceeds --context");
            }
        } else {
            engine.prefill(input);
        }
        if (args.benchmark_decode > 0) {
            const lfm::RuntimeMetrics runtime = engine.runtime_metrics();
            const double prefill_ms = runtime.last_prefill_ms;
            const double prefill_tps = runtime.prefill_tokens_per_second();
            std::cerr << std::fixed << std::setprecision(3)
                      << "benchmark.prefill_tokens=" << runtime.prefill_tokens << '\n'
                      << "benchmark.prefill_ms=" << prefill_ms << '\n'
                      << "benchmark.prefill_tokens_per_second=" << prefill_tps << '\n';
            const lfm::DecodeBenchmark benchmark = engine.benchmark_decode(
                args.benchmark_warmup, args.benchmark_decode);
            std::cerr << "benchmark.decode_warmup=" << benchmark.warmup_steps << '\n'
                      << "benchmark.decode_tokens=" << benchmark.measured_steps << '\n'
                      << "benchmark.decode_ms=" << benchmark.elapsed_ms << '\n'
                      << "benchmark.decode_ms_per_token="
                      << benchmark.milliseconds_per_token() << '\n'
                      << "benchmark.decode_tokens_per_second="
                      << benchmark.tokens_per_second() << '\n';
            return 0;
        }

        if (!args.dump_logits.empty() || args.print_top > 0) {
            const std::vector<float> logits = engine.copy_logits();
            if (!args.dump_logits.empty()) dump_logits_file(args.dump_logits, logits);
            if (args.print_top > 0) print_top_logits(logits, args.print_top);
        }

        std::vector<int32_t> generated;
        generated.reserve(static_cast<size_t>(args.max_new_tokens));
        for (int i = 0; i < args.max_new_tokens; ++i) {
            const int32_t next = engine.decode();
            if (next == tokenizer.eos_id()) break;
            generated.push_back(next);
            std::cout << tokenizer.decode({next}, true) << std::flush;
        }
        std::cout << '\n';
        if (!args.save_session.empty()) engine.save_session(args.save_session);
        if (args.runtime_metrics) {
            const lfm::RuntimeMetrics runtime = engine.runtime_metrics();
            std::cerr << std::fixed << std::setprecision(3)
                      << "runtime.prefill_tokens=" << runtime.prefill_tokens << '\n'
                      << "runtime.prefill_ms=" << runtime.last_prefill_ms << '\n'
                      << "runtime.prefill_tokens_per_second="
                      << runtime.prefill_tokens_per_second() << '\n'
                      << "runtime.decode_tokens=" << runtime.decoded_tokens << '\n'
                      << "runtime.decode_ms=" << runtime.cumulative_decode_ms << '\n'
                      << "runtime.decode_tokens_per_second="
                      << runtime.decode_tokens_per_second() << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
