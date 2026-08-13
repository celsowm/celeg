#include "celeg/text/chat_template.hpp"
#include "celeg/checkpoint/downloader.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/runtime/request_types.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/backend/cuda/model.hpp"
#include "celeg/text/tokenizer.hpp"
#include "celeg/runtime/context.hpp"

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

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

// The NVIDIA CUDA driver/runtime DLLs (nvcuda.dll, cudart64_*.dll,
// cublasLt64_*.dll) have exit-time DLL_PROCESS_DETACH teardown that is prone
// to an intermittent access violation / heap corruption on Windows once the
// CUDA context is torn down (this reproduces on unmodified master, with
// --no-cuda-graph, and with CUDA_MODULE_LOADING=EAGER, so it is not
// celeg-specific state; it happens after all real work — model teardown,
// generation, printing — is already complete). ExitProcess() still runs
// every attached DLL's DLL_PROCESS_DETACH handler, so even std::_Exit()
// hits the same crash; only TerminateProcess on our own process skips DLL
// notifications entirely and avoids it. Everything meaningful (stdout,
// session files) must be flushed/closed before this is called.
[[noreturn]] void exit_process_immediately(int code) {
    std::cout.flush();
    std::cerr.flush();
#if defined(_WIN32)
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code));
#endif
    std::exit(code);
}

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
    int top_k = 1;
    float temperature = 0.1f;
    float top_p = 1.0f;
    float repetition_penalty = 1.05f;
    uint64_t seed = 1;
    int benchmark_decode = 0;
    int benchmark_warmup = 8;
    int benchmark_prefill_tokens = 0;
    int lt_workspace_mb = 64;
    int lt_heuristics = 8;
    std::string gemm_backend = "cublas";
    std::string weight_mode = "auto";
    std::string kv_cache_mode = "bf16";
    std::string attention_mode = "auto";
    int attention_chunk_tokens = 32;
    int attention_auto_threshold = 1;
    bool raw_prompt = false;
    bool fused_residuals = true;
    bool fast_attention = true;
    bool fused_projections = true;
    bool print_config = false;
    bool tokens_only = false;
    bool no_cuda_graph = false;
    bool lt_autotune = false;
    bool memory_report = false;
    bool runtime_metrics = false;
    bool enable_mtp = false;
    int mtp_speculative_tokens = 1;
    std::string expert_offload = "none";
    std::string expert_host_mode = "mapped";
    std::string expert_cache_policy = "lfu-lru";
    int expert_cache_mib = 0;
    int expert_cache_per_layer = 0;
    int maximum_pinned_host_mib = 9216;
    int gpu_memory_reserve_mib = 768;
    int prefill_chunk = 256;
    std::string expert_backing = "host";
    int expert_host_cache_mib = 4096;
    std::string expert_io_backend = "auto";
    int expert_io_workers = 4;
    int expert_io_queue_depth = 16;
    std::string expert_sidecar;
    std::string expert_usage_profile;
    bool expert_direct_io = false;
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
        else if (key == "--benchmark-prefill-tokens") args.benchmark_prefill_tokens = std::stoi(value());
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
        else if (key == "--lt-autotune") args.lt_autotune = true;
        else if (key == "--memory-report") args.memory_report = true;
        else if (key == "--runtime-metrics") args.runtime_metrics = true;
        else if (key == "--mtp") args.enable_mtp = true;
        else if (key == "--mtp-speculative-tokens") args.mtp_speculative_tokens = std::stoi(value());
        else if (key == "--expert-offload") args.expert_offload = value();
        else if (key == "--expert-host-mode") args.expert_host_mode = value();
        else if (key == "--expert-cache-policy") args.expert_cache_policy = value();
        else if (key == "--expert-cache-mib") args.expert_cache_mib = std::stoi(value());
        else if (key == "--expert-cache-per-layer") args.expert_cache_per_layer = std::stoi(value());
        else if (key == "--maximum-pinned-host-mib") args.maximum_pinned_host_mib = std::stoi(value());
        else if (key == "--gpu-memory-reserve-mib") args.gpu_memory_reserve_mib = std::stoi(value());
        else if (key == "--prefill-chunk") args.prefill_chunk = std::stoi(value());
        else if (key == "--expert-backing") args.expert_backing = value();
        else if (key == "--expert-host-cache-mib") args.expert_host_cache_mib = std::stoi(value());
        else if (key == "--expert-io-backend") args.expert_io_backend = value();
        else if (key == "--expert-io-workers") args.expert_io_workers = std::stoi(value());
        else if (key == "--expert-io-queue-depth") args.expert_io_queue_depth = std::stoi(value());
        else if (key == "--expert-sidecar") args.expert_sidecar = value();
        else if (key == "--expert-usage-profile") args.expert_usage_profile = value();
        else if (key == "--expert-direct-io") args.expert_direct_io = true;
        else if (key == "--help") {
            std::cout
                << "celeg-run [--model DIR | --repo REPO_ID] [--prompt TEXT] [--system TEXT]\n"
                << "  [--max-new-tokens N] [--context N] [--raw]\n"
                << "  [--fused-residuals] [--fast-attention] [--fused-projections]\n"
                << "  [--print-config] [--tokens-only] [--no-cuda-graph]\n"
                << "  [--gemm-backend cublas|cublaslt] [--lt-autotune]\n"
                << "  [--lt-workspace-mb N] [--lt-heuristics N]\n"
                << "  [--weight-mode auto|bf16|int8|int4|native] [--kv-cache bf16|int8]\n"
                << "  [--attention-mode single|segmented|auto]\n"
                << "  [--attention-chunk-tokens N] [--attention-auto-threshold N]\n"
                << "  [--memory-report] [--benchmark-decode N]\n"
                << "  [--benchmark-warmup N]\n"
                << "  [--benchmark-prefill-tokens N]\n"
                << "  [--temperature F] [--top-k N] [--top-p F]\n"
                << "  [--repetition-penalty F] [--seed N]\n"
                << "  [--dump-logits FILE.f32] [--print-top N]\n"
                << "  [--save-session FILE] [--load-session FILE]\n"
                << "  [--runtime-metrics]\n"
                << "  [--mtp] [--mtp-speculative-tokens N]\n"
                << "  [--expert-offload none|auto|host]\n"
                << "  [--expert-host-mode mapped|pinned-copy|staged]\n"
                << "  [--expert-cache-policy static|lru|lfu-lru]\n"
                << "  [--expert-cache-mib N] [--expert-cache-per-layer N]\n"
                << "  [--maximum-pinned-host-mib N] [--gpu-memory-reserve-mib N]\n"
                << "  [--prefill-chunk N]\n"
                << "  [--expert-backing host|disk]\n"
                << "  [--expert-host-cache-mib N]\n"
                << "  [--expert-io-backend auto|thread-pool|io-uring|overlapped]\n"
                << "  [--expert-io-workers N]\n"
                << "  [--expert-io-queue-depth N]\n"
                << "  [--expert-sidecar PATH]\n"
                << "  [--expert-usage-profile PATH]\n"
                << "  [--expert-direct-io]\n";
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
        args.benchmark_prefill_tokens < 0 ||
        args.lt_workspace_mb < 0 || args.lt_heuristics <= 0 ||
        args.lt_heuristics > 64 || args.attention_chunk_tokens <= 0 ||
        args.attention_auto_threshold <= 0) {
        throw std::runtime_error("invalid numeric argument");
    }
    if (args.gemm_backend != "cublas" && args.gemm_backend != "cublaslt") {
        throw std::runtime_error("--gemm-backend must be cublas or cublaslt");
    }
    if (args.weight_mode != "auto" && args.weight_mode != "bf16" &&
        args.weight_mode != "int8" && args.weight_mode != "int4" &&
        args.weight_mode != "native") {
        throw std::runtime_error("--weight-mode must be auto, bf16, int8, int4 or native");
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
    if (args.expert_offload != "none" && args.expert_offload != "auto" &&
        args.expert_offload != "host") {
        throw std::runtime_error(
            "--expert-offload must be none, auto or host");
    }
    if (args.expert_host_mode != "mapped" &&
        args.expert_host_mode != "pinned-copy" &&
        args.expert_host_mode != "staged") {
        throw std::runtime_error(
            "--expert-host-mode must be mapped, pinned-copy or staged");
    }
    if (args.expert_cache_policy != "static" && args.expert_cache_policy != "lru" &&
        args.expert_cache_policy != "lfu-lru") {
        throw std::runtime_error(
            "--expert-cache-policy must be static, lru or lfu-lru");
    }
    if (args.expert_cache_mib < 0 || args.expert_cache_per_layer < 0 ||
        args.maximum_pinned_host_mib < 0 || args.gpu_memory_reserve_mib < 0 ||
        args.prefill_chunk <= 0) {
        throw std::runtime_error("invalid numeric argument");
    }
    if (args.expert_backing != "host" && args.expert_backing != "disk") {
        throw std::runtime_error("--expert-backing must be host or disk");
    }
    if (args.expert_io_backend != "auto" && args.expert_io_backend != "thread-pool" &&
        args.expert_io_backend != "io-uring" && args.expert_io_backend != "overlapped") {
        throw std::runtime_error("--expert-io-backend must be auto, thread-pool, io-uring or overlapped");
    }
    if (args.expert_host_cache_mib < 0 || args.expert_io_workers <= 0 || args.expert_io_queue_depth <= 0) {
        throw std::runtime_error("invalid numeric argument");
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

void print_memory_stats(const celeg::ModelMemoryStats& stats) {
    std::cerr << "memory.weights=" << format_bytes(stats.weights) << '\n'
              << "memory.kv_cache=" << format_bytes(stats.kv_cache) << '\n'
              << "memory.conv_state=" << format_bytes(stats.conv_state) << '\n'
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
        // Split an optional `:QUANT_TAG` suffix off the --repo argument so the
        // CLI can select a specific GGUF shard (e.g. `...LFM2.5-230M-GGUF:Q4_K_M`).
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
        const bool direct_gguf = !args.model_dir.empty() &&
            std::filesystem::path(args.model_dir).extension() == ".gguf";
        const bool is_gguf = repo_is_gguf || direct_gguf;

        std::filesystem::path model;        // checkpoint dir (safetensors)
        std::filesystem::path gguf_path;    // concrete .gguf file (GGUF)
        if (repo_is_gguf) {
            gguf_path = celeg::resolve_hf_gguf(repo_id,
                quant_tag.empty() ? "Q4_K_M" : quant_tag);
        } else if (!args.repo.empty()) {
            model = celeg::resolve_hf_model(args.repo);
        } else {
            model = std::filesystem::path(args.model_dir);
            if (direct_gguf) gguf_path = model;
        }
        const auto runtime = celeg::create_builtin_runtime_context();
        const celeg::detail::ModelBootstrap bootstrap =
            celeg::detail::load_model_bootstrap(is_gguf ? gguf_path : model, *runtime);
        const auto& topology = bootstrap.model.topology;
        if (args.context > topology.dims.max_position_embeddings) {
            throw std::runtime_error("--context exceeds max_position_embeddings");
        }
        if (args.print_config) {
            std::cout << topology.summary() << '\n';
            exit_process_immediately(0);
        }

        const auto chat_catalog = celeg::make_chat_template_catalog();
        const celeg::IChatTemplate* chat_template = nullptr;
        if (!args.raw_prompt) {
            chat_template = &chat_catalog.find(bootstrap.model.provenance.chat_template_id);
        }
        const auto& tokenizer_provider = celeg::select_tokenizer_provider(
            *runtime, bootstrap.checkpoint, is_gguf ? gguf_path : model);
        const auto tokenizer_storage = tokenizer_provider.create(
            bootstrap.checkpoint, is_gguf ? gguf_path : model);
        const celeg::ITokenizer& tokenizer = *tokenizer_storage;
        // A JSON checkpoint may omit bos_token_id while its tokenizer config
        // still names a non-zero BOS token.  BOS is not injected when the
        // chat template requests add_bos=false, so the unresolved zero is not
        // a runtime disagreement; EOS remains a hard generation boundary.
        if ((topology.dims.token_policy.bos_token_id != 0 &&
             tokenizer.bos_id() != topology.dims.token_policy.bos_token_id) ||
            !celeg::is_stop_token(topology.dims.token_policy.eos_token_ids, tokenizer.eos_id())) {
            throw std::runtime_error("tokenizer special IDs disagree with config: bos=" +
                std::to_string(tokenizer.bos_id()) + "/" +
                std::to_string(topology.dims.token_policy.bos_token_id) + " eos=" +
                std::to_string(tokenizer.eos_id()));
        }

        std::vector<int32_t> input;
        if (args.load_session.empty()) {
            std::vector<celeg::ChatMessage> chat_messages;
            if (!args.system.empty()) {
                chat_messages.push_back({celeg::ChatRole::System, args.system});
            }
            chat_messages.push_back({celeg::ChatRole::User, args.prompt});
            const std::string formatted = args.raw_prompt
                ? args.prompt : celeg::render_chat(chat_messages, *chat_template);
            // Chat formatting contains <|startoftext|>; raw prompts receive BOS automatically.
            input = tokenizer.encode(formatted, args.raw_prompt);
            if (args.benchmark_prefill_tokens > 0) {
                if (input.empty()) throw std::runtime_error("benchmark prompt produced no tokens");
                if (static_cast<int>(input.size()) > args.benchmark_prefill_tokens) {
                    input.resize(static_cast<size_t>(args.benchmark_prefill_tokens));
                } else {
                    const int32_t fill = input.back();
                    input.resize(static_cast<size_t>(args.benchmark_prefill_tokens), fill);
                }
            }
            if (args.tokens_only) {
                for (size_t i = 0; i < input.size(); ++i) {
                    if (i) std::cout << ' ';
                    std::cout << input[i];
                }
                std::cout << '\n';
                exit_process_immediately(0);
            }
            if (static_cast<int>(input.size()) + args.max_new_tokens > args.context) {
                throw std::runtime_error("prompt plus output exceeds --context");
            }
        } else if (args.tokens_only) {
            throw std::runtime_error("--tokens-only cannot be used with --load-session");
        }

        celeg::CudaModelOptions model_options;
        model_options.fused_residuals = args.fused_residuals;
        model_options.fast_attention = args.fast_attention;
        model_options.fused_projections = args.fused_projections;
        model_options.cuda_graph = !args.no_cuda_graph;
        model_options.enable_mtp = args.enable_mtp;
        model_options.mtp_speculative_tokens = args.mtp_speculative_tokens;
        model_options.gemm_backend = args.gemm_backend == "cublaslt"
            ? celeg::GemmBackend::CublasLt
            : celeg::GemmBackend::Cublas;
        model_options.lt_workspace_bytes =
            static_cast<size_t>(args.lt_workspace_mb) * 1024ULL * 1024ULL;
        model_options.lt_heuristics = args.lt_heuristics;
        model_options.lt_autotune = args.lt_autotune;
        if (args.weight_mode == "auto") {
            // GGUF checkpoints benefit from INT8 re-quantization: decode reads
            // 2x less weight traffic than BF16 while prefill falls back to
            // BF16 cuBLAS tensor-core GEMM via the kept BF16 device buffer.
            model_options.weight_mode = is_gguf
                ? celeg::WeightMode::Int8
                : celeg::WeightMode::Bf16;
        } else if (args.weight_mode == "int8") {
            model_options.weight_mode = celeg::WeightMode::Int8;
        } else if (args.weight_mode == "int4") {
            model_options.weight_mode = celeg::WeightMode::Int4;
        } else if (args.weight_mode == "native") {
            model_options.weight_mode = celeg::WeightMode::NativeGguf;
        } else {
            model_options.weight_mode = celeg::WeightMode::Bf16;
        }
        model_options.kv_cache_mode = args.kv_cache_mode == "int8"
            ? celeg::KvCacheMode::Int8 : celeg::KvCacheMode::Bf16;
        if (args.attention_mode == "segmented") {
            model_options.attention_mode = celeg::AttentionMode::Segmented;
        } else if (args.attention_mode == "auto") {
            model_options.attention_mode = celeg::AttentionMode::Auto;
        } else {
            model_options.attention_mode = celeg::AttentionMode::Single;
        }
        model_options.attention_chunk_tokens = args.attention_chunk_tokens;
        model_options.attention_auto_threshold = args.attention_auto_threshold;
        {
            celeg::ExpertOffloadOptions& off = model_options.expert_offload;
            if (args.expert_offload == "auto") {
                off.mode = celeg::ExpertOffloadMode::Auto;
            } else if (args.expert_offload == "host") {
                off.mode = celeg::ExpertOffloadMode::Host;
            } else {
                off.mode = celeg::ExpertOffloadMode::None;
            }
            if (args.expert_host_mode == "pinned-copy") {
                off.host_mode = celeg::ExpertHostMode::PinnedCopy;
            } else if (args.expert_host_mode == "staged") {
                off.host_mode = celeg::ExpertHostMode::Staged;
            } else {
                off.host_mode = celeg::ExpertHostMode::Mapped;
            }
            if (args.expert_cache_policy == "static") {
                off.policy = celeg::ExpertCachePolicy::Static;
            } else if (args.expert_cache_policy == "lru") {
                off.policy = celeg::ExpertCachePolicy::Lru;
            } else {
                off.policy = celeg::ExpertCachePolicy::LayerLocalLfuLru;
            }
            off.gpu_expert_cache_bytes =
                static_cast<size_t>(args.expert_cache_mib) * 1024ULL * 1024ULL;
            off.experts_per_layer = args.expert_cache_per_layer;
            off.maximum_pinned_host_bytes =
                static_cast<size_t>(args.maximum_pinned_host_mib) * 1024ULL * 1024ULL;
            off.gpu_memory_reserve_bytes =
                static_cast<size_t>(args.gpu_memory_reserve_mib) * 1024ULL * 1024ULL;
            off.prefill_chunk_tokens = args.prefill_chunk;

            if (args.expert_backing == "disk") {
                off.backing = celeg::ExpertBackingMode::DiskCached;
            } else {
                off.backing = celeg::ExpertBackingMode::HostResident;
            }
            off.host_expert_cache_bytes =
                static_cast<size_t>(args.expert_host_cache_mib) * 1024ULL * 1024ULL;
            if (args.expert_io_backend == "thread-pool") {
                off.io_backend = celeg::ExpertIoBackend::ThreadPool;
            } else if (args.expert_io_backend == "io-uring") {
                off.io_backend = celeg::ExpertIoBackend::IoUring;
            } else if (args.expert_io_backend == "overlapped") {
                off.io_backend = celeg::ExpertIoBackend::WindowsOverlapped;
            } else {
                off.io_backend = celeg::ExpertIoBackend::Auto;
            }
            off.io_workers = args.expert_io_workers;
            off.io_queue_depth = args.expert_io_queue_depth;
            off.expert_sidecar_path = args.expert_sidecar;
            off.usage_profile_path = args.expert_usage_profile;
            off.direct_io = args.expert_direct_io;
        }
        celeg::GenerationConfig generation;
        generation.temperature = args.temperature;
        generation.top_k = args.top_k;
        generation.top_p = args.top_p;
        generation.repetition_penalty = args.repetition_penalty;
        generation.seed = args.seed;
        // Single-file checkpoints ship model.safetensors; sharded checkpoints
        // ship model.safetensors.index.json in the same directory. The model
        // constructor resolves the index when given the directory root. GGUF
        // checkpoints pass the concrete .gguf file path directly.
        const std::string model_path =
            is_gguf ? gguf_path.string()
                    : [&] {
                          const std::filesystem::path single =
                              model / "model.safetensors";
                          const std::filesystem::path index =
                              model / "model.safetensors.index.json";
                          return std::filesystem::is_regular_file(single)
                                     ? single.string()
                                     : std::filesystem::is_regular_file(index)
                                           ? model.string()
                                           : single.string();
                      }();
        celeg::CudaModel engine(
            model_path, args.context,
            model_options, generation);
        if (is_gguf) {
            std::cerr << "source=gguf(q4_k,q6_k)\n"
                      << "weight_mode=" << args.weight_mode << "\n"
                      << "pack_path=none\n";
        }
        if (args.memory_report) print_memory_stats(engine.diagnostics().memory_stats());

        if (!args.load_session.empty()) {
            engine.persistence().load_session(args.load_session);
            if (engine.session().position() + args.max_new_tokens > args.context) {
                throw std::runtime_error("loaded session plus output exceeds --context");
            }
        } else {
            if (args.benchmark_prefill_tokens > 0 && args.lt_autotune) {
                std::vector<int32_t> warmup(
                    static_cast<size_t>(args.benchmark_prefill_tokens), input[0]);
                engine.session().prefill(warmup);
            }
            engine.session().prefill(input);
        }
        if (args.benchmark_decode > 0) {
            const celeg::RuntimeMetrics runtime = engine.diagnostics().runtime_metrics();
            const double prefill_ms = runtime.last_prefill_ms;
            const double prefill_tps = runtime.prefill_tokens_per_second();
            std::cerr << std::fixed << std::setprecision(3)
                      << "benchmark.prefill_tokens=" << runtime.prefill_tokens << '\n'
                      << "benchmark.prefill_ms=" << prefill_ms << '\n'
                      << "benchmark.prefill_tokens_per_second=" << prefill_tps << '\n';
            const celeg::DecodeBenchmark benchmark = engine.diagnostics().benchmark_decode(
                args.benchmark_warmup, args.benchmark_decode);
            std::cerr << "benchmark.decode_warmup=" << benchmark.warmup_steps << '\n'
                      << "benchmark.decode_tokens=" << benchmark.measured_steps << '\n'
                      << "benchmark.decode_ms=" << benchmark.elapsed_ms << '\n'
                      << "benchmark.decode_ms_per_token="
                      << benchmark.milliseconds_per_token() << '\n'
                      << "benchmark.decode_tokens_per_second="
                      << benchmark.tokens_per_second() << '\n';

        const celeg::CudaModelDiagnostics::ExpertOffloadStats off =
                engine.diagnostics().expert_offload_stats();
            if (off.hit_rate >= 0.0) {
                std::cerr << "expert_offload.experts_per_layer="
                          << off.experts_per_layer << '\n'
                          << "expert_offload.host_experts_per_layer="
                          << off.host_experts_per_layer << '\n'
                          << "expert_offload.hits=" << off.hits << '\n'
                          << "expert_offload.misses=" << off.misses << '\n'
                          << "expert_offload.hit_rate=" << off.hit_rate << '\n';
            }
            exit_process_immediately(0);
        }

        if (!args.dump_logits.empty() || args.print_top > 0) {
            const std::vector<float> logits = engine.diagnostics().copy_logits();
            if (!args.dump_logits.empty()) dump_logits_file(args.dump_logits, logits);
            if (args.print_top > 0) print_top_logits(logits, args.print_top);
        }

        std::vector<int32_t> generated;
        generated.reserve(static_cast<size_t>(args.max_new_tokens));
        for (int i = 0; i < args.max_new_tokens; ++i) {
            const int32_t next = engine.session().decode();
            if (celeg::is_stop_token(topology.dims.token_policy.eos_token_ids, next)) break;
            generated.push_back(next);
            std::cout << tokenizer.decode({next}, true) << std::flush;
        }
        std::cout << '\n';
        if (!args.save_session.empty()) engine.persistence().save_session(args.save_session);
        if (args.runtime_metrics) {
            const celeg::RuntimeMetrics runtime = engine.diagnostics().runtime_metrics();
            std::cerr << std::fixed << std::setprecision(3)
                      << "runtime.prefill_tokens=" << runtime.prefill_tokens << '\n'
                      << "runtime.prefill_ms=" << runtime.last_prefill_ms << '\n'
                      << "runtime.prefill_tokens_per_second="
                      << runtime.prefill_tokens_per_second() << '\n'
                      << "runtime.decode_tokens=" << runtime.decoded_tokens << '\n'
                      << "runtime.decode_ms=" << runtime.cumulative_decode_ms << '\n'
                      << "runtime.decode_tokens_per_second="
                      << runtime.decode_tokens_per_second() << '\n'
                      << "runtime.mtp_forward_tokens=" << runtime.mtp_forward_tokens << '\n'
                      << "runtime.mtp_verified_tokens=" << runtime.mtp_verified_tokens << '\n'
                      << "runtime.mtp_accepted_tokens=" << runtime.mtp_accepted_tokens << '\n'
                      << "runtime.mtp_rejected_tokens=" << runtime.mtp_rejected_tokens << '\n'
                      << "runtime.mtp_used_tokens=" << runtime.mtp_used_tokens << '\n';
        }
        exit_process_immediately(0);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        exit_process_immediately(1);
    }
}
