#include "App.h"

#include "celeg/app/run_preparation.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/checkpoint/downloader.hpp"
#include "celeg/serve/cpu_inference_service.hpp"
#ifdef CELEG_SERVE_CUDA
#include "celeg/backend/cuda/cuda_inference_service.hpp"
#endif
#include "celeg/serve/generation_dispatcher.hpp"
#include "celeg/text/tokenizer.hpp"
#include "celeg/runtime/context.hpp"
#include "routes/chat_completions.hpp"
#include "routes/chat_ui.hpp"
#include "routes/docs.hpp"
#include "routes/health.hpp"
#include "routes/models.hpp"
#include "routes/tokenize.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {

using celeg::serve::GenerationDispatcher;

struct Args {
    std::string model_dir;
    std::string chat_template_file;
    std::string served_model_name;
    int port = 8080;
    std::string host = "127.0.0.1";
    int context = 4096;
    int threads = 0;
    int max_active_requests = 1;
    int max_batched_tokens = 256;
    int prefill_chunk_tokens = 256;
    std::string backend = "cpu";
    std::string repo;
    std::string quant = "Q4_K_M";
    std::string format = "auto";
    std::string expert_offload = "none";
    std::string expert_cache_policy = "lru";
    std::string expert_backing = "host";
    int expert_cache_per_layer = 0;
    int expert_host_cache_mib = 4096;
    bool enable_mtp = false;
    int mtp_speculative_tokens = 1;
    bool cuda_graph = true;
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
        else if (key == "--chat-template-file") args.chat_template_file = value();
        else if (key == "--served-model-name") args.served_model_name = value();
        else if (key == "--port") args.port = std::stoi(value());
        else if (key == "--host") args.host = value();
        else if (key == "--context") args.context = std::stoi(value());
        else if (key == "--threads") args.threads = std::stoi(value());
        else if (key == "--max-active-requests") args.max_active_requests = std::stoi(value());
        else if (key == "--max-batched-tokens") args.max_batched_tokens = std::stoi(value());
        else if (key == "--prefill-chunk-tokens") args.prefill_chunk_tokens = std::stoi(value());
        else if (key == "--backend") args.backend = value();
        else if (key == "--repo") args.repo = value();
        else if (key == "--quant") args.quant = value();
        else if (key == "--format") args.format = value();
        else if (key == "--expert-offload") args.expert_offload = value();
        else if (key == "--expert-cache-policy") args.expert_cache_policy = value();
        else if (key == "--expert-backing") args.expert_backing = value();
        else if (key == "--expert-cache-per-layer") args.expert_cache_per_layer = std::stoi(value());
        else if (key == "--expert-host-cache-mib") args.expert_host_cache_mib = std::stoi(value());
        else if (key == "--mtp") args.enable_mtp = true;
        else if (key == "--mtp-speculative-tokens") args.mtp_speculative_tokens = std::stoi(value());
        else if (key == "--no-cuda-graph") args.cuda_graph = false;
        else if (key == "--help") {
            std::cout << "celeg-serve (--model PATH | --repo HF_REPO | HF_REPO_OR_PATH) [--host 127.0.0.1] [--port 8080] [--context 4096] "
                         "[--threads N] [--max-active-requests N] [--max-batched-tokens N] "
                         "[--prefill-chunk-tokens N] [--backend cpu|cuda] "
                         "[--format auto|safetensors|gguf] [--quant TAG] "
                         "[--expert-offload none|auto|host] [--expert-cache-policy static|lru|lfu-lru] "
                         "[--expert-backing host|disk] [--expert-cache-per-layer N] "
                         "[--expert-host-cache-mib N] "
                         "[--mtp] [--mtp-speculative-tokens N] [--no-cuda-graph] "
                         "[--served-model-name NAME] [--chat-template-file PATH]\n";
            std::exit(0);
        } else if (!key.empty() && key.rfind("--", 0) != 0 &&
                   args.model_dir.empty() && args.repo.empty()) {
            celeg::app::assign_model_or_repo_token(key, args.model_dir, args.repo);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.model_dir.empty() == args.repo.empty()) {
        throw std::runtime_error("exactly one of --model or --repo is required");
    }
    if (args.format != "auto" && args.format != "safetensors" && args.format != "gguf") {
        throw std::runtime_error("--format must be auto, safetensors, or gguf");
    }
    return args;
}

bool looks_like_gguf_repo(const std::string& repo) {
    return repo.size() >= 5 &&
        repo.compare(repo.size() - 5, 5, "-GGUF") == 0;
}

std::filesystem::path executable_directory(const char* argv0) {
#ifdef _WIN32
    std::array<wchar_t, 32768> path{};
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length > 0 && length < path.size()) {
        return std::filesystem::path(std::wstring_view(path.data(), length)).parent_path();
    }
#else
    std::error_code error;
    const auto executable = std::filesystem::canonical("/proc/self/exe", error);
    if (!error) return executable.parent_path();
#endif
    return std::filesystem::absolute(argv0).parent_path();
}

}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const bool use_gguf = !args.repo.empty() &&
            (args.format == "gguf" || (args.format == "auto" && looks_like_gguf_repo(args.repo)));
        const std::filesystem::path model = args.repo.empty()
            ? std::filesystem::path(args.model_dir)
            : (use_gguf
                ? celeg::resolve_hf_gguf(args.repo, args.quant)
                : celeg::resolve_hf_model(args.repo, "main", false));
        const auto runtime = celeg::create_builtin_runtime_context();
        const celeg::detail::ModelBootstrap bootstrap =
            celeg::detail::load_model_bootstrap(model, *runtime);
        const auto& topology = bootstrap.model.topology;
        if (args.context > topology.dims.max_position_embeddings) {
            throw std::runtime_error("--context exceeds model maximum");
        }

        const auto& tokenizer_provider = celeg::select_tokenizer_provider(
            *runtime, bootstrap.checkpoint, model);
        const auto tokenizer_storage = tokenizer_provider.create(
            bootstrap.checkpoint, model);
        const celeg::ITokenizer& tokenizer = *tokenizer_storage;
        const celeg::ResolvedInteraction chat_template = celeg::resolve_interaction(
            bootstrap.checkpoint.metadata, tokenizer,
            args.chat_template_file.empty()
                ? std::nullopt
                : std::optional{std::filesystem::path(args.chat_template_file)},
            model.parent_path() / "chat_template.jinja");
        std::cerr << "chat.template=" << chat_template.source_origin()
                  << " fingerprint=" << chat_template.fingerprint() << '\n';
        for (const std::string& diagnostic : chat_template.diagnostics()) {
            std::cerr << "chat.template_diagnostic=" << diagnostic << '\n';
        }

        const std::string model_name =
            args.served_model_name.empty() ? bootstrap.model.provenance.identity : args.served_model_name;
        const std::vector<std::int32_t> eos_token_ids(
            bootstrap.model.topology.dims.token_policy.eos_token_ids.begin(),
            bootstrap.model.topology.dims.token_policy.eos_token_ids.end());

        celeg::VisualEmbeddingProvider visual_embeddings;
        for (const auto& provider_id : runtime->vision_providers().ids()) {
            const auto& provider = runtime->vision_providers().find(provider_id);
            if (provider.supports(model)) {
                visual_embeddings = provider.create(model);
                break;
            }
        }

        std::unique_ptr<celeg::serve::ServiceBundle> service;
        if (args.backend == "cpu") {
            celeg::CpuModelOptions model_options;
            model_options.threads = static_cast<std::size_t>(args.threads);
            celeg::CpuConcurrentEngineOptions engine_options;
            engine_options.worker_thread = false;
            service = std::make_unique<celeg::serve::ServiceBundle>(
                std::make_unique<celeg::serve::CpuInferenceService>(
                    model.string(), args.context,
                    model_options, engine_options, visual_embeddings, runtime));
        } else if (args.backend == "cuda") {
#ifdef CELEG_SERVE_CUDA
            celeg::CudaModelOptions model_options;
            model_options.enable_mtp = args.enable_mtp;
            model_options.mtp_speculative_tokens = args.mtp_speculative_tokens;
            model_options.cuda_graph = args.cuda_graph;
            if (args.enable_mtp) model_options.cuda_graph = false;
            if (args.expert_offload == "auto") {
                model_options.expert_offload.mode = celeg::ExpertOffloadMode::Auto;
            } else if (args.expert_offload == "host") {
                model_options.expert_offload.mode = celeg::ExpertOffloadMode::Host;
            } else if (args.expert_offload != "none") {
                throw std::runtime_error("--expert-offload must be none, auto, or host");
            }
            if (args.expert_cache_policy == "static") {
                model_options.expert_offload.policy = celeg::ExpertCachePolicy::Static;
            } else if (args.expert_cache_policy == "lru") {
                model_options.expert_offload.policy = celeg::ExpertCachePolicy::Lru;
            } else if (args.expert_cache_policy == "lfu-lru") {
                model_options.expert_offload.policy = celeg::ExpertCachePolicy::LayerLocalLfuLru;
            } else {
                throw std::runtime_error("--expert-cache-policy must be static, lru, or lfu-lru");
            }
            if (args.expert_backing == "host") {
                model_options.expert_offload.backing = celeg::ExpertBackingMode::HostResident;
            } else if (args.expert_backing == "disk") {
                model_options.expert_offload.backing = celeg::ExpertBackingMode::DiskCached;
            } else {
                throw std::runtime_error("--expert-backing must be host or disk");
            }
            if (args.expert_cache_per_layer < 0 || args.expert_host_cache_mib <= 0) {
                throw std::runtime_error("expert cache values must be positive");
            }
            model_options.expert_offload.experts_per_layer = args.expert_cache_per_layer;
            model_options.expert_offload.host_expert_cache_bytes =
                static_cast<std::size_t>(args.expert_host_cache_mib) * 1024 * 1024;
            celeg::ConcurrentEngineOptions engine_options;
            engine_options.worker_thread = false;
            engine_options.max_active_requests = args.max_active_requests;
            engine_options.max_batched_tokens = args.max_batched_tokens;
            engine_options.prefill_chunk_tokens = args.prefill_chunk_tokens;
            if (topology.exec.gated_delta_net_layer_count > 0 ||
                topology.exec.mlp_only_layer_count > 0) {
                engine_options.packed_decode = false;
                engine_options.ragged_packed_prefill = false;
            }
            service = std::make_unique<celeg::serve::ServiceBundle>(
                std::make_unique<celeg::serve::CudaInferenceService>(
                    model.string(), args.context, model_options, engine_options,
                    visual_embeddings, runtime));
#else
            throw std::runtime_error("CUDA serving is not available in this build");
#endif
        } else {
            throw std::runtime_error("--backend must be cpu or cuda");
        }

        celeg::ChatCapabilities chat_capabilities = chat_template.capabilities();
        chat_capabilities.vision = static_cast<bool>(visual_embeddings);

        GenerationDispatcher dispatcher(service->requests(), service->scheduler());
        dispatcher.start();

        uWS::Loop* loop = uWS::Loop::get();
        const std::filesystem::path asset_root =
            executable_directory(argv[0]) / "celeg-serve-assets";

        uWS::App app;
        celeg::app::serve::register_health_routes(app);
        celeg::app::serve::register_chat_ui_routes(app, asset_root / "chat");
        celeg::app::serve::register_docs_routes(app, model_name, asset_root / "docs");
        celeg::app::serve::register_models_route(
            app, model_name, static_cast<std::size_t>(args.context), chat_capabilities);
        celeg::app::serve::register_tokenize_route(
            app, tokenizer, chat_template, static_cast<std::size_t>(args.context));
        celeg::app::serve::register_chat_completions_route(
            app, dispatcher, service->requests(), tokenizer, chat_template,
            model_name, eos_token_ids, static_cast<std::size_t>(args.context), loop);

        app.listen(args.host, args.port, [&](auto* listen_socket) {
              if (listen_socket) {
                  std::cout << "celeg-serve listening on http://" << args.host << ':' << args.port
                            << " (model=" << model_name << ")" << std::endl;
              } else {
                  std::cerr << "failed to listen on port " << args.port << std::endl;
              }
          })
            .run();

        dispatcher.stop();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "celeg-serve: " << error.what() << std::endl;
        return 1;
    }
}
