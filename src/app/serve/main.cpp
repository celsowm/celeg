#include "App.h"

#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/checkpoint/downloader.hpp"
#include "celeg/checkpoint/repositories/gguf.hpp"
#include "celeg/serve/cpu_inference_service.hpp"
#ifdef CELEG_SERVE_CUDA
#include "celeg/backend/cuda/cuda_inference_service.hpp"
#endif
#include "celeg/serve/generation_dispatcher.hpp"
#include "celeg/text/tokenizer.hpp"
#include "celeg/models/gemma4/vision.hpp"
#include "routes/chat_completions.hpp"
#include "routes/docs.hpp"
#include "routes/health.hpp"
#include "routes/models.hpp"
#include "routes/tokenize.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using celeg::serve::GenerationDispatcher;

struct Args {
    std::string model_dir;
    std::string served_model_name;
    int port = 8080;
    int context = 4096;
    int threads = 0;
    std::string backend = "cpu";
    std::string repo;
    std::string quant = "Q4_K_M";
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
        else if (key == "--served-model-name") args.served_model_name = value();
        else if (key == "--port") args.port = std::stoi(value());
        else if (key == "--context") args.context = std::stoi(value());
        else if (key == "--threads") args.threads = std::stoi(value());
        else if (key == "--backend") args.backend = value();
        else if (key == "--repo") args.repo = value();
        else if (key == "--quant") args.quant = value();
        else if (key == "--help") {
            std::cout << "celeg-serve (--model PATH | --repo HF_REPO) [--port 8080] [--context 4096] "
                         "[--threads N] [--backend cpu|cuda] "
                         "[--served-model-name NAME]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.model_dir.empty() == args.repo.empty()) {
        throw std::runtime_error("exactly one of --model or --repo is required");
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::filesystem::path model = args.repo.empty()
            ? std::filesystem::path(args.model_dir)
            : celeg::resolve_hf_gguf(args.repo, args.quant);
        const bool direct_gguf = std::filesystem::is_regular_file(model) &&
            model.extension() == ".gguf";

        const celeg::detail::ModelBootstrap bootstrap = celeg::detail::load_model_bootstrap(model);
        const auto& model_definition = bootstrap.model.definition;
        const auto& topology = bootstrap.model.topology;
        if (args.context > topology.max_position_embeddings) {
            throw std::runtime_error("--context exceeds model maximum");
        }

        const auto chat_catalog = celeg::make_chat_profile_catalog();
        const auto& chat_template = chat_catalog.find(bootstrap.model.chat_profile_id);
        const auto* gguf_repository = dynamic_cast<const celeg::GgufRepository*>(
            bootstrap.checkpoint.repository.get());
        const celeg::BpeTokenizer tokenizer = gguf_repository
            ? celeg::BpeTokenizer(celeg::BpeTokenizer::FromGguf{}, gguf_repository->file())
            : celeg::BpeTokenizer((std::filesystem::is_directory(model)
                ? model / "tokenizer.json" : model.parent_path() / "tokenizer.json").string());

        const std::string model_name =
            args.served_model_name.empty() ? bootstrap.model.identity : args.served_model_name;
        const std::int32_t eos_token_id = model_definition.tokens.eos;

        celeg::VisualEmbeddingProvider visual_embeddings;
        const std::filesystem::path projector = model.parent_path() / "mmproj-BF16.gguf";
        if (bootstrap.model.chat_profile_id == "gemma4-instruct" &&
            std::filesystem::is_regular_file(projector)) {
            visual_embeddings = celeg::make_gemma4_visual_embedding_provider(projector);
        }

        std::unique_ptr<celeg::serve::ServiceBundle> service;
        if (args.backend == "cpu") {
            celeg::CpuModelOptions model_options;
            model_options.threads = static_cast<std::size_t>(args.threads);
            celeg::CpuConcurrentEngineOptions engine_options;
            engine_options.worker_thread = false;
            service = std::make_unique<celeg::serve::ServiceBundle>(
                std::make_unique<celeg::serve::CpuInferenceService>(
                    (direct_gguf ? model : model / "model.safetensors").string(), args.context,
                    model_options, engine_options, visual_embeddings));
        } else if (args.backend == "cuda") {
#ifdef CELEG_SERVE_CUDA
            celeg::ConcurrentEngineOptions engine_options;
            engine_options.worker_thread = false;
            service = std::make_unique<celeg::serve::ServiceBundle>(
                std::make_unique<celeg::serve::CudaInferenceService>(
                    model.string(), args.context, celeg::CudaModelOptions{}, engine_options,
                    visual_embeddings));
#else
            throw std::runtime_error("CUDA serving is not available in this build");
#endif
        } else {
            throw std::runtime_error("--backend must be cpu or cuda");
        }

        celeg::ChatCapabilities chat_capabilities =
            chat_catalog.capabilities(bootstrap.model.chat_profile_id);
        chat_capabilities.vision = static_cast<bool>(visual_embeddings);

        GenerationDispatcher dispatcher(service->requests(), service->scheduler());
        dispatcher.start();

        uWS::Loop* loop = uWS::Loop::get();

        uWS::App app;
        celeg::app::serve::register_health_routes(app);
        celeg::app::serve::register_docs_routes(app, model_name);
        celeg::app::serve::register_models_route(app, model_name);
        celeg::app::serve::register_tokenize_route(
            app, tokenizer, chat_template, static_cast<std::size_t>(args.context));
        celeg::app::serve::register_chat_completions_route(
            app, dispatcher, service->requests(), tokenizer, chat_template,
            chat_capabilities,
            model_name, eos_token_id, loop);

        app.listen(args.port, [&](auto* listen_socket) {
              if (listen_socket) {
                  std::cout << "celeg-serve listening on http://127.0.0.1:" << args.port
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
