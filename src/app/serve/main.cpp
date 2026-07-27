#include "App.h"

#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/model/config/config.hpp"
#include "lfm/serve/cpu_inference_service.hpp"
#include "lfm/serve/generation_dispatcher.hpp"
#include "lfm/text/tokenizer.hpp"
#include "routes/chat_completions.hpp"
#include "routes/docs.hpp"
#include "routes/health.hpp"
#include "routes/models.hpp"
#include "routes/tokenize.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using lfm::serve::GenerationDispatcher;

struct Args {
    std::string model_dir;
    std::string served_model_name;
    int port = 8080;
    int context = 4096;
    int threads = 0;
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
        else if (key == "--help") {
            std::cout << "lfm25-serve --model DIR [--port 8080] [--context 4096] "
                         "[--threads N] [--served-model-name NAME]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.model_dir.empty()) throw std::runtime_error("--model is required");
    return args;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::filesystem::path model(args.model_dir);

        const lfm::detail::ModelBootstrap bootstrap = lfm::detail::load_model_bootstrap(model);
        const lfm::ModelConfig& config = bootstrap.config;
        const lfm::IModelVariant& variant = *bootstrap.variant;
        if (args.context > config.max_position_embeddings) {
            throw std::runtime_error("--context exceeds model maximum");
        }

        const lfm::BpeTokenizer tokenizer((model / "tokenizer.json").string(),
            lfm::make_chat_template(variant.chat_template_kind()));

        const std::string model_name =
            args.served_model_name.empty() ? std::string(variant.id()) : args.served_model_name;
        const std::int32_t eos_token_id = config.eos_token_id;

        lfm::CpuModelOptions model_options;
        model_options.threads = static_cast<std::size_t>(args.threads);
        lfm::CpuConcurrentEngineOptions engine_options;

        lfm::serve::CpuInferenceService service(
            (model / "model.safetensors").string(), args.context, model_options, engine_options);

        GenerationDispatcher dispatcher(service);
        dispatcher.start();

        uWS::Loop* loop = uWS::Loop::get();

        uWS::App app;
        lfm::app::serve::register_health_routes(app);
        lfm::app::serve::register_docs_routes(app, model_name);
        lfm::app::serve::register_models_route(app, model_name);
        lfm::app::serve::register_tokenize_route(app, tokenizer, static_cast<std::size_t>(args.context));
        lfm::app::serve::register_chat_completions_route(
            app, dispatcher, service, tokenizer, model_name, eos_token_id, loop);

        app.listen(args.port, [&](auto* listen_socket) {
              if (listen_socket) {
                  std::cout << "lfm25-serve listening on http://127.0.0.1:" << args.port
                            << " (model=" << model_name << ")" << std::endl;
              } else {
                  std::cerr << "failed to listen on port " << args.port << std::endl;
              }
          })
            .run();

        dispatcher.stop();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "lfm25-serve: " << error.what() << std::endl;
        return 1;
    }
}
