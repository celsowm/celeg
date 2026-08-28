#include "celeg/app/run_preparation.hpp"
#include "celeg/serve/metal_inference_service.hpp"
#include "celeg/text/utf8.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model;
    std::string repo;
    std::string prompt;
    int context = 4096;
    int max_new_tokens = 4;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + key);
            return argv[i];
        };
        if (key == "--model") args.model = value();
        else if (key == "--repo") args.repo = value();
        else if (key == "--prompt") args.prompt = value();
        else if (key == "--context") args.context = std::stoi(value());
        else if (key == "--max-new-tokens") args.max_new_tokens = std::stoi(value());
        else if (key == "--help") {
            std::cout << "celeg-metal-run [--model DIR | --repo REPO_ID] --prompt TEXT "
                         "[--context N] [--max-new-tokens N]\n";
            std::exit(0);
        } else if (!key.empty() && key.rfind("--", 0) != 0 &&
                   args.model.empty() && args.repo.empty()) {
            celeg::app::assign_model_or_repo_token(key, args.model, args.repo);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.model.empty() == args.repo.empty()) {
        throw std::runtime_error("exactly one of --model or --repo is required");
    }
    if (args.prompt.empty() || args.context <= 0 || args.max_new_tokens < 0) {
        throw std::runtime_error("prompt and positive context are required");
    }
    return args;
}

}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const celeg::app::RunInputs inputs{
            args.model, args.repo, args.prompt, {}, {}, args.context,
            args.max_new_tokens, 1, 0.1f, 1.0f, 1.05f, 1, false};
        auto prepared = celeg::app::prepare_run(inputs, true);
        const auto prompt = celeg::app::prepare_prompt(inputs, prepared);
        celeg::MetalEngineOptions engine_options;
        celeg::serve::MetalInferenceService service(
            prepared.checkpoint_path.string(), args.context, {}, engine_options,
            prepared.runtime);
        celeg::serve::GenerateRequest request;
        request.prompt_tokens = prompt;
        request.max_output_tokens = static_cast<size_t>(args.max_new_tokens);
        request.generation = celeg::app::generation_config(inputs);
        request.eos_token_ids.assign(
            prepared.bootstrap.model.topology.dims.token_policy.eos_token_ids.begin(),
            prepared.bootstrap.model.topology.dims.token_policy.eos_token_ids.end());
        const auto id = service.submit(std::move(request));
        service.start();
        std::string pending;
        for (;;) {
            service.step();
            const auto event = service.poll(id, 1);
            for (const int32_t token : event.tokens) {
                pending += prepared.tokenizer->decode({token}, true);
                const size_t safe_size = celeg::text::complete_utf8_prefix(pending);
                std::cout << pending.substr(0, safe_size) << std::flush;
                pending.erase(0, safe_size);
            }
            if (event.finished) break;
        }
        service.stop();
        std::cout << pending << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
