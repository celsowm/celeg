#pragma once

#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/runtime/context.hpp"
#include "celeg/runtime/request_types.hpp"
#include "celeg/text/chat_template.hpp"
#include "celeg/text/tokenizer.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace celeg::app {

struct RunInputs {
    std::string model;
    std::string repo;
    std::string prompt;
    std::string system;
    std::string chat_template_file;
    int context = 4096;
    int max_new_tokens = 128;
    int top_k = 1;
    float temperature = 0.1f;
    float top_p = 1.0f;
    float repetition_penalty = 1.05f;
    std::uint64_t seed = 1;
    bool raw_prompt = false;
};

struct PreparedRun {
    std::filesystem::path checkpoint_path;
    bool is_gguf = false;
    std::shared_ptr<const RuntimeContext> runtime;
    detail::ModelBootstrap bootstrap;
    std::unique_ptr<ITokenizer> tokenizer;
    std::optional<ResolvedInteraction> chat_template;
};

PreparedRun prepare_run(const RunInputs& inputs, bool resolve_chat_template = true);
std::vector<std::int32_t> prepare_prompt(const RunInputs& inputs,
                                         const PreparedRun& prepared);
GenerationConfig generation_config(const RunInputs& inputs);

}
