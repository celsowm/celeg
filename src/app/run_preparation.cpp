#include "celeg/app/run_preparation.hpp"

#include "celeg/checkpoint/downloader.hpp"

#include <stdexcept>

namespace celeg::app {
namespace {

std::pair<std::string, std::string> split_repo_quant_tag(const std::string& repo) {
    const std::size_t colon = repo.find(':');
    return colon == std::string::npos
        ? std::pair{repo, std::string{}}
        : std::pair{repo.substr(0, colon), repo.substr(colon + 1)};
}

bool is_gguf_repo(std::string_view repo, std::string_view quant_tag) {
    return !repo.empty() && (repo.ends_with("-GGUF") || !quant_tag.empty());
}

} // namespace

PreparedRun prepare_run(const RunInputs& inputs, bool resolve_chat) {
    if (inputs.model.empty() == inputs.repo.empty()) {
        throw std::invalid_argument("exactly one of --model or --repo is required");
    }
    if (inputs.context <= 0 || inputs.max_new_tokens < 0 || inputs.top_k <= 0 ||
        inputs.temperature < 0.0f || inputs.top_p <= 0.0f || inputs.top_p > 1.0f ||
        inputs.repetition_penalty < 1.0f || inputs.seed == 0) {
        throw std::invalid_argument("invalid shared runner option");
    }

    PreparedRun prepared;
    if (!inputs.repo.empty()) {
        const auto [repo, quant_tag] = split_repo_quant_tag(inputs.repo);
        prepared.is_gguf = is_gguf_repo(repo, quant_tag);
        prepared.checkpoint_path = prepared.is_gguf
            ? resolve_hf_gguf(repo, quant_tag.empty() ? "Q4_K_M" : quant_tag)
            : resolve_hf_model(repo);
    } else {
        prepared.checkpoint_path = inputs.model;
        prepared.is_gguf = prepared.checkpoint_path.extension() == ".gguf";
    }

    prepared.runtime = create_builtin_runtime_context();
    prepared.bootstrap = detail::load_model_bootstrap(
        prepared.checkpoint_path, *prepared.runtime);
    if (inputs.context > prepared.bootstrap.model.topology.dims.max_position_embeddings) {
        throw std::invalid_argument("--context exceeds model maximum");
    }
    const auto& tokenizer_provider = select_tokenizer_provider(
        *prepared.runtime, prepared.bootstrap.checkpoint, prepared.checkpoint_path);
    prepared.tokenizer = tokenizer_provider.create(
        prepared.bootstrap.checkpoint, prepared.checkpoint_path);
    if (!prepared.tokenizer) throw std::runtime_error("tokenizer provider returned null");

    if (resolve_chat) {
        prepared.chat_template.emplace(resolve_chat_template(
            prepared.bootstrap.checkpoint.metadata, *prepared.tokenizer,
            inputs.chat_template_file.empty()
                ? std::nullopt
                : std::optional{std::filesystem::path(inputs.chat_template_file)}));
    }
    return prepared;
}

std::vector<std::int32_t> prepare_prompt(const RunInputs& inputs,
                                         const PreparedRun& prepared) {
    if (inputs.raw_prompt) return prepared.tokenizer->encode(inputs.prompt, true);
    if (!prepared.chat_template) {
        throw std::logic_error("chat prompt requested without a resolved template");
    }
    std::vector<ChatMessage> messages;
    if (!inputs.system.empty()) messages.push_back({ChatRole::System, inputs.system});
    messages.push_back({ChatRole::User, inputs.prompt});
    return prepared.tokenizer->encode(render_chat(messages, *prepared.chat_template), false);
}

GenerationConfig generation_config(const RunInputs& inputs) {
    GenerationConfig generation;
    generation.temperature = inputs.temperature;
    generation.top_k = inputs.top_k;
    generation.top_p = inputs.top_p;
    generation.repetition_penalty = inputs.repetition_penalty;
    generation.seed = inputs.seed;
    return generation;
}

} // namespace celeg::app
