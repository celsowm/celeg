#include "celeg/runtime/providers.hpp"

#include "celeg/text/tokenizer.hpp"
#include "celeg/text/tokenizer_definition.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>

namespace celeg {
namespace {

std::filesystem::path tokenizer_json_path(const std::filesystem::path& model_path) {
    const auto root = std::filesystem::is_directory(model_path)
        ? model_path : model_path.parent_path();
    return root / "tokenizer.json";
}

class BpeTokenizerProvider final : public ITokenizerProvider {
public:
    explicit BpeTokenizerProvider(std::vector<TokenizerPreTokenizerRule> rules)
        : rules_(std::move(rules)) {}

    std::string_view id() const override { return "bpe"; }

    bool supports(const CheckpointView& checkpoint,
                  const std::filesystem::path& model_path) const override {
        return checkpoint.tokenizer != nullptr ||
               std::filesystem::is_regular_file(tokenizer_json_path(model_path));
    }

    std::unique_ptr<ITokenizer> create(
        const CheckpointView& checkpoint,
        const std::filesystem::path& model_path) const override {
        if (checkpoint.tokenizer) {
            return std::make_unique<BpeTokenizer>(
                resolve_tokenizer_definition(*checkpoint.tokenizer, rules_));
        }
        const auto path = tokenizer_json_path(model_path);
        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error("tokenizer provider found no tokenizer data: " +
                                     path.string());
        }
        return std::make_unique<BpeTokenizer>(load_tokenizer_definition_json(path.string()));
    }
private:
    std::vector<TokenizerPreTokenizerRule> rules_;
};

}

std::unique_ptr<ITokenizerProvider> make_builtin_tokenizer_provider(
    std::vector<TokenizerPreTokenizerRule> rules) {
    return std::make_unique<BpeTokenizerProvider>(std::move(rules));
}

}
