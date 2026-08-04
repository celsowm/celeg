#include "celeg/runtime/providers.hpp"

#include "celeg/text/tokenizer.hpp"

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
    std::string_view id() const override { return "bpe"; }

    bool supports(const CheckpointView& checkpoint,
                  const std::filesystem::path& model_path) const override {
        return checkpoint.tokenizer != nullptr ||
               std::filesystem::is_regular_file(tokenizer_json_path(model_path));
    }

    std::unique_ptr<BpeTokenizer> create(
        const CheckpointView& checkpoint,
        const std::filesystem::path& model_path) const override {
        if (checkpoint.tokenizer) {
            return std::make_unique<BpeTokenizer>(*checkpoint.tokenizer);
        }
        const auto path = tokenizer_json_path(model_path);
        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error("tokenizer provider found no tokenizer data: " +
                                     path.string());
        }
        return std::make_unique<BpeTokenizer>(path.string());
    }
};

} // namespace

std::unique_ptr<ITokenizerProvider> make_builtin_tokenizer_provider() {
    return std::make_unique<BpeTokenizerProvider>();
}

} // namespace celeg
