#include "celeg/models/gemma4/vision.hpp"

#include "celeg/runtime/providers.hpp"

#include <filesystem>
#include <memory>

namespace celeg {
namespace {

class Gemma4VisionProviderFactory final : public IVisionProviderFactory {
public:
    std::string_view id() const override { return "gemma4"; }

    bool supports(std::string_view architecture_id,
                  const std::filesystem::path& model_path) const override {
        return architecture_id == "gemma4" &&
               std::filesystem::is_regular_file(model_path / "mmproj-BF16.gguf");
    }

    std::shared_ptr<const IVisualEmbeddingProvider> create(
        const std::filesystem::path& model_path) const override {
        return make_gemma4_visual_embedding_provider(model_path / "mmproj-BF16.gguf");
    }
};

} // namespace

std::unique_ptr<IVisionProviderFactory> make_gemma4_vision_provider_factory() {
    return std::make_unique<Gemma4VisionProviderFactory>();
}

} // namespace celeg
