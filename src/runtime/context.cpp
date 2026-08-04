#include "celeg/runtime/context.hpp"
#include "celeg/model/visual_embeddings.hpp"
#include "celeg/models/gemma4/vision.hpp"
#include "celeg/text/tokenizer.hpp"

#include <stdexcept>

namespace celeg {

namespace {

class BuiltinBackendFactory final : public IBackendFactory {
public:
    BuiltinBackendFactory(std::string_view id, BackendKind backend)
        : id_(id), backend_(backend) {}

    std::string_view id() const override { return id_; }
    bool supports(BackendKind backend) const override { return backend == backend_; }

private:
    std::string_view id_;
    BackendKind backend_;
};

class Gemma4VisionProviderFactory final : public IVisionProviderFactory {
public:
    std::string_view id() const override { return "gemma4"; }

    bool supports(std::string_view architecture_id,
                  const std::filesystem::path& projector_path) const override {
        return architecture_id == "gemma4" &&
               std::filesystem::is_regular_file(projector_path);
    }

    std::shared_ptr<const IVisualEmbeddingProvider> create(
        const std::filesystem::path& projector_path) const override {
        return make_gemma4_visual_embedding_provider(projector_path);
    }
};

} // namespace

RuntimeBuilder::RuntimeBuilder()
    : architectures_(std::make_shared<ArchitectureCatalog>()),
      checkpoint_formats_(std::make_shared<CheckpointFormatCatalog>()),
      chat_profiles_(std::make_shared<ChatProfileCatalog>()),
      tokenizer_providers_(std::make_shared<TokenizerProviderCatalog>()),
      backends_(std::make_shared<BackendFactoryCatalog>()),
      vision_providers_(std::make_shared<VisionProviderCatalog>()) {}

RuntimeBuilder& RuntimeBuilder::add_builtins() {
    add_builtin_architectures(*architectures_);
    add_builtin_checkpoint_formats(*checkpoint_formats_);
    add_builtin_chat_profiles(*chat_profiles_);
    tokenizer_providers_->add(make_builtin_tokenizer_provider());
    backends_->add(std::make_unique<BuiltinBackendFactory>("cpu", BackendKind::Cpu));
    backends_->add(std::make_unique<BuiltinBackendFactory>("cuda", BackendKind::Cuda));
    vision_providers_->add(std::make_unique<Gemma4VisionProviderFactory>());
    return *this;
}

const ITokenizerProvider& select_tokenizer_provider(
    const RuntimeContext& runtime,
    const CheckpointView& checkpoint,
    const std::filesystem::path& model_path) {
    return runtime.tokenizer_providers().select_if(
        [&](const ITokenizerProvider& provider) {
            return provider.supports(checkpoint, model_path);
        });
}

RuntimeBuilder& RuntimeBuilder::add_architecture(
    std::unique_ptr<IArchitecture> architecture) {
    architectures_->add(std::move(architecture));
    return *this;
}

RuntimeBuilder& RuntimeBuilder::add_checkpoint_format(
    std::unique_ptr<ICheckpointFormat> format) {
    checkpoint_formats_->add(std::move(format));
    return *this;
}

RuntimeBuilder& RuntimeBuilder::add_chat_profile(
    std::string profile_id,
    std::unique_ptr<IChatTemplate> chat_template,
    std::unique_ptr<IChatToolCallCodec> tool_call_codec,
    ChatCapabilities capabilities) {
    chat_profiles_->add(std::move(profile_id), std::move(chat_template),
                        std::move(tool_call_codec), capabilities);
    return *this;
}

RuntimeBuilder& RuntimeBuilder::add_tokenizer_provider(
    std::unique_ptr<ITokenizerProvider> provider) {
    tokenizer_providers_->add(std::move(provider));
    return *this;
}

RuntimeBuilder& RuntimeBuilder::add_backend_factory(
    std::unique_ptr<IBackendFactory> factory) {
    backends_->add(std::move(factory));
    return *this;
}

RuntimeBuilder& RuntimeBuilder::add_vision_provider(
    std::unique_ptr<IVisionProviderFactory> provider) {
    vision_providers_->add(std::move(provider));
    return *this;
}

RuntimeContext RuntimeBuilder::build() {
    architectures_->freeze();
    checkpoint_formats_->freeze();
    chat_profiles_->freeze();
    tokenizer_providers_->freeze();
    backends_->freeze();
    vision_providers_->freeze();
    return RuntimeContext{std::move(architectures_), std::move(checkpoint_formats_),
                          std::move(chat_profiles_), std::move(tokenizer_providers_),
                          std::move(backends_), std::move(vision_providers_)};
}

std::shared_ptr<const RuntimeContext> create_builtin_runtime_context() {
    static const std::shared_ptr<const RuntimeContext> context = [] {
        return std::make_shared<const RuntimeContext>(
            RuntimeBuilder{}.add_builtins().build());
    }();
    return context;
}

} // namespace celeg
