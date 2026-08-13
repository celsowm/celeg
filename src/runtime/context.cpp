#include "celeg/runtime/context.hpp"
#include "celeg/model/runtime_modules.hpp"

#include <stdexcept>

namespace celeg {

RuntimeBuilder::RuntimeBuilder()
    : architectures_(std::make_shared<ArchitectureCatalog>()),
      checkpoint_formats_(std::make_shared<CheckpointFormatCatalog>()),
      tokenizer_providers_(std::make_shared<TokenizerProviderCatalog>()),
      backends_(std::make_shared<BackendFactoryCatalog>()),
      vision_providers_(std::make_shared<VisionProviderCatalog>()) {}

RuntimeBuilder& RuntimeBuilder::add_builtins() {
    add_builtin_checkpoint_formats(*checkpoint_formats_);
    for (auto& module : make_builtin_runtime_modules()) add_module(std::move(module));
    return *this;
}

RuntimeBuilder& RuntimeBuilder::add_module(std::unique_ptr<IRuntimeModule> module) {
    if (!module || module->id().empty()) {
        throw std::invalid_argument("runtime module must have a non-empty id");
    }
    for (const auto& existing : modules_) {
        if (existing->id() == module->id()) {
            throw std::invalid_argument("duplicate runtime module id: " +
                                        std::string(module->id()));
        }
    }
    module->register_into(*this);
    modules_.push_back(std::move(module));
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
    tokenizer_providers_->freeze();
    backends_->freeze();
    vision_providers_->freeze();
    return RuntimeContext{std::move(architectures_), std::move(checkpoint_formats_),
                          std::move(tokenizer_providers_),
                          std::move(backends_), std::move(vision_providers_)};
}

std::shared_ptr<const RuntimeContext> RuntimeBuilder::build_shared() {
    return std::make_shared<const RuntimeContext>(build());
}

std::shared_ptr<const RuntimeContext> create_builtin_runtime_context() {
    static const std::shared_ptr<const RuntimeContext> context = [] {
        return std::make_shared<const RuntimeContext>(
            RuntimeBuilder{}.add_builtins().build());
    }();
    return context;
}

} // namespace celeg
