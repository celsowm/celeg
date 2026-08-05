#pragma once

#include "celeg/checkpoint/catalog.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/text/chat_template.hpp"
#include "celeg/runtime/providers.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace celeg {

// Immutable ownership boundary for runtime extension catalogs. A context is
// cheap to copy and safe to share after RuntimeBuilder::build() returns.
class RuntimeContext {
public:
    const ArchitectureCatalog& architectures() const { return *architectures_; }
    const CheckpointFormatCatalog& checkpoint_formats() const {
        return *checkpoint_formats_;
    }
    const ChatProfileCatalog& chat_profiles() const { return *chat_profiles_; }
    const TokenizerProviderCatalog& tokenizer_providers() const {
        return *tokenizer_providers_;
    }
    const BackendFactoryCatalog& backends() const { return *backends_; }
    const VisionProviderCatalog& vision_providers() const { return *vision_providers_; }

private:
    friend class RuntimeBuilder;
    RuntimeContext(std::shared_ptr<const ArchitectureCatalog> architectures,
                   std::shared_ptr<const CheckpointFormatCatalog> checkpoint_formats,
                   std::shared_ptr<const ChatProfileCatalog> chat_profiles,
                   std::shared_ptr<const TokenizerProviderCatalog> tokenizer_providers,
                   std::shared_ptr<const BackendFactoryCatalog> backends,
                   std::shared_ptr<const VisionProviderCatalog> vision_providers)
        : architectures_(std::move(architectures)),
          checkpoint_formats_(std::move(checkpoint_formats)),
          chat_profiles_(std::move(chat_profiles)),
          tokenizer_providers_(std::move(tokenizer_providers)),
          backends_(std::move(backends)),
          vision_providers_(std::move(vision_providers)) {}

    std::shared_ptr<const ArchitectureCatalog> architectures_;
    std::shared_ptr<const CheckpointFormatCatalog> checkpoint_formats_;
    std::shared_ptr<const ChatProfileCatalog> chat_profiles_;
    std::shared_ptr<const TokenizerProviderCatalog> tokenizer_providers_;
    std::shared_ptr<const BackendFactoryCatalog> backends_;
    std::shared_ptr<const VisionProviderCatalog> vision_providers_;
};

const ITokenizerProvider& select_tokenizer_provider(
    const RuntimeContext& runtime,
    const CheckpointView& checkpoint,
    const std::filesystem::path& model_path);

class RuntimeBuilder {
public:
    RuntimeBuilder();

    RuntimeBuilder& add_builtins();
    RuntimeBuilder& add_architecture(std::unique_ptr<IArchitecture> architecture);
    RuntimeBuilder& add_checkpoint_format(std::unique_ptr<ICheckpointFormat> format);
    RuntimeBuilder& add_chat_profile(
        std::string profile_id,
        std::unique_ptr<IChatTemplate> chat_template,
        std::unique_ptr<IChatToolCallCodec> tool_call_codec = nullptr,
        ChatCapabilities capabilities = {});
    RuntimeBuilder& add_tokenizer_provider(std::unique_ptr<ITokenizerProvider> provider);
    RuntimeBuilder& add_backend_factory(std::unique_ptr<IBackendFactory> factory);
    RuntimeBuilder& add_vision_provider(std::unique_ptr<IVisionProviderFactory> provider);
    RuntimeBuilder& add_module(std::unique_ptr<IRuntimeModule> module);

    // Registration-only catalog views used by RuntimeModule implementations.
    // They are invalid after build() and are deliberately not exposed by
    // RuntimeContext.
    ArchitectureCatalog& architecture_catalog_for_registration() { return *architectures_; }
    ChatProfileCatalog& chat_profile_catalog_for_registration() { return *chat_profiles_; }
    VisionProviderCatalog& vision_catalog_for_registration() { return *vision_providers_; }
    RuntimeContext build();

private:
    std::shared_ptr<ArchitectureCatalog> architectures_;
    std::shared_ptr<CheckpointFormatCatalog> checkpoint_formats_;
    std::shared_ptr<ChatProfileCatalog> chat_profiles_;
    std::shared_ptr<TokenizerProviderCatalog> tokenizer_providers_;
    std::shared_ptr<BackendFactoryCatalog> backends_;
    std::shared_ptr<VisionProviderCatalog> vision_providers_;
    std::vector<std::unique_ptr<IRuntimeModule>> modules_;
};

std::shared_ptr<const RuntimeContext> create_builtin_runtime_context();

} // namespace celeg
