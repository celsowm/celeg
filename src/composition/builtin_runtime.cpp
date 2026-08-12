#include "celeg/model/runtime_modules.hpp"

#include "celeg/model/descriptor.hpp"
#include "celeg/runtime/context.hpp"
#include "celeg/runtime/providers.hpp"
#include "celeg/runtime/vision/providers.hpp"
#include "celeg/text/semantic_chat_templates.hpp"

#include <filesystem>

namespace celeg {
namespace {

class CheckpointResolutionModule final : public IRuntimeModule {
public:
    std::string_view id() const override { return "checkpoint-resolution"; }

    void register_into(RuntimeBuilder& builder) const override {
        builder.add_architecture(make_automatic_architecture());
        std::filesystem::path descriptor_directory(CELEG_DESCRIPTOR_DIRECTORY);
        if (!std::filesystem::exists(descriptor_directory)) {
#ifdef CELEG_DESCRIPTOR_INSTALL_DIRECTORY
            descriptor_directory = std::filesystem::path(CELEG_DESCRIPTOR_INSTALL_DIRECTORY);
#endif
        }
        register_descriptor_architectures(
            builder.architecture_catalog_for_registration(), descriptor_directory);
    }
};

class TokenizerRuntimeModule final : public IRuntimeModule {
public:
    std::string_view id() const override { return "tokenizer-runtime"; }

    void register_into(RuntimeBuilder& builder) const override {
        builder.add_tokenizer_provider(make_builtin_tokenizer_provider());
    }
};

class VisionRuntimeModule final : public IRuntimeModule {
public:
    std::string_view id() const override { return "vision-runtime"; }

    void register_into(RuntimeBuilder& builder) const override {
        builder.add_vision_provider(make_gguf_vision_provider_factory());
        builder.add_vision_provider(make_safetensor_vision_provider_factory());
        builder.add_vision_provider(make_patch_vision_provider_factory());
    }
};

class ChatTemplateRuntimeModule final : public IRuntimeModule {
public:
    std::string_view id() const override { return "chat-template-runtime"; }

    void register_into(RuntimeBuilder& builder) const override {
        ChatTemplateCatalog& catalog =
            builder.chat_template_catalog_for_registration();
        add_delimited_chat_template(catalog);
        add_role_envelope_chat_template(catalog);
        add_turn_chat_template(catalog);
        add_thinking_function_chat_template(catalog);
        add_reasoning_xml_chat_template(catalog);
        add_metadata_thinking_chat_template(catalog);
        add_vision_role_chat_template(catalog);
        add_patch_role_chat_template(catalog);
        add_thinking_role_chat_template(catalog);
        add_tagged_role_chat_template(catalog);
    }
};

} // namespace

std::vector<std::unique_ptr<IRuntimeModule>> make_builtin_runtime_modules() {
    std::vector<std::unique_ptr<IRuntimeModule>> modules;
    modules.push_back(std::make_unique<CheckpointResolutionModule>());
    modules.push_back(std::make_unique<TokenizerRuntimeModule>());
    modules.push_back(std::make_unique<VisionRuntimeModule>());
    modules.push_back(std::make_unique<ChatTemplateRuntimeModule>());
    return modules;
}

} // namespace celeg
