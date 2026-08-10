#include "celeg/model/runtime_modules.hpp"

#include "celeg/models/gemma4/chat_template.hpp"
#include "celeg/models/gemma4/vision.hpp"
#include "celeg/models/granite/chat_template.hpp"
#include "celeg/models/lfm2/chat_template.hpp"
#include "celeg/models/minicpm5/chat_template.hpp"
#include "celeg/model/descriptor.hpp"
#include "celeg/models/nanbeige/chat_template.hpp"
#include "celeg/models/nemotron_h/chat_template.hpp"
#include "celeg/models/qwen35/chat_template.hpp"
#include "celeg/models/qwen35/vision.hpp"
#include "celeg/models/smollm3/chat_template.hpp"
#include "celeg/runtime/context.hpp"
#include "celeg/text/tokenizer.hpp"
#include "celeg/text/tokenizer_definition.hpp"

#include <filesystem>

namespace celeg {
namespace {

using ChatRegistrar = void (*)(ChatProfileCatalog&);

struct BuiltinChatRegistration {
    std::string_view module_id;
    ChatRegistrar chat;
};

constexpr BuiltinChatRegistration kDeclarativeChats[] = {
    {"lfm2-chat", &add_lfm2_chat_profile},
    {"granite-chat", &add_granite_chat_profile},
    {"gemma4-chat", &add_gemma4_chat_profile},
    {"minicpm5-chat", &add_minicpm5_chat_profile},
    {"smollm3-chat", &add_smollm3_chat_profile},
    {"nanbeige42-chat", &add_nanbeige42_chat_profile},
    {"qwen35-chat", &add_qwen35_chat_profile},
    {"nemotron-h-chat", &add_nemotron_h_chat_profile},
};

class DeclarativeArchitectureModule final : public IRuntimeModule {
public:
    std::string_view id() const override { return "declarative-dense-architectures"; }

    void register_into(RuntimeBuilder& builder) const override {
        builder.add_architecture(make_automatic_architecture());
        std::filesystem::path descriptor_directory(CELEG_DESCRIPTOR_DIRECTORY);
        if (!std::filesystem::exists(descriptor_directory)) {
#ifdef CELEG_DESCRIPTOR_INSTALL_DIRECTORY
            descriptor_directory = std::filesystem::path(CELEG_DESCRIPTOR_INSTALL_DIRECTORY);
#endif
        }
        register_descriptor_architectures(
            builder.architecture_catalog_for_registration(),
            descriptor_directory);
    }
};

class ChatProfileModule final : public IRuntimeModule {
public:
    ChatProfileModule(std::string_view id, ChatRegistrar registrar)
        : id_(id), registrar_(registrar) {}

    std::string_view id() const override { return id_; }
    void register_into(RuntimeBuilder& builder) const override {
        registrar_(builder.chat_profile_catalog_for_registration());
    }

private:
    std::string_view id_;
    ChatRegistrar registrar_;
};

class BuiltinTokenizerModule final : public IRuntimeModule {
public:
    std::string_view id() const override { return "builtin-tokenizer"; }

    void register_into(RuntimeBuilder& builder) const override {
        builder.add_tokenizer_provider(make_builtin_tokenizer_provider({
            {"lfm2", TokenizerPreTokenizerKind::NumericTriplets},
            {"smaug-bpe", TokenizerPreTokenizerKind::NumericTriplets},
            {"gemma4", TokenizerPreTokenizerKind::RawUtf8},
            {"gpt2", TokenizerPreTokenizerKind::NumericRuns},
            {"granite", TokenizerPreTokenizerKind::NumericRuns, true},
        }));
    }
};

class Gemma4VisionModule final : public IRuntimeModule {
public:
    std::string_view id() const override { return "gemma4-vision"; }

    void register_into(RuntimeBuilder& builder) const override {
        builder.vision_catalog_for_registration().add(
            make_gemma4_vision_provider_factory());
    }
};

class Qwen35VisionModule final : public IRuntimeModule {
public:
    std::string_view id() const override { return "qwen35-vision"; }

    void register_into(RuntimeBuilder& builder) const override {
        builder.vision_catalog_for_registration().add(
            make_qwen35_vision_provider_factory());
    }
};

} // namespace

std::vector<std::unique_ptr<IRuntimeModule>> make_builtin_runtime_modules() {
    std::vector<std::unique_ptr<IRuntimeModule>> modules;
    modules.push_back(std::make_unique<DeclarativeArchitectureModule>());
    for (const BuiltinChatRegistration& chat : kDeclarativeChats) {
        modules.push_back(std::make_unique<ChatProfileModule>(chat.module_id, chat.chat));
    }
    modules.push_back(std::make_unique<BuiltinTokenizerModule>());
    modules.push_back(std::make_unique<Gemma4VisionModule>());
    modules.push_back(std::make_unique<Qwen35VisionModule>());
    return modules;
}

} // namespace celeg
