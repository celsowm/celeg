#include "celeg/model/runtime_modules.hpp"

#include "celeg/models/gemma4/architecture.hpp"
#include "celeg/models/gemma4/chat_template.hpp"
#include "celeg/models/gemma4/vision.hpp"
#include "celeg/models/granite/architecture.hpp"
#include "celeg/models/granite/chat_template.hpp"
#include "celeg/models/lfm2/architecture.hpp"
#include "celeg/models/lfm2/chat_template.hpp"
#include "celeg/models/minicpm5/architecture.hpp"
#include "celeg/models/minicpm5/chat_template.hpp"
#include "celeg/models/nanbeige/architecture.hpp"
#include "celeg/models/nanbeige/chat_template.hpp"
#include "celeg/models/nemotron_h/architecture.hpp"
#include "celeg/models/nemotron_h/chat_template.hpp"
#include "celeg/models/qwen35/architecture.hpp"
#include "celeg/models/qwen35/chat_template.hpp"
#include "celeg/models/qwen35/vision.hpp"
#include "celeg/models/smollm3/architecture.hpp"
#include "celeg/models/smollm3/chat_template.hpp"
#include "celeg/runtime/context.hpp"
#include "celeg/text/tokenizer.hpp"
#include "celeg/text/tokenizer_definition.hpp"

namespace celeg {
namespace {

using ArchitectureRegistrar = void (*)(ArchitectureCatalog&);
using ChatRegistrar = void (*)(ChatProfileCatalog&);

struct BuiltinFamilyRegistration {
    std::string_view module_id;
    ArchitectureRegistrar architecture;
    ChatRegistrar chat;
};

constexpr BuiltinFamilyRegistration kBuiltinFamilies[] = {
    {"lfm2", &detail::register_lfm2_architecture, &add_lfm2_chat_profile},
    {"granite", &detail::register_granite_architecture, &add_granite_chat_profile},
    {"gemma4", &detail::register_gemma4_architecture, &add_gemma4_chat_profile},
    {"minicpm5", &detail::register_minicpm5_architecture, &add_minicpm5_chat_profile},
    {"nanbeige42", &detail::register_nanbeige42_architecture, &add_nanbeige42_chat_profile},
    {"smollm3", &detail::register_smollm3_architecture, &add_smollm3_chat_profile},
    {"qwen35", &detail::register_qwen35_architecture, &add_qwen35_chat_profile},
    {"nemotron-h", &detail::register_nemotron_h_architecture, &add_nemotron_h_chat_profile},
};

class ArchitectureFamilyModule final : public IRuntimeModule {
public:
    ArchitectureFamilyModule(std::string_view module_id,
                             ArchitectureRegistrar architecture,
                             ChatRegistrar chat)
        : id_(module_id), architecture_(architecture), chat_(chat) {}

    std::string_view id() const override { return id_; }

    void register_into(RuntimeBuilder& builder) const override {
        architecture_(builder.architecture_catalog_for_registration());
        chat_(builder.chat_profile_catalog_for_registration());
    }

private:
    std::string_view id_;
    ArchitectureRegistrar architecture_;
    ChatRegistrar chat_;
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
    for (const BuiltinFamilyRegistration& family : kBuiltinFamilies) {
        modules.push_back(std::make_unique<ArchitectureFamilyModule>(
            family.module_id, family.architecture, family.chat));
    }
    modules.push_back(std::make_unique<BuiltinTokenizerModule>());
    modules.push_back(std::make_unique<Gemma4VisionModule>());
    modules.push_back(std::make_unique<Qwen35VisionModule>());
    return modules;
}

} // namespace celeg
