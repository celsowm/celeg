#include "celeg/model/runtime_modules.hpp"

#include "celeg/detail/model/builtin_architectures.hpp"
#include "celeg/models/gemma4/vision.hpp"
#include "celeg/models/qwen35/vision.hpp"
#include "celeg/runtime/context.hpp"
#include "celeg/text/chat_template.hpp"
#include "celeg/text/tokenizer.hpp"

namespace celeg {
namespace {

using ArchitectureRegistrar = void (*)(ArchitectureCatalog&);

struct BuiltinFamilyRegistration {
    std::string_view module_id;
    ArchitectureRegistrar register_architecture;
};

constexpr BuiltinFamilyRegistration kBuiltinFamilies[] = {
    {"lfm2", &detail::register_lfm2_architecture},
    {"granite", &detail::register_granite_architecture},
    {"gemma4", &detail::register_gemma4_architecture},
    {"minicpm5", &detail::register_minicpm5_architecture},
    {"smollm3", &detail::register_smollm3_architecture},
    {"qwen35", &detail::register_qwen35_architecture},
    {"nemotron-h", &detail::register_nemotron_h_architecture},
};

class ArchitectureFamilyModule final : public IRuntimeModule {
public:
    ArchitectureFamilyModule(std::string_view module_id,
                             ArchitectureRegistrar registrar)
        : id_(module_id), registrar_(registrar) {}

    std::string_view id() const override { return id_; }

    void register_into(RuntimeBuilder& builder) const override {
        registrar_(builder.architecture_catalog_for_registration());
    }

private:
    std::string_view id_;
    ArchitectureRegistrar registrar_;
};

class BuiltinTextModule final : public IRuntimeModule {
public:
    std::string_view id() const override { return "builtin-text"; }

    void register_into(RuntimeBuilder& builder) const override {
        add_builtin_chat_profiles(builder.chat_profile_catalog_for_registration());
        builder.add_tokenizer_provider(make_builtin_tokenizer_provider());
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

void add_builtin_architectures(ArchitectureCatalog& catalog) {
    for (const BuiltinFamilyRegistration& family : kBuiltinFamilies) {
        family.register_architecture(catalog);
    }
}

std::vector<std::unique_ptr<IRuntimeModule>> make_builtin_runtime_modules() {
    std::vector<std::unique_ptr<IRuntimeModule>> modules;
    for (const BuiltinFamilyRegistration& family : kBuiltinFamilies) {
        modules.push_back(std::make_unique<ArchitectureFamilyModule>(
            family.module_id, family.register_architecture));
    }
    modules.push_back(std::make_unique<BuiltinTextModule>());
    modules.push_back(std::make_unique<Gemma4VisionModule>());
    modules.push_back(std::make_unique<Qwen35VisionModule>());
    return modules;
}

} // namespace celeg
