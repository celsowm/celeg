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

std::vector<std::unique_ptr<IRuntimeModule>> make_builtin_runtime_modules() {
    using namespace detail;
    std::vector<std::unique_ptr<IRuntimeModule>> modules;
    modules.push_back(std::make_unique<ArchitectureFamilyModule>(
        "lfm2", &register_lfm2_architecture));
    modules.push_back(std::make_unique<ArchitectureFamilyModule>(
        "granite", &register_granite_architecture));
    modules.push_back(std::make_unique<ArchitectureFamilyModule>(
        "gemma4", &register_gemma4_architecture));
    modules.push_back(std::make_unique<ArchitectureFamilyModule>(
        "minicpm5", &register_minicpm5_architecture));
    modules.push_back(std::make_unique<ArchitectureFamilyModule>(
        "smollm3", &register_smollm3_architecture));
    modules.push_back(std::make_unique<ArchitectureFamilyModule>(
        "qwen35", &register_qwen35_architecture));
    modules.push_back(std::make_unique<ArchitectureFamilyModule>(
        "nemotron-h", &register_nemotron_h_architecture));
    modules.push_back(std::make_unique<BuiltinTextModule>());
    modules.push_back(std::make_unique<Gemma4VisionModule>());
    modules.push_back(std::make_unique<Qwen35VisionModule>());
    return modules;
}

} // namespace celeg
