#include "celeg/model/runtime_modules.hpp"

#include "celeg/detail/model/builtin_architectures.hpp"
#include "celeg/models/gemma4/vision.hpp"
#include "celeg/runtime/context.hpp"
#include "celeg/text/chat_template.hpp"

namespace celeg {
namespace {

class BuiltinFamilyModule final : public IRuntimeModule {
public:
    std::string_view id() const override { return "builtin-families"; }

    void register_into(RuntimeBuilder& builder) const override {
        add_builtin_architectures(builder.architecture_catalog_for_registration());
        add_builtin_chat_profiles(builder.chat_profile_catalog_for_registration());
        builder.vision_catalog_for_registration().add(
            make_gemma4_vision_provider_factory());
    }
};

} // namespace

std::vector<std::unique_ptr<IRuntimeModule>> make_builtin_runtime_modules() {
    std::vector<std::unique_ptr<IRuntimeModule>> modules;
    modules.push_back(std::make_unique<BuiltinFamilyModule>());
    return modules;
}

} // namespace celeg
