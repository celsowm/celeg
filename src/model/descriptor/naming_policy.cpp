#include "detail.hpp"

#include <string>
#include <utility>

namespace celeg::descriptor_detail {
namespace {

class DescriptorNamingPolicy final : public ITensorNamingPolicy {
public:
    explicit DescriptorNamingPolicy(const Descriptor& descriptor) {
        for (const BindingPattern& pattern : descriptor.bindings) {
            bindings_[pattern.role] = pattern.candidates;
        }
    }

    std::vector<std::string> candidates(const TensorRequest& request) const override {
        const auto it = bindings_.find(request.role);
        if (it == bindings_.end()) return {};
        std::vector<std::string> result;
        result.reserve(it->second.size());
        for (const std::string& pattern : it->second) {
            std::string value = pattern;
            replace(value, "{layer}", request.layer >= 0 ? std::to_string(request.layer) : "");
            replace(value, "{physical_layer}", request.physical_layer >= 0
                ? std::to_string(request.physical_layer) : "");
            replace(value, "{expert}", request.expert >= 0 ? std::to_string(request.expert) : "");
            result.push_back(std::move(value));
        }
        return result;
    }

private:
    static void replace(std::string& value, std::string_view needle,
                        std::string_view replacement) {
        size_t position = 0;
        while ((position = value.find(needle, position)) != std::string::npos) {
            value.replace(position, needle.size(), replacement);
            position += replacement.size();
        }
    }

    std::unordered_map<TensorRole, std::vector<std::string>> bindings_;
};

}

std::unique_ptr<ITensorNamingPolicy> create_naming_policy(const Descriptor& descriptor) {
    return std::make_unique<DescriptorNamingPolicy>(descriptor);
}

}
