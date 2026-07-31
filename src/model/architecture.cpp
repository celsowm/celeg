#include "celeg/model/architecture.hpp"

#include "celeg/detail/model/builtin_architectures.hpp"

#include <algorithm>
#include <stdexcept>

namespace celeg {

void ArchitectureCatalog::add(std::unique_ptr<IArchitecture> architecture) {
    if (frozen_) throw std::logic_error("architecture catalog is frozen");
    if (!architecture || architecture->id().empty()) {
        throw std::invalid_argument("architecture must have a non-empty id");
    }
    if (find(architecture->id()) != nullptr) {
        throw std::invalid_argument("duplicate architecture id: " +
                                    std::string(architecture->id()));
    }
    architectures_.push_back(std::move(architecture));
}

void ArchitectureCatalog::freeze() {
    frozen_ = true;
}

const IArchitecture& ArchitectureCatalog::select(
    const CheckpointMetadata& metadata) const {
    const IArchitecture* selected = nullptr;
    int selected_specificity = -1;
    for (const auto& architecture : architectures_) {
        const ProbeResult result = architecture->probe(metadata);
        if (!result.supported) continue;
        if (result.specificity > selected_specificity) {
            selected = architecture.get();
            selected_specificity = result.specificity;
        } else if (result.specificity == selected_specificity) {
            throw std::runtime_error("multiple architectures match checkpoint metadata");
        }
    }
    if (selected == nullptr) {
        throw std::runtime_error("no registered architecture supports checkpoint metadata");
    }
    return *selected;
}

const IArchitecture* ArchitectureCatalog::find(std::string_view id) const {
    const auto it = std::find_if(architectures_.begin(), architectures_.end(),
        [id](const auto& architecture) { return architecture->id() == id; });
    return it == architectures_.end() ? nullptr : it->get();
}

std::vector<std::string_view> ArchitectureCatalog::ids() const {
    std::vector<std::string_view> result;
    result.reserve(architectures_.size());
    for (const auto& architecture : architectures_) result.push_back(architecture->id());
    return result;
}

std::shared_ptr<const ArchitectureCatalog> create_builtin_architecture_catalog() {
    auto catalog = std::make_shared<ArchitectureCatalog>();
    catalog->add(detail::make_lfm2_architecture());
    catalog->add(detail::make_granite_architecture());
    catalog->freeze();
    return catalog;
}

} // namespace celeg
