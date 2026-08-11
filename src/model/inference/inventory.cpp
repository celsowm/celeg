#include "celeg/model/inference.hpp"

#include "support.hpp"


#include <algorithm>
#include <regex>
#include <unordered_set>

namespace celeg {


TensorInventory::TensorInventory(std::vector<TensorInventoryEntry> entries)
    : entries_(std::move(entries)) {
    std::sort(entries_.begin(), entries_.end(),
              [](const auto& left, const auto& right) { return left.name < right.name; });
    for (size_t index = 1; index < entries_.size(); ++index) {
        if (entries_[index - 1].name == entries_[index].name) {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedTensorLayout,
                "checkpoint contains duplicate tensor identity: " + entries_[index].name);
        }
    }
}

const TensorInventoryEntry* TensorInventory::find(std::string_view name) const noexcept {
    const auto it = std::lower_bound(entries_.begin(), entries_.end(), name,
        [](const auto& entry, std::string_view value) { return entry.name < value; });
    return it != entries_.end() && it->name == name ? &*it : nullptr;
}

std::vector<const TensorInventoryEntry*> TensorInventory::with_prefix(
    std::string_view prefix) const {
    std::vector<const TensorInventoryEntry*> result;
    for (const auto& entry : entries_) {
        if (entry.name.starts_with(prefix)) result.push_back(&entry);
    }
    return result;
}

TensorInventory build_tensor_inventory(const IWeightRepository& repository) {
    std::vector<TensorInventoryEntry> entries;
    for (const std::string& name : repository.names()) {
        const HostTensorView tensor = repository.tensor(name);
        if (name.empty() || tensor.shape.empty() || tensor.shape.size() > 4) {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedTensorLayout,
                "tensor inventory contains invalid tensor metadata: " + name);
        }
        entries.push_back({name, tensor.shape, tensor.dtype});
    }
    return TensorInventory(std::move(entries));
}

InferenceInput build_inference_input(const CheckpointView& checkpoint) {
    if (!checkpoint.repository) {
        inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                               "automatic checkpoint resolution requires a tensor repository");
    }
    return {normalize_model_metadata(checkpoint.metadata),
            build_tensor_inventory(*checkpoint.repository), checkpoint.metadata.source_format};
}

} // namespace celeg
