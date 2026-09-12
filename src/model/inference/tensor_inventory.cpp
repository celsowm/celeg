#include "celeg/model/inference.hpp"

#include "normalize_internal.hpp"
#include "celeg/checkpoint/packed/nvfp4.hpp"
#include "support.hpp"

#include <algorithm>
#include <unordered_map>

namespace celeg {

TensorInventory::TensorInventory(std::vector<TensorInventoryEntry> entries)
    : entries_(std::move(entries)) {
    std::unordered_map<std::string, const TensorInventoryEntry*> by_name;
    by_name.reserve(entries_.size());
    for (const auto& entry : entries_) by_name.emplace(entry.name, &entry);
    std::vector<TensorInventoryEntry> derived;
    for (const auto& entry : entries_) {
        constexpr std::string_view packed_suffix = "_packed";
        if (!entry.name.ends_with(packed_suffix)) continue;
        const std::string base = entry.name.substr(
            0, entry.name.size() - packed_suffix.size());
        if (by_name.contains(base)) continue;
        if (entry.dtype == TensorDType::I32 && entry.shape.size() == 2) {
            const auto scale_it = by_name.find(base + "_scale");
            const auto shape_it = by_name.find(base + "_shape");
            if (scale_it == by_name.end() || shape_it == by_name.end() ||
                scale_it->second->dtype != TensorDType::BF16 ||
                scale_it->second->shape.size() != 2 ||
                shape_it->second->dtype != TensorDType::I64 ||
                shape_it->second->shape != std::vector<int64_t>{2}) continue;
            const int64_t rows = entry.shape[0];
            const int64_t packed_words = entry.shape[1];
            const int64_t scale_columns = scale_it->second->shape[1];
            if (rows <= 0 || packed_words <= 0 || scale_columns <= 0 ||
                scale_it->second->shape[0] != rows) continue;
            const int64_t cols = scale_columns == 1
                ? packed_words * 4 : scale_columns * 32;
            const int64_t expected_words = scale_columns == 1
                ? (cols + 3) / 4 : (cols + 7) / 8;
            if (expected_words != packed_words) continue;
            derived.push_back({base, {rows, cols}, TensorDType::Quantized});
        } else if (entry.dtype == TensorDType::U8 && entry.shape.size() == 2) {
            /// NVFP4-pack-quantized (compressed-tensors): "<base>_packed" is
            /// [rows, cols/2] nibble-packed, alongside a per-16-block
            /// "<base>_scale" and a per-tensor "<base>_global_scale". See
            /// celeg/checkpoint/packed/nvfp4.hpp, which loads the same three
            /// sidecars by the same naming convention once the loader binds
            /// this derived entry to an actual weight.
            const auto scale_it = by_name.find(base + "_scale");
            const auto global_scale_it = by_name.find(base + "_global_scale");
            if (scale_it == by_name.end() || global_scale_it == by_name.end() ||
                scale_it->second->dtype != TensorDType::F8_E4M3 ||
                scale_it->second->shape.size() != 2 ||
                global_scale_it->second->dtype != TensorDType::F32 ||
                global_scale_it->second->shape != std::vector<int64_t>{1}) continue;
            const int64_t rows = entry.shape[0];
            const int64_t packed_cols = entry.shape[1];
            const int64_t scale_columns = scale_it->second->shape[1];
            if (rows <= 0 || packed_cols <= 0 || scale_columns <= 0 ||
                scale_it->second->shape[0] != rows) continue;
            const int64_t cols = packed_cols * 2;
            if (scale_columns * kNvfp4PackedBlockSize != cols) continue;
            derived.push_back({base, {rows, cols}, TensorDType::Quantized});
        }
    }
    entries_.insert(entries_.end(), derived.begin(), derived.end());
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
        /// Rank 5 covers the vision-tower temporal patch-embed conv
        /// (SafetensorProjectionProvider requires exactly rank 5: [hidden,
        /// channels, temporal, patch_h, patch_w]); nothing else in the
        /// generic inference pipeline depends on this bound being tighter.
        if (name.empty() || tensor.shape.size() > 5) {
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
    inference_detail::clear_metadata_consumption();
    NormalizedModelMetadata metadata = normalize_model_metadata(checkpoint.metadata);
    normalize_attention_schedule(checkpoint.metadata, metadata);
    normalize_structural_norm_schedule(checkpoint.metadata, metadata);
    normalize_rope_scaling(checkpoint.metadata, metadata);
    /// Tensor inventory is built before the consumption-ledger gate so the
    /// gate can condition tower-absence rules (linear/vision keys) on the
    /// grammars actually present instead of guessing from config alone.
    TensorInventory inventory = build_tensor_inventory(*checkpoint.repository);
    /// Consumption ledger gate runs last, after every resolver read, so any
    /// semantic key left unread warns (fails under `CELEG_STRICT_SEMANTICS=1`)
    /// instead of silently dropping mathematics.
    inference_detail::reject_unknown_semantic_metadata(checkpoint.metadata, inventory);
    return {std::move(metadata), std::move(inventory),
            checkpoint.metadata.source_format, checkpoint.metadata.architecture_type()};
}


}
