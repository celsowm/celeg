#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/model/inference.hpp"

#include <initializer_list>
#include <optional>
#include <string_view>
#include <vector>

namespace celeg {

/// Schedule helpers defined in `attention.cpp` and shared with the norm
/// phase in `norm.cpp`.
LayerScopedValue<AttentionPatternKind> attention_pattern_metadata(
    const CheckpointMetadata& metadata,
    const std::optional<int>& layer_count,
    std::vector<EvidenceItem>& evidence);
LayerScopedValue<bool> boolean_schedule_metadata(
    const CheckpointMetadata& metadata,
    std::initializer_list<std::string_view> aliases,
    const std::optional<int>& layer_count,
    std::vector<EvidenceItem>& evidence,
    std::string_view fact);

/// Rope helpers defined in the parent `inventory.cpp` (rope owner) and
/// consumed by `normalize_attention_schedule` in `attention.cpp`.
void apply_rope_layer_flags(const CheckpointMetadata& source,
                            NormalizedModelMetadata& metadata);
void expand_per_pattern_rope(const CheckpointMetadata& source,
                             NormalizedModelMetadata& metadata);

}
