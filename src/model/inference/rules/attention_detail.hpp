#pragma once

#include "../canonical_internal.hpp"

#include "celeg/model/definition.hpp"

#include <string>
#include <vector>

namespace celeg::inference_detail {

/// Attention-spec builders shared by the standard and latent rule units.
/// Moved verbatim from `rules_attention.cpp`.
AttentionSpec make_attention( const NormalizedModelMetadata& metadata, int layer, int query_heads, int key_value_heads, int head_dim, bool query_key_norm, NormGranularity query_norm_granularity = NormGranularity::PerHead, NormGranularity key_norm_granularity = NormGranularity::PerHead, NormWeightKind norm_weight_kind = NormWeightKind::Scale);
std::vector<std::string> query_norm_candidates(int layer);
std::vector<std::string> key_norm_candidates(int layer);
const TensorInventoryEntry* find_optional_unique( const TensorInventory& inventory, const std::vector<std::string>& candidates, TensorRole role, int layer);
NormGranularity infer_qk_norm_granularity( const TensorInventoryEntry& tensor, TensorRole role, int layer, int per_head_width, int whole_width);

}
