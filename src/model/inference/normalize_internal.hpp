#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/model/inference.hpp"

namespace celeg {

/// Normalization phases defined in inventory.cpp and consumed by
/// build_inference_input in tensor_inventory.cpp. Internal to
/// src/model/inference; not part of the public model API.

void normalize_attention_schedule(const CheckpointMetadata& source,
                                  NormalizedModelMetadata& metadata);

void normalize_structural_norm_schedule(const CheckpointMetadata& source,
                                        NormalizedModelMetadata& metadata);

void normalize_rope_scaling(const CheckpointMetadata& source,
                            NormalizedModelMetadata& metadata);

}
