#pragma once

#include "../canonical_internal.hpp"

namespace celeg::inference_detail {

void bind_global_tensors(CanonicalInferenceContext& context);
void validate_query_key_norm_consistency(
    CanonicalInferenceContext& context, int layer);
void infer_and_bind_layer_norms(
    CanonicalInferenceContext& context, int layer);
void resolve_layer_feed_forward(
    CanonicalInferenceContext& context, int layer);
void apply_attention_output_scale(CanonicalInferenceContext& context);

}
