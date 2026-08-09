#pragma once

#include "celeg/model/resolved.hpp"

namespace celeg::descriptor_detail {

void append_attention_weight_requirements(ResolvedModel& model,
                                          const RuntimeTopology& topology,
                                          int layer, int physical_layer);
void append_short_conv_weight_requirements(ResolvedModel& model,
                                           const RuntimeTopology& topology,
                                           int layer, int physical_layer);
void append_gated_delta_net_weight_requirements(ResolvedModel& model,
                                                const RuntimeTopology& topology,
                                                int layer, int physical_layer);
void append_mamba2_weight_requirements(ResolvedModel& model,
                                       const RuntimeTopology& topology,
                                       int layer, int physical_layer);
void append_dense_ffn_weight_requirements(ResolvedModel& model,
                                          const RuntimeTopology& topology,
                                          int layer, int physical_layer,
                                          int intermediate);
void append_mlp_only_weight_requirements(ResolvedModel& model,
                                          const RuntimeTopology& topology,
                                          int layer, int physical_layer);
void append_moe_weight_requirements(ResolvedModel& model,
                                    const RuntimeTopology& topology,
                                    int layer, int physical_layer);

} // namespace celeg::descriptor_detail
