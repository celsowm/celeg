#include "celeg/backend/attention_capabilities.hpp"
#include "celeg/backend/cpu/attention_capabilities.hpp"
#include "celeg/backend/cuda/attention_capabilities.hpp"
#include "celeg/backend/metal/attention_capabilities.hpp"

namespace {

using celeg::AttentionBackendCapabilities;

constexpr bool same_capabilities(const AttentionBackendCapabilities& lhs,
                                 const AttentionBackendCapabilities& rhs) {
    return lhs.full_causal == rhs.full_causal &&
           lhs.sliding_window == rhs.sliding_window &&
           lhs.bidirectional == rhs.bidirectional &&
           lhs.prefix_lm == rhs.prefix_lm &&
           lhs.block_sparse == rhs.block_sparse &&
           lhs.dynamic_sparse == rhs.dynamic_sparse &&
           lhs.external_memory == rhs.external_memory &&
           lhs.alibi == rhs.alibi &&
           lhs.relative_position_bias == rhs.relative_position_bias &&
           lhs.no_position == rhs.no_position &&
           lhs.rope == rhs.rope &&
           lhs.multi_axis_rope == rhs.multi_axis_rope &&
           lhs.standard_execution == rhs.standard_execution &&
           lhs.latent_execution == rhs.latent_execution &&
           lhs.factorized_latent_execution == rhs.factorized_latent_execution &&
           lhs.value_norm == rhs.value_norm;
}

constexpr AttentionBackendCapabilities kExpectedCpu{
    .full_causal = true,
    .sliding_window = true,
    .bidirectional = true,
    .prefix_lm = true,
    .block_sparse = true,
    .dynamic_sparse = true,
    .external_memory = true,
    .alibi = true,
    .relative_position_bias = true,
    .no_position = true,
    .rope = true,
    .multi_axis_rope = true,
    .standard_execution = true,
    .latent_execution = true,
    .factorized_latent_execution = true,
    .value_norm = true,
};

constexpr AttentionBackendCapabilities kExpectedCuda{
    .full_causal = true,
    .sliding_window = true,
    .bidirectional = true,
    .prefix_lm = true,
    .block_sparse = true,
    .dynamic_sparse = true,
    .external_memory = false,
    .alibi = true,
    .relative_position_bias = true,
    .no_position = true,
    .rope = true,
    .multi_axis_rope = true,
    .standard_execution = true,
    .latent_execution = true,
    .factorized_latent_execution = true,
    .value_norm = true,
};

constexpr AttentionBackendCapabilities kExpectedMetal{
    .full_causal = true,
    .sliding_window = true,
    .bidirectional = false,
    .prefix_lm = false,
    .block_sparse = false,
    .dynamic_sparse = false,
    .external_memory = false,
    .alibi = true,
    .relative_position_bias = true,
    .no_position = true,
    .rope = true,
    .multi_axis_rope = true,
    .standard_execution = true,
    .latent_execution = false,
    .factorized_latent_execution = false,
    .value_norm = true,
};

static_assert(sizeof(AttentionBackendCapabilities) == 16 * sizeof(bool),
              "AttentionBackendCapabilities changed: update the parity matrix test");
static_assert(same_capabilities(celeg::cpu_attention_capabilities(), kExpectedCpu));
static_assert(same_capabilities(celeg::cuda_attention_capabilities(), kExpectedCuda));
static_assert(same_capabilities(celeg::metal_attention_capabilities(), kExpectedMetal));

}

int main() {
    return same_capabilities(celeg::cpu_attention_capabilities(), kExpectedCpu) &&
           same_capabilities(celeg::cuda_attention_capabilities(), kExpectedCuda) &&
           same_capabilities(celeg::metal_attention_capabilities(), kExpectedMetal)
        ? 0 : 1;
}
