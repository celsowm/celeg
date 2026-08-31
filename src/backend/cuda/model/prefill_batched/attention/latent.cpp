#include "latent_path.hpp"

namespace celeg::prefill_detail {

void run_latent_attention(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    const CompiledLayerProgram& semantics,
    int rows) {
    const AttentionSpec& layout = attention.layout;
    if (model.resources_.options().kv_cache_mode == KvCacheMode::Int8) {
        throw std::invalid_argument(
            "CUDA latent attention requires BF16 state storage");
    }

    switch (latent_prefill_path(layout)) {
    case LatentPrefillPath::Factorized:
        run_factorized_latent_attention(
            model, attention, owner, semantics, rows);
        return;
    case LatentPrefillPath::Projected:
        if (layout.output_gate.has_value() || layout.multi_axis_position()) {
            throw std::invalid_argument(
                "CUDA projected latent attention does not support query gates or M-RoPE yet");
        }
        run_projected_latent_attention(
            model, attention, owner, semantics, rows);
        return;
    }

    throw std::logic_error("unreachable latent prefill path");
}

}
