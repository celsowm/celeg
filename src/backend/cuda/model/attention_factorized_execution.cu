#include "detail/compiled_model.hpp"
#include "attention_kv_store.hpp"
#include "attention_latent_dispatch.hpp"
#include "attention_layer_support.hpp"
#include "attention_projection.hpp"
#include "attention_qk_prepare.hpp"
#include "backend/cuda/phase_profile.hpp"

#include <stdexcept>

namespace celeg {

PhaseProfile& decode_phase_profile();

void CudaCompiledModel::enqueue_decode_factorized_latent_attention(
    Layer& layer, int layer_index) {
    const CompiledLayerProgram& semantics = resources_.program_.layers.at(
        static_cast<std::size_t>(layer_index));
    const auto* compiled_attention =
        std::get_if<CompiledAttentionProgram>(&semantics.mixer);
    if (!compiled_attention || compiled_attention->execution.kind !=
        AttentionExecutionKind::FactorizedLatent) {
        throw std::logic_error(
            "CUDA factorized attention has an invalid execution descriptor");
    }
    AttentionLayer* attention = as_attention(layer);
    if (!attention) throw std::logic_error("CUDA layer is not attention");
    if (resources_.options().kv_cache_mode == KvCacheMode::Int8) {
        throw std::invalid_argument(
            "CUDA latent attention requires BF16 state storage");
    }
    const AttentionSpec& layout = attention->layout;
    const float qk_norm_epsilon = layout.query_norm
        ? layout.query_norm->epsilon : resources_.program_.final_norm.epsilon;
    const CudaAttentionOwner resolved_owner = resolve_cuda_attention_owner(
        *attention, layer_index, resources_.layers_);
    AttentionLayer& owner = *resolved_owner.layer;
    require_cuda_factorized_latent_bindings(*attention);

    decode_phase_profile().begin(stream_.get());
    project_cuda_factorized_latent_attention_qkv(*this, *attention);
    decode_phase_profile().end(DecodePhase::Projection, stream_.get());

    prepare_cuda_factorized_latent_attention_qk({
        .layout = &layout,
        .query_rope = workspace_.latent_query_rope_.data(),
        .key_rope = workspace_.latent_key_rope_.data(),
        .norm_epsilon = qk_norm_epsilon,
        .rows = 1,
        .device_position = position_device_.data(),
        .stream = stream_.get()});
    store_cuda_latent_kv_contiguous(*this, *attention, owner);
    dispatch_cuda_latent_attention_contiguous(*this, *attention, owner);
    project_cuda_factorized_latent_attention_output(*this, *attention, semantics);
}

}
