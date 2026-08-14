namespace celeg::prefill_detail {

void run_latent_attention(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    LayerCommon& common_layer,
    const CompiledLayerProgram& semantics,
    int rows) {
    const AttentionSpec& layout = attention.layout;
    const auto* latent = layout.latent_state();
    if (!latent) {
        throw std::logic_error(
            "CUDA latent prefill selected without latent attention state");
    }
    if (model.resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
        throw std::invalid_argument(
            "CUDA latent attention requires BF16 state storage");
    }

    if (latent->factorized) {
        run_factorized_latent_attention(
            model, attention, owner, common_layer, semantics, rows);
        return;
    }

    if (layout.output_gate.enabled() || layout.multi_axis_position()) {
        throw std::invalid_argument(
            "CUDA projected latent attention does not support query gates or M-RoPE yet");
    }

    run_projected_latent_attention(
        model, attention, owner, common_layer, semantics, rows);
}

} // namespace celeg::prefill_detail
