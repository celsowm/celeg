#include "../../attention_output_gate.hpp"
#include "../../attention_projection.hpp"
#include "../../attention_qk_prepare.hpp"

namespace celeg::prefill_detail {

void require_regular_attention_bindings(const AttentionLayer& attention) {
    if (!attention.query || !attention.out) {
        throw std::logic_error(
            "CUDA prefill attention has incomplete query/output bindings");
    }
    if ((attention.key == nullptr) != (attention.value == nullptr)) {
        throw std::logic_error(
            "CUDA prefill attention must bind key/value together");
    }
}

void run_regular_attention(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    const CompiledLayerProgram& semantics,
    int rows) {
    require_regular_attention_bindings(attention);

    auto& prof = prefill_phase_profile();
    const AttentionSpec& layout = attention.layout;
    const AttentionSpec& owner_layout = owner.layout;

    prof.begin(model.stream_.get());
    project_cuda_prefill_standard_attention_qkv(model, attention, rows);
    prof.end(PrefillPhase::QkvProj, model.stream_.get());

    prof.begin(model.stream_.get());
    prepare_cuda_prefill_attention_gate(model, attention, rows);
    prepare_cuda_prefill_attention_qk({
        .layout = &layout,
        .query = model.workspace_.prefill_q_.data(),
        .key = attention.key ? model.workspace_.prefill_k_.data() : nullptr,
        .query_norm = attention.q_norm,
        .key_norm = attention.k_norm,
        .fallback_norm_epsilon = model.resources_.program_.final_norm.epsilon,
        .rows = rows,
        .stream = model.stream_.get()});
    prof.end(PrefillPhase::RopeKv, model.stream_.get());

    prof.begin(model.stream_.get());
    const AttentionCapability plan =
        select_attention_plan(model, attention, owner_layout, rows);
    store_and_attend(model, attention, owner, plan, rows);
    prof.end(PrefillPhase::Attention, model.stream_.get());

    apply_cuda_prefill_attention_gate(model, attention, rows);

    prof.begin(model.stream_.get());
    project_cuda_prefill_standard_attention_output(
        model, attention, semantics, rows);
    prof.end(PrefillPhase::AttnOut, model.stream_.get());
}

}
