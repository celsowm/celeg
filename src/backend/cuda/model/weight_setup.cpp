#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/weight_policy.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/checkpoint/packed/int4.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/backend/cuda/weight_setup_support.hpp"
#include "celeg/backend/cuda/weight_setup.hpp"
#include "celeg/backend/cuda/moe.hpp"
#include "celeg/backend/cuda/moe/expert_source.hpp"

#include <filesystem>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace celeg {

namespace {
std::string layer_name(int index, const std::string& suffix) {
    return cuda_layer_name(index, suffix);
}

std::string tensor_name(std::span<const TensorRequest> requests, TensorRole role,
                        int layer = -1) {
    return cuda_tensor_name(requests, role, layer);
}

}

void CudaCompiledModel::load_checkpoint_weights(
    const std::string& model_path,
    const detail::ModelBootstrap& bootstrap) {
    CudaWeightSetup::load(*this, model_path, bootstrap,
        [this](const IWeightRepository& repo) {
    configure_cuda_expert_resources(*this);
    const int mtp_layer_count = resources_.options_.enable_mtp
        ? resources_.dims_.mtp_num_hidden_layers : 0;
    const int resource_layer_count = resources_.shape_.num_hidden_layers +
        mtp_layer_count;
    workspace_.expert_caches_.resize(static_cast<size_t>(resource_layer_count));
    if (resources_.weights_->expert_controllers.size() <
        static_cast<size_t>(resource_layer_count)) {
        resources_.weights_->expert_controllers.resize(
            static_cast<size_t>(resource_layer_count));
    }
    workspace_.expert_catalog_.resize(static_cast<size_t>(resource_layer_count));
    if (resources_.weights_->expert_catalog.size() <
        static_cast<size_t>(resource_layer_count)) {
        resources_.weights_->expert_catalog.resize(
            static_cast<size_t>(resource_layer_count));
    }

    resources_.layers_.reserve(static_cast<size_t>(resources_.shape_.num_hidden_layers));
    std::vector<int> shared_owner(2, -1);
    for (int i = 0; i < resources_.shape_.num_hidden_layers; ++i) {
        LayerCommon common_layer;
        const CompiledLayerProgram& semantic_layer = resources_.program_.layers.at(
            static_cast<size_t>(i));
        const bool mixer_only_layer =
            std::holds_alternative<std::monostate>(semantic_layer.feed_forward);
        const auto load_norm = [&](TensorRole role, const NormSpec& spec) {
            const std::string name = spec.weightless()
                ? std::string{} : tensor_name(resources_.model_.weight_plan.requests, role, i);
            return resources_.weight_loader_->load_rms_norm_weight(
                repo, name, {resources_.program_.hidden}, spec.weight_kind);
        };
        common_layer.operator_norm = resources_.weight_loader_->load_rms_norm_weight(
            repo, semantic_layer.operator_norm.weightless() ? std::string{} :
                tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionInputNorm, i),
            {resources_.program_.hidden}, semantic_layer.operator_norm.weight_kind);
        if (!mixer_only_layer) {
            common_layer.ffn_norm = load_norm(TensorRole::FfnInputNorm,
                                              *semantic_layer.feed_forward_norm);
        }
        if (!mixer_only_layer && semantic_layer.post_attention_norm.has_value()) {
            common_layer.post_attention_norm = load_norm(
                TensorRole::AttentionPostNorm, *semantic_layer.post_attention_norm);
        }
        if (!mixer_only_layer && semantic_layer.post_feed_forward_norm.has_value()) {
            common_layer.post_feed_forward_norm = load_norm(
                TensorRole::FfnOutputNorm, *semantic_layer.post_feed_forward_norm);
        }
        if (resources_.program_.per_layer_input.enabled) {
            common_layer.per_layer_input_gate = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::PerLayerInputGate, i),
                {resources_.program_.per_layer_input.input_size, resources_.program_.hidden});
            common_layer.per_layer_projection = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::PerLayerProjection, i),
                {resources_.program_.hidden, resources_.program_.per_layer_input.input_size});
            common_layer.per_layer_input_norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::PerLayerInputNorm, i),
                {resources_.program_.hidden});
            common_layer.layer_scalar = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::LayerScalar, i), {1});
        }
        if (mixer_only_layer) {
            common_layer.feed_forward = std::monostate{};
        } else if (const auto* moe_program =
                       std::get_if<MoeLayerProgram>(&semantic_layer.feed_forward)) {
            const MoeLayerProgram& moe_semantics = *moe_program;
            const int E = moe_semantics.router.expert_count;
            const int inter = moe_semantics.routed.mlp.intermediate_size;
            /// Routed-expert tensor names and the packed-vs-individual layout
            /// decision come from the resolved weight plan; setup never
            /// re-derives them from checkpoint tensor spellings.
            const MoeExpertTensorNames expert_names = moe_expert_tensor_names(
                resources_.model_.weight_plan.requests, i, E);
            const float* expert_bias = nullptr;
            if (moe_semantics.router.has_expert_bias) {
                expert_bias = resources_.weight_loader_->load_f32_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::MoeRouterBias, i),
                    {static_cast<int64_t>(E)});
            }
            const LinearWeight* router = resources_.weight_loader_->load_router_weight_named(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::MoeRouter, i),
                E, resources_.program_.hidden);

            DeviceBuffer<float>& router_float = workspace_.moe_router_float_[static_cast<size_t>(i)];
            router_float.reset(static_cast<size_t>(E) * resources_.program_.hidden);
            launch_cast_bf16_to_float(
                std::get<Bf16LinearStorage>(router->storage).data, router_float.data(),
                static_cast<int>(E) * resources_.program_.hidden, stream_.get());

            MoeFfnWeights moe_weights{};
            moe_weights.router = router;
            moe_weights.expert_bias = expert_bias;
            moe_weights.router_float = router_float.data();

            if (moe_semantics.shared) {
                const int shared_inter = moe_semantics.shared->mlp.intermediate_size;
                moe_weights.shared_w13 = resources_.weight_loader_->load_concat_linear_weight(
                    repo, layer_name(i, "shared_expert.w13.weight"),
                    {
                        {tensor_name(resources_.model_.weight_plan.requests,
                                     TensorRole::MoeSharedGate, i),
                         {shared_inter, resources_.program_.hidden}},
                        {tensor_name(resources_.model_.weight_plan.requests,
                                     TensorRole::MoeSharedUp, i),
                         {shared_inter, resources_.program_.hidden}},
                    });
                moe_weights.shared_w2 = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::MoeSharedDown, i),
                    {resources_.program_.hidden, shared_inter});
                const auto gate_request = std::find_if(
                    resources_.model_.weight_plan.requests.begin(),
                    resources_.model_.weight_plan.requests.end(),
                    [i](const TensorRequest& request) {
                        return request.role == TensorRole::MoeSharedGateWeight &&
                               request.layer == i && request.expert == -1;
                    });
                if (gate_request != resources_.model_.weight_plan.requests.end()) {
                    moe_weights.shared_gate = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::MoeSharedGateWeight, i),
                        {1, resources_.program_.hidden});
                }
            }

            if (workspace_.expert_offload_plan_.enabled) {
                /// The routed-expert checkpoint layout (packed vs individual)
                /// and every expert tensor name were already decided once,
                /// from real checkpoint evidence, when the weight plan was
                /// built (see append_moe() / bind_moe()); setup consumes
                /// expert_names instead of probing repo.contains() on literal
                /// tensor-name spellings. Checkpoint-packed int4 experts are
                /// addressed by a virtual base name, so they are probed via
                /// the int4 convention rather than a direct tensor view.
                const std::string& probe_name = expert_names.packed()
                    ? expert_names.packed_gate_up : expert_names.gate.front();
                if (!has_packed_int4_matrix(repo, probe_name)) {
                    const HostTensorView expert_probe = repo.tensor(probe_name);
                    if (expert_probe.dtype == TensorDType::Quantized) {
                        throw std::invalid_argument(
                            "native GGUF MoE experts do not support BF16 offload; "
                            "disable expert offload to keep packed Q4/Q6 weights "
                            "resident");
                    }
                }
                if (resources_.options_.expert_offload.backing == ExpertBackingMode::DiskCached) {
                    std::vector<ExpertLocation> catalog =
                        resources_.weight_loader_->build_expert_catalog(
                            repo, expert_names, E, inter,
                            resources_.program_.hidden);
                    workspace_.expert_catalog_[static_cast<size_t>(i)] = catalog;
                    if (resources_.weights_->expert_catalog[static_cast<size_t>(i)].empty()) {
                        resources_.weights_->expert_catalog[static_cast<size_t>(i)] = catalog;
                    }

                    size_t gate_up_bytes = 2 * inter * resources_.program_.hidden * sizeof(__nv_bfloat16);
                    size_t down_bytes = resources_.program_.hidden * inter * sizeof(__nv_bfloat16);
                    auto cache = std::make_unique<ExpertLayerCache>(
                        E, workspace_.expert_offload_plan_.experts_per_layer,
                        gate_up_bytes, down_bytes);
                    cache->set_policy(resources_.options_.expert_offload.policy);
                    std::vector<const __nv_bfloat16*> empty_host_dev(static_cast<size_t>(E), nullptr);
                    cache->set_host_sources(empty_host_dev, empty_host_dev);

                    auto controller = std::make_unique<ResidencyController>();
                    controller->cache = std::move(cache);
                    controller->transfer_stream = std::make_unique<CudaStream>();

                    if (workspace_.expert_offload_plan_.experts_per_layer > 0) {
                        for (int s = 0; s < workspace_.expert_offload_plan_.experts_per_layer; ++s) {
                            const ExpertLocation& loc = catalog[static_cast<size_t>(s)];
                            ExpertHostLease lease = resources_.weights_->host_expert_cache->acquire(i, s, [&](std::span<std::byte> payload) {
                                if (!resources_.weights_->expert_source) {
                                    throw std::runtime_error("CUDA expert source is not initialized");
                                }
                                resources_.weights_->expert_source->read(i, s, payload);
                            });
                            controller->cache->promote(
                                           s, s,
                                           reinterpret_cast<const __nv_bfloat16*>(lease.payload()),
                                           reinterpret_cast<const __nv_bfloat16*>(lease.payload() + loc.w1.bytes + loc.w3.bytes),
                                           controller->transfer_stream->get());
                            auto ev = std::make_unique<CudaEvent>();
                            ev->record(controller->transfer_stream->get());
                            controller->inflight_transfers.push_back({std::move(lease), std::move(ev)});
                        }
                    }
                    CELEG_CUDA(cudaStreamSynchronize(controller->transfer_stream->get()));
                    controller->inflight_transfers.clear();
                    moe_weights.storage = OffloadedExpertWeights{
                        controller->cache->gate_up_ptrs(), controller->cache->down_ptrs()};
                    resources_.weights_->expert_controllers[static_cast<size_t>(i)] = std::move(controller);
                    workspace_.expert_caches_[static_cast<size_t>(i)] = resources_.weights_->expert_controllers[static_cast<size_t>(i)]->cache.get();
                } else {
                    WeightLoader::HostExpertLayer host_layer =
                        resources_.weight_loader_->load_moe_experts_host(
                            repo, expert_names, E, inter,
                            resources_.program_.hidden,
                            workspace_.host_expert_store_,
                            resources_.options_.expert_offload.host_mode);
                    auto cache = std::make_unique<ExpertLayerCache>(
                        E, workspace_.expert_offload_plan_.experts_per_layer,
                        host_layer.gate_up_bytes, host_layer.down_bytes);
                    cache->set_policy(resources_.options_.expert_offload.policy);
                    cache->set_host_sources(host_layer.gate_up_host_dev,
                                            host_layer.down_host_dev);

                    auto controller = std::make_unique<ResidencyController>();
                    controller->cache = std::move(cache);
                    controller->transfer_stream = std::make_unique<CudaStream>();

                    std::vector<int> seed(static_cast<size_t>(
                        workspace_.expert_offload_plan_.experts_per_layer));
                    for (int s = 0; s < workspace_.expert_offload_plan_.experts_per_layer; ++s) {
                        seed[static_cast<size_t>(s)] = s;
                    }
                    controller->cache->seed(seed, controller->transfer_stream->get());
                    CELEG_CUDA(cudaStreamSynchronize(controller->transfer_stream->get()));
                    moe_weights.storage = OffloadedExpertWeights{
                        controller->cache->gate_up_ptrs(), controller->cache->down_ptrs()};
                    resources_.weights_->expert_controllers[static_cast<size_t>(i)] = std::move(controller);
                    workspace_.expert_caches_[static_cast<size_t>(i)] = resources_.weights_->expert_controllers[static_cast<size_t>(i)]->cache.get();
                }
            } else {
                /// Same plan-driven layout decision as the offload branch
                /// above: expert_names carries it, including which storage
                /// family each resolved tensor belongs to.
                if (expert_names.packed()) {
                    const ExpertLinearWeight* gate_up =
                        resources_.weight_loader_->load_expert_linear_weight(
                            repo, expert_names.packed_gate_up,
                            E, 2 * inter, resources_.program_.hidden);
                    const ExpertLinearWeight* down =
                        resources_.weight_loader_->load_expert_linear_weight(
                            repo, expert_names.packed_down,
                            E, resources_.program_.hidden, inter);
                    moe_weights.storage = ResidentExpertWeights{gate_up, down};
                } else {
                    const ExpertLinearWeight* gate_up =
                        resources_.weight_loader_->load_moe_gate_up(
                            repo, expert_names, E, inter,
                            resources_.program_.hidden);
                    const ExpertLinearWeight* down =
                        resources_.weight_loader_->load_moe_down(
                            repo, expert_names, E, inter,
                            resources_.program_.hidden);
                    moe_weights.storage = ResidentExpertWeights{gate_up, down};
                }
            }

            common_layer.feed_forward = moe_weights;
        } else {
            const auto* dense = std::get_if<CompiledDenseFeedForwardProgram>(
                &semantic_layer.feed_forward);
            if (!dense || dense->intermediate_size <= 0) {
                throw std::runtime_error("compiled dense layer has no FFN width");
            }
            const int intermediate = dense->intermediate_size;
            const LinearWeight* w13 = resources_.weight_loader_->load_concat_linear_weight(
                repo, layer_name(i, "feed_forward.w13.weight"),
                {
                    {tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnGate, i),
                     {intermediate, resources_.program_.hidden}},
                    {tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnUp, i),
                     {intermediate, resources_.program_.hidden}},
                });
            const LinearWeight* w2 = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnDown, i),
                {resources_.program_.hidden, intermediate});
            common_layer.feed_forward = DenseFfnWeights{w13, w2};
        }

        if (const auto* compiled_attention =
                std::get_if<CompiledAttentionProgram>(&semantic_layer.mixer)) {
            AttentionLayer attention_layer;
            attention_layer.common = common_layer;
            attention_layer.layout = compiled_attention->semantics;
            std::string query_name;
            const AttentionSpec& layout = attention_layer.layout;
            if (const auto* alibi = std::get_if<AlibiBiasSpec>(&layout.bias)) {
                if (static_cast<int>(alibi->slopes.size()) != layout.query_heads) {
                    throw std::invalid_argument("CUDA ALiBi slope count does not match query heads");
                }
                attention_layer.alibi_slopes.reset(alibi->slopes.size());
                CELEG_CUDA(cudaMemcpy(
                    attention_layer.alibi_slopes.data(), alibi->slopes.data(),
                    attention_layer.alibi_slopes.bytes(), cudaMemcpyHostToDevice));
            }
            if (layout.uses_latent_state()) {
                if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                    throw std::invalid_argument(
                        "CUDA latent attention currently requires BF16 state storage");
                }
                const auto& latent = *layout.latent_state();
                const auto* factorized = latent.factorized_projection();
                const bool owns_latent_state =
                    !std::holds_alternative<SharedKvConsumer>(layout.kv_sharing);
                if (factorized) {
                    attention_layer.latent_query_projection = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentQueryProjection, i),
                        {factorized->query_rank, resources_.program_.hidden});
                    attention_layer.latent_query_expansion = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentQueryExpansion, i),
                        {layout.query_heads * (latent.nope_head_dim + latent.rope_head_dim),
                         factorized->query_rank});
                    attention_layer.latent_query_norm = resources_.weight_loader_->load_rms_norm_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentQueryNorm, i),
                        {factorized->query_rank}, factorized->query_latent_norm.weight_kind);
                    attention_layer.latent_key_projection = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentKeyProjection, i),
                        {latent.latent_rank + latent.rope_head_dim, resources_.program_.hidden});
                    attention_layer.latent_key_norm = resources_.weight_loader_->load_rms_norm_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentKeyNorm, i),
                        {latent.latent_rank}, factorized->key_latent_norm.weight_kind);
                    attention_layer.latent_expansion = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentExpansion, i),
                        {layout.query_heads * (latent.nope_head_dim + factorized->value_head_dim),
                         latent.latent_rank});
                    attention_layer.gate = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionGate, i),
                        {layout.output_gate_width(), resources_.program_.hidden});
                    attention_layer.out = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentOutput, i),
                        {resources_.program_.hidden, layout.latent_output_width()});
                    if (!std::holds_alternative<Bf16LinearStorage>(
                            attention_layer.latent_query_expansion->storage) ||
                        !std::holds_alternative<Bf16LinearStorage>(
                            attention_layer.latent_expansion->storage)) {
                        throw std::invalid_argument(
                            "CUDA factorized latent attention currently requires BF16 expansion weights");
                    }
                } else {
                attention_layer.latent_query = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::AttentionLatentQuery, i),
                    {layout.latent_query_content_width(), resources_.program_.hidden});
                if (layout.latent_query_rope_width() != 0) {
                    attention_layer.latent_query_rope = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentQueryRope, i),
                        {layout.latent_query_rope_width(), resources_.program_.hidden});
                }
                if (owns_latent_state) {
                    attention_layer.latent_key = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentKey, i),
                        {latent.latent_rank, resources_.program_.hidden});
                    attention_layer.latent_value = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentValue, i),
                        {latent.latent_rank, resources_.program_.hidden});
                    if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                        attention_layer.latent_key_rope = resources_.weight_loader_->load_linear_weight(
                            repo, tensor_name(resources_.model_.weight_plan.requests,
                                              TensorRole::AttentionLatentKeyRope, i),
                            {latent.rope_head_dim, resources_.program_.hidden});
                    }
                }
                attention_layer.out = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::AttentionLatentOutput, i),
                    {resources_.program_.hidden, layout.latent_query_content_width()});
                }
                attention_layer.state = LatentAttentionRuntimeState{};
                if (resources_.options_.allocate_local_kv_cache && owns_latent_state) {
                    auto& latent_state = std::get<LatentAttentionRuntimeState>(attention_layer.state);
                    latent_state.latent_key_cache.reset(
                        static_cast<size_t>(max_context_) * latent.latent_rank);
                    latent_state.latent_value_cache.reset(
                        static_cast<size_t>(max_context_) * latent.latent_rank);
                    if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                        latent_state.latent_key_rope_cache.reset(
                            static_cast<size_t>(max_context_) * latent.rope_head_dim);
                    }
                }
                if (kv_sharing_publishes(layout.kv_sharing)) {
                    shared_owner[static_cast<size_t>(kv_sharing_group(layout.kv_sharing))] = i;
                    attention_layer.kv_owner_layer = i;
                }
                if (!kv_sharing_shared(layout.kv_sharing)) attention_layer.kv_owner_layer = i;
                resources_.layers_.emplace_back(std::move(attention_layer));
                continue;
            }
            query_name = tensor_name(resources_.model_.weight_plan.requests,
                                     TensorRole::AttentionQuery, i);
            attention_layer.query = resources_.weight_loader_->load_linear_weight(
                repo, query_name,
                {layout.query_projection_width(), resources_.program_.hidden});
            if (layout.output_gate.has_value() && !layout.output_gate->packed_with_query) {
                attention_layer.gate = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::AttentionGate, i),
                    {layout.query_width(), resources_.program_.hidden});
            }
            if (!std::holds_alternative<SharedKvConsumer>(layout.kv_sharing)) {
                attention_layer.key = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionKey, i),
                    {layout.key_value_width(), resources_.program_.hidden});
                attention_layer.value = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionValue, i),
                    {layout.key_value_width(), resources_.program_.hidden});
            } else {
                if (kv_sharing_group(layout.kv_sharing) < 0 ||
                    kv_sharing_group(layout.kv_sharing) >= static_cast<int>(shared_owner.size()) ||
                    shared_owner[static_cast<size_t>(kv_sharing_group(layout.kv_sharing))] < 0) {
                    throw std::runtime_error("CUDA shared KV consumer has no owner");
                }
                attention_layer.kv_owner_layer =
                    shared_owner[static_cast<size_t>(kv_sharing_group(layout.kv_sharing))];
            }
            attention_layer.out = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionOutput, i),
                {resources_.program_.hidden, layout.query_width()});
            if (layout.has_query_key_norm()) {
                attention_layer.q_norm = resources_.weight_loader_->load_rms_norm_weight(
                    repo, layout.query_norm->weightless() ? std::string{} :
                        tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionQueryNorm, i),
                    {layout.head_dim}, layout.query_norm->weight_kind);
                if (attention_layer.key) {
                    attention_layer.k_norm = resources_.weight_loader_->load_rms_norm_weight(
                    repo, layout.key_norm->weightless() ? std::string{} :
                        tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionKeyNorm, i),
                        {layout.head_dim}, layout.key_norm->weight_kind);
                }
            }

            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                attention_layer.state = OrdinaryInt8KvState{};
            } else {
                attention_layer.state = OrdinaryBf16KvState{};
            }
            if (resources_.options_.allocate_local_kv_cache && attention_layer.key) {
                const size_t cache_elements = static_cast<size_t>(max_context_) *
                    static_cast<size_t>(layout.key_value_width());
                if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                    auto& ordinary_state = std::get<OrdinaryInt8KvState>(attention_layer.state);
                    ordinary_state.key_cache.reset(cache_elements);
                    ordinary_state.value_cache.reset(cache_elements);
                    const size_t scale_elements =
                        static_cast<size_t>(max_context_) *
                        static_cast<size_t>(layout.key_value_heads);
                    ordinary_state.key_scales.reset(scale_elements);
                    ordinary_state.value_scales.reset(scale_elements);
                } else {
                    auto& ordinary_state = std::get<OrdinaryBf16KvState>(attention_layer.state);
                    ordinary_state.key_cache.reset(cache_elements);
                    ordinary_state.value_cache.reset(cache_elements);
                }
            }
            if (kv_sharing_publishes(layout.kv_sharing)) {
                shared_owner[static_cast<size_t>(kv_sharing_group(layout.kv_sharing))] = i;
                attention_layer.kv_owner_layer = i;
            }
            if (!kv_sharing_shared(layout.kv_sharing)) attention_layer.kv_owner_layer = i;
            resources_.layers_.emplace_back(std::move(attention_layer));
        } else if (const auto* gated_delta =
                       std::get_if<GatedDeltaNetSpec>(&semantic_layer.mixer)) {
            GatedDeltaNetLayer gated_delta_layer;
            gated_delta_layer.common = common_layer;
            gated_delta_layer.spec = *gated_delta;
            const GatedDeltaNetSpec& spec = gated_delta_layer.spec;
            const int key_width = spec.key_heads * spec.key_head_dim;
            const int value_width = spec.value_heads * spec.value_head_dim;
            const int qkv_width = 2 * key_width + value_width;
            if (spec.factorized_projections) {
                gated_delta_layer.q = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetQuery, i),
                    {key_width, resources_.program_.hidden});
                gated_delta_layer.k = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetKey, i),
                    {key_width, resources_.program_.hidden});
                gated_delta_layer.v = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetValue, i),
                    {value_width, resources_.program_.hidden});
                gated_delta_layer.z = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetOutputGate, i),
                    {value_width, resources_.program_.hidden});
            } else {
                gated_delta_layer.qkv = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetQkv, i),
                    {qkv_width, resources_.program_.hidden});
                gated_delta_layer.z = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetZ, i),
                    {value_width, resources_.program_.hidden});
            }
            gated_delta_layer.b = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetBeta, i),
                {spec.value_heads, resources_.program_.hidden});
            gated_delta_layer.a = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  spec.factorized_projections
                                      ? TensorRole::GatedDeltaNetDecay
                                      : TensorRole::GatedDeltaNetAlpha, i),
                {spec.decay_width(), resources_.program_.hidden});
            if (spec.factorized_projections) {
                const auto* q_conv = resources_.weight_loader_->load_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetQueryConv, i),
                    {key_width, 1, spec.conv_kernel});
                const auto* k_conv = resources_.weight_loader_->load_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetKeyConv, i),
                    {key_width, 1, spec.conv_kernel});
                const auto* v_conv = resources_.weight_loader_->load_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetValueConv, i),
                    {value_width, 1, spec.conv_kernel});
                gated_delta_layer.factorized_conv_weight.reset(
                    static_cast<size_t>(qkv_width) * spec.conv_kernel);
                const size_t q_bytes = static_cast<size_t>(key_width) * spec.conv_kernel *
                    sizeof(__nv_bfloat16);
                const size_t v_bytes = static_cast<size_t>(value_width) * spec.conv_kernel *
                    sizeof(__nv_bfloat16);
                CELEG_CUDA(cudaMemcpy(gated_delta_layer.factorized_conv_weight.data(),
                    q_conv, q_bytes, cudaMemcpyDeviceToDevice));
                CELEG_CUDA(cudaMemcpy(gated_delta_layer.factorized_conv_weight.data() +
                    key_width * spec.conv_kernel, k_conv, q_bytes,
                    cudaMemcpyDeviceToDevice));
                CELEG_CUDA(cudaMemcpy(gated_delta_layer.factorized_conv_weight.data() +
                    2 * key_width * spec.conv_kernel, v_conv, v_bytes,
                    cudaMemcpyDeviceToDevice));
                gated_delta_layer.conv_weight = gated_delta_layer.factorized_conv_weight.data();
            } else {
                gated_delta_layer.conv_weight = resources_.weight_loader_->load_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetConv, i),
                    {qkv_width, 1, spec.conv_kernel});
            }
            gated_delta_layer.dt_bias = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetDtBias, i),
                {spec.decay_width()});
            gated_delta_layer.a_log = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetALog, i),
                {spec.value_heads});
            gated_delta_layer.norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetNorm, i),
                {spec.value_head_dim});
            gated_delta_layer.out = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetOutput, i),
                {resources_.program_.hidden, value_width});
            const int conv_dim = qkv_width;
            gated_delta_layer.conv_state.reset(static_cast<size_t>(conv_dim) * spec.conv_kernel);
            gated_delta_layer.recurrent_state.reset(static_cast<size_t>(spec.value_heads) *
                spec.key_head_dim * spec.value_head_dim);
            resources_.layers_.emplace_back(std::move(gated_delta_layer));
        } else if (const auto* mamba =
                       std::get_if<Mamba2Spec>(&semantic_layer.mixer)) {
            Mamba2Layer mamba_layer;
            mamba_layer.common = common_layer;
            mamba_layer.spec = *mamba;
            const Mamba2Spec& spec = mamba_layer.spec;
            const int conv_dim = spec.intermediate_size +
                2 * spec.group_count * spec.state_size;
            mamba_layer.in = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Input, i),
                {2 * spec.intermediate_size + 2 * spec.group_count * spec.state_size +
                 spec.num_heads, resources_.program_.hidden});
            mamba_layer.conv_weight = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Conv, i),
                {conv_dim, 1, spec.conv_kernel});
            mamba_layer.conv_bias = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2ConvBias, i),
                {conv_dim});
            mamba_layer.dt_bias = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2DtBias, i),
                {spec.num_heads});
            mamba_layer.a_log = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2ALog, i),
                {spec.num_heads});
            mamba_layer.d = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2D, i),
                {spec.num_heads});
            mamba_layer.norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Norm, i),
                {spec.intermediate_size});
            mamba_layer.out = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Output, i),
                {resources_.program_.hidden, spec.intermediate_size});
            mamba_layer.conv_state.reset(static_cast<size_t>(conv_dim) * spec.conv_kernel);
            mamba_layer.ssm_state.reset(static_cast<size_t>(spec.intermediate_size) * spec.state_size);
            resources_.layers_.emplace_back(std::move(mamba_layer));
        } else if (const auto* mlp =
                       std::get_if<MlpBlockSpec>(&semantic_layer.mixer)) {
            MlpOnlyLayer mlp_layer;
            mlp_layer.common = common_layer;
            mlp_layer.spec = *mlp;
            mlp_layer.up = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnUp, i),
                {mlp_layer.spec.intermediate_size, resources_.program_.hidden});
            mlp_layer.down = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnDown, i),
                {resources_.program_.hidden, mlp_layer.spec.intermediate_size});
            resources_.layers_.emplace_back(std::move(mlp_layer));
        } else {
            ConvolutionLayer convolution_layer;
            convolution_layer.common = common_layer;
            convolution_layer.spec = std::get<ShortConvolutionSpec>(resources_.program_.layers[i].mixer);
            convolution_layer.conv_in = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::ShortConvInput, i),
                {3 * resources_.program_.hidden, resources_.program_.hidden});
            convolution_layer.conv_weight = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::ShortConvKernel, i),
                {resources_.program_.hidden, 1, convolution_layer.spec.cache_length});
            convolution_layer.conv_out = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::ShortConvOutput, i),
                {resources_.program_.hidden, resources_.program_.hidden});
            convolution_layer.conv_state.reset(
                static_cast<size_t>(convolution_layer.spec.cache_length) * resources_.program_.hidden);
            resources_.layers_.emplace_back(std::move(convolution_layer));
        }
    }
    load_mtp_weights(*this, repo);
        });
}

}