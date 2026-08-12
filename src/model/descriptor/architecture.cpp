#include "detail.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace celeg::descriptor_detail {

class DescriptorArchitecture final : public IArchitecture {
public:
    explicit DescriptorArchitecture(Descriptor descriptor)
        : descriptor_(std::move(descriptor)) {}

    std::string_view id() const override { return descriptor_.id; }

    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        for (const auto& alternative : descriptor_.probe_alternatives) {
            if (std::all_of(alternative.begin(), alternative.end(), [&](const ProbeCondition& condition) {
                    return probe_condition(metadata, condition);
                })) {
                return {true, descriptor_.specificity, "declarative model descriptor"};
            }
        }
        return {false, 0, "descriptor probe did not match"};
    }

    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        if (!probe(checkpoint.metadata).supported) {
            throw std::runtime_error("descriptor cannot resolve checkpoint: " + descriptor_.id);
        }
        const CheckpointMetadata& metadata = checkpoint.metadata;
        const int hidden = integer_value(metadata, descriptor_.dimensions.at("hidden"));
        const int query_heads = integer_value(metadata, descriptor_.dimensions.at("query_heads"));
        RuntimeTopology topology;
        topology.hidden = hidden;
        topology.intermediate = integer_value(metadata, descriptor_.dimensions.at("intermediate"));
        topology.dense_intermediate = topology.intermediate;
        topology.max_feed_forward_intermediate = topology.intermediate;
        const int physical_layer_count = integer_value(metadata, descriptor_.dimensions.at("layers"));
        const int repeat_count = descriptor_.repeated_layers
            ? integer_value(metadata, descriptor_.repeat_count) : 1;
        if (physical_layer_count <= 0 || repeat_count <= 0) {
            throw std::invalid_argument("descriptor has invalid layer schedule");
        }
        topology.num_hidden_layers = physical_layer_count * repeat_count;
        if (const auto mtp = descriptor_.dimensions.find("mtp_layers");
            mtp != descriptor_.dimensions.end()) {
            topology.mtp_num_hidden_layers = integer_value(metadata, mtp->second);
        }
        topology.vocab_size = integer_value(metadata, descriptor_.dimensions.at("vocab"));
        if (topology.vocab_size == 0 && metadata.is_gguf() && metadata.contains("tokenizer.ggml.tokens")) {
            topology.vocab_size = static_cast<int>(metadata.strings("tokenizer.ggml.tokens").size());
        }
        topology.max_position_embeddings = integer_value(metadata, descriptor_.dimensions.at("context"));
        topology.conv_cache = descriptor_.convolution_cache.has_value()
            ? integer_value(metadata, *descriptor_.convolution_cache) : 0;
        topology.conv_dim = descriptor_.convolution_channels.has_value()
            ? integer_value(metadata, *descriptor_.convolution_channels, hidden) : 0;
        const int kv_heads = integer_value(metadata, descriptor_.dimensions.at("kv_heads"));
        const int head_dim = integer_value(metadata, descriptor_.dimensions.at("head_dim"), hidden, query_heads);
        topology.token_policy.bos_token_id = token_value(metadata, descriptor_.bos, descriptor_.gguf_bos);
        topology.token_policy.eos_token_ids = eos_values(metadata, descriptor_);
        topology.token_policy.pad_token_id = token_value(metadata, descriptor_.pad, descriptor_.gguf_pad);
        const auto& numbers = descriptor_.numbers;
        topology.numerical_policy.norm_eps = static_cast<float>(number_value(metadata, numbers.at("norm_eps")));
        topology.numerical_policy.post_norm_eps = numbers.contains("post_norm_eps")
            ? static_cast<float>(number_value(metadata, numbers.at("post_norm_eps")))
            : topology.numerical_policy.norm_eps;
        const double rope_theta = number_value(metadata, numbers.at("rope_theta"));
        const std::vector<float> scheduled_rope_theta = scaling_factor_values(
            metadata, descriptor_.rope_theta_schedule);
        if (!scheduled_rope_theta.empty() && scheduled_rope_theta.size() !=
            static_cast<size_t>(topology.num_hidden_layers)) {
            throw std::invalid_argument("descriptor RoPE schedule length does not match layer schedule");
        }
        RopeScalingSpec scaling;
        scaling.kind = parse_scaling_kind(scaling_kind_value(metadata, descriptor_));
        scaling.factor = scaling_number_value(metadata, descriptor_.rope_scaling_factor, 1.0);
        scaling.original_context = scaling_integer_value(
            metadata, descriptor_.rope_scaling_original_context);
        scaling.attention_factor = scaling_number_value(
            metadata, descriptor_.rope_scaling_attention_factor, 1.0);
        scaling.beta_fast = scaling_number_value(metadata, descriptor_.rope_scaling_beta_fast, 0.0);
        scaling.beta_slow = scaling_number_value(metadata, descriptor_.rope_scaling_beta_slow, 0.0);
        scaling.low_frequency_factor = scaling_number_value(
            metadata, descriptor_.rope_scaling_low_frequency_factor, 1.0);
        scaling.high_frequency_factor = scaling_number_value(
            metadata, descriptor_.rope_scaling_high_frequency_factor, 1.0);
        scaling.short_factors = scaling_factor_values(
            metadata, descriptor_.rope_scaling_short_factors);
        scaling.long_factors = scaling_factor_values(
            metadata, descriptor_.rope_scaling_long_factors);
        const double rotary_fraction = descriptor_.rotary_fraction.has_value()
            ? number_value(metadata, *descriptor_.rotary_fraction) : 1.0;
        topology.numerical_policy.embedding_multiplier = static_cast<float>(
            number_value(metadata, numbers.at("embedding_multiplier"), hidden));
        topology.numerical_policy.logits_multiplier = numbers.contains("logits_multiplier")
            ? static_cast<float>(number_value(metadata, numbers.at("logits_multiplier"))) : 1.0f;
        topology.numerical_policy.attention_multiplier = static_cast<float>(number_value(metadata, numbers.at("attention_multiplier")));
        topology.numerical_policy.residual_multiplier = static_cast<float>(number_value(metadata, numbers.at("residual_multiplier")));
        topology.numerical_policy.logits_divisor = static_cast<float>(number_value(metadata, numbers.at("logits_divisor")));
        const std::vector<std::string> scheduled_mixer = mixer_schedule_values(metadata, descriptor_);
        if (!scheduled_mixer.empty() && scheduled_mixer.size() !=
            static_cast<size_t>(topology.num_hidden_layers)) {
            throw std::invalid_argument("descriptor mixer schedule length does not match layer schedule");
        }
        const std::vector<int> scheduled_kv_heads = field_integer_values(
            metadata, descriptor_.kv_heads_schedule);
        if (!scheduled_kv_heads.empty() && scheduled_kv_heads.size() !=
            static_cast<size_t>(topology.num_hidden_layers)) {
            throw std::invalid_argument("descriptor KV-head schedule length does not match layer schedule");
        }
        topology.mixer_kinds.assign(static_cast<size_t>(topology.num_hidden_layers),
                                    MixerKind::Attention);
        for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
            if (!scheduled_mixer.empty()) {
                const std::string& kind = scheduled_mixer[static_cast<size_t>(layer)];
                if (kind == descriptor_.convolution_value || kind == "conv" ||
                    kind == "short_convolution") {
                    topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::ShortConvolution;
                } else if (kind == "linear_attention" || kind == "gated_delta_net") {
                    topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::GatedDeltaNet;
                } else if (kind == "mamba" || kind == "mamba2" || kind == "M") {
                    topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::Mamba2;
                } else if (kind == "mlp_only" || kind == "MlpOnly" || kind == "-") {
                    topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::MlpOnly;
                } else if (kind != "full_attention" && kind != "attention" && kind != "*") {
                    throw std::invalid_argument("descriptor has unsupported mixer kind: " + kind);
                }
            }
        }
        topology.feed_forward_kinds.assign(static_cast<size_t>(topology.num_hidden_layers), FeedForwardKind::Dense);
        topology.feed_forward_intermediates.assign(static_cast<size_t>(topology.num_hidden_layers), topology.intermediate);
        topology.feed_forward_activations.assign(static_cast<size_t>(topology.num_hidden_layers),
                                                 descriptor_.feed_forward_activation);
        const int dense_layers = descriptor_.moe_dense_layers.has_value()
            ? integer_value(metadata, *descriptor_.moe_dense_layers) : topology.num_hidden_layers;
        const int moe_experts = descriptor_.moe_experts.has_value()
            ? integer_value(metadata, *descriptor_.moe_experts) : 0;
        const bool has_moe = moe_experts > 0 && dense_layers < topology.num_hidden_layers;
        if (has_moe) {
            topology.num_dense_layers = dense_layers;
            topology.num_experts = moe_experts;
            topology.moe_intermediate = integer_value(metadata, *descriptor_.moe_intermediate);
            topology.experts_per_token = integer_value(metadata, *descriptor_.moe_experts_per_token);
            topology.shared_expert_intermediate = descriptor_.moe_shared_intermediate.has_value()
                ? integer_value(metadata, *descriptor_.moe_shared_intermediate) : 0;
            topology.normalize_topk = boolean_value(metadata, descriptor_.moe_normalize_topk, true);
            topology.moe_router_softmax = topology.normalize_topk;
            topology.use_expert_bias = boolean_value(metadata, descriptor_.moe_expert_bias, false);
            topology.routed_scaling_factor = static_cast<float>(
                descriptor_.moe_routed_scaling.has_value()
                    ? number_value(metadata, *descriptor_.moe_routed_scaling) : 1.0);
            for (int layer = dense_layers; layer < topology.num_hidden_layers; ++layer) {
                topology.feed_forward_kinds[static_cast<size_t>(layer)] =
                    FeedForwardKind::MixtureOfExperts;
            }
        }
        topology.gated_delta_net_layouts.resize(static_cast<size_t>(topology.num_hidden_layers));
        topology.mamba2_layouts.resize(static_cast<size_t>(topology.num_hidden_layers));
        topology.mlp_only_layouts.resize(static_cast<size_t>(topology.num_hidden_layers));
        const int gated_key_heads = descriptor_.recurrent_key_heads.has_value()
            ? integer_value(metadata, *descriptor_.recurrent_key_heads) : 0;
        const int gated_value_heads = descriptor_.recurrent_value_heads.has_value()
            ? integer_value(metadata, *descriptor_.recurrent_value_heads) : 0;
        const int gated_key_dim = descriptor_.recurrent_key_dim.has_value()
            ? integer_value(metadata, *descriptor_.recurrent_key_dim) : 0;
        const int gated_value_dim = descriptor_.recurrent_value_dim.has_value()
            ? integer_value(metadata, *descriptor_.recurrent_value_dim) : 0;
        const int gated_conv_kernel = descriptor_.recurrent_conv_kernel.has_value()
            ? integer_value(metadata, *descriptor_.recurrent_conv_kernel) : 0;
        int mamba_intermediate = descriptor_.mamba_intermediate.has_value()
            ? integer_value(metadata, *descriptor_.mamba_intermediate) : 0;
        const int mamba_state_size = descriptor_.mamba_state_size.has_value()
            ? integer_value(metadata, *descriptor_.mamba_state_size) : 0;
        const int mamba_time_step_rank = descriptor_.mamba_time_step_rank.has_value()
            ? integer_value(metadata, *descriptor_.mamba_time_step_rank) : 0;
        const int mamba_heads = descriptor_.mamba_heads.has_value()
            ? integer_value(metadata, *descriptor_.mamba_heads) : 0;
        const int mamba_head_dim = descriptor_.mamba_head_dim.has_value()
            ? integer_value(metadata, *descriptor_.mamba_head_dim) : 0;
        const int mamba_groups = descriptor_.mamba_groups.has_value()
            ? integer_value(metadata, *descriptor_.mamba_groups) : 0;
        const int mamba_chunk_size = descriptor_.mamba_chunk_size.has_value()
            ? integer_value(metadata, *descriptor_.mamba_chunk_size) : 0;
        if (mamba_intermediate == 0 && mamba_heads > 0 && mamba_head_dim > 0) {
            mamba_intermediate = mamba_heads * mamba_head_dim;
        }
        topology.mamba2_intermediate = mamba_intermediate;
        for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
            const MixerKind kind = topology.mixer_kinds[static_cast<size_t>(layer)];
            if (kind == MixerKind::GatedDeltaNet) {
                topology.gated_delta_net_layouts[static_cast<size_t>(layer)] =
                    GatedDeltaNetSpec{gated_conv_kernel, gated_key_dim, gated_value_dim,
                                      gated_key_heads, gated_value_heads};
            } else if (kind == MixerKind::Mamba2) {
                topology.mamba2_layouts[static_cast<size_t>(layer)] =
                    Mamba2Spec{gated_conv_kernel, mamba_intermediate, mamba_state_size,
                               mamba_time_step_rank, mamba_heads, mamba_head_dim,
                               mamba_groups, mamba_chunk_size, false, false};
            } else if (kind == MixerKind::MlpOnly) {
                topology.mlp_only_layouts[static_cast<size_t>(layer)] =
                    MlpBlockSpec{topology.intermediate, descriptor_.feed_forward_activation};
            }
        }
        topology.attention_layer_count = 0;
        topology.conv_layer_count = 0;
        topology.attention_slot_for_layer.assign(static_cast<size_t>(topology.num_hidden_layers), -1);
        topology.layer_for_attention_slot.clear();
        for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
            if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
                topology.attention_slot_for_layer[static_cast<size_t>(layer)] =
                    topology.attention_layer_count++;
                topology.layer_for_attention_slot.push_back(layer);
            } else if (topology.mixer_kinds[static_cast<size_t>(layer)] ==
                       MixerKind::ShortConvolution) {
                ++topology.conv_layer_count;
            } else if (topology.mixer_kinds[static_cast<size_t>(layer)] ==
                       MixerKind::GatedDeltaNet) {
                ++topology.gated_delta_net_layer_count;
            } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Mamba2) {
                ++topology.mamba2_layer_count;
            } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::MlpOnly) {
                ++topology.mlp_only_layer_count;
            }
        }
        if (descriptor_.map_physical_layers) {
            topology.checkpoint_layer_for_layer.resize(static_cast<size_t>(topology.num_hidden_layers));
            for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
                topology.checkpoint_layer_for_layer[static_cast<size_t>(layer)] =
                    layer % physical_layer_count;
            }
        }
        topology.attention_layouts.resize(static_cast<size_t>(topology.num_hidden_layers));
        const std::string disable_key = metadata.is_gguf() ? descriptor_.disable_rope_gguf
                                                            : descriptor_.disable_rope_json;
        const std::vector<int> disabled = integer_values(metadata, disable_key);
        const std::string position_kind = position_kind_value(metadata, descriptor_);
        const std::vector<float> alibi_slopes = scaling_factor_values(
            metadata, descriptor_.alibi_slopes);
        const std::vector<int> mrope_sections = field_integer_values(
            metadata, descriptor_.mrope_sections);
        if (!mrope_sections.empty() && mrope_sections.size() != 3) {
            throw std::invalid_argument("descriptor M-RoPE requires three sections");
        }
        const bool mrope_interleaved = boolean_value(
            metadata, descriptor_.mrope_interleaved, false);
        const int relative_bucket_count = descriptor_.relative_bucket_count.has_value()
            ? integer_value(metadata, *descriptor_.relative_bucket_count) : 0;
        const int relative_max_distance = descriptor_.relative_max_distance.has_value()
            ? integer_value(metadata, *descriptor_.relative_max_distance) : 0;
        const std::vector<bool> scheduled_sliding = attention_pattern_values(metadata, descriptor_);
        if (descriptor_.attention_pattern.has_value()) {
            const std::string pattern_key = selected_key(metadata, *descriptor_.attention_pattern);
            if (!pattern_key.empty() && metadata.contains(pattern_key) &&
                scheduled_sliding.size() == 1 && topology.num_hidden_layers > 1) {
                const MetadataValue& pattern_value = metadata.value(pattern_key);
                if (std::holds_alternative<std::vector<std::string>>(pattern_value) ||
                    std::holds_alternative<std::vector<int64_t>>(pattern_value)) {
                    throw std::invalid_argument(
                        "descriptor attention pattern length does not match layer schedule");
                }
            }
        }
        if (!scheduled_sliding.empty() && scheduled_sliding.size() != 1 &&
            scheduled_sliding.size() != static_cast<size_t>(physical_layer_count) &&
            scheduled_sliding.size() != static_cast<size_t>(topology.num_hidden_layers)) {
            throw std::invalid_argument("descriptor attention pattern length does not match layer schedule");
        }
        const int sliding_window = descriptor_.sliding_window.has_value()
            ? integer_value(metadata, *descriptor_.sliding_window) : 0;
        const bool has_attention_variants = descriptor_.full_attention_variant.has_value() ||
            descriptor_.sliding_attention_variant.has_value();
        const int shared_layers = descriptor_.shared_kv_suffix_layers.has_value()
            ? integer_value(metadata, *descriptor_.shared_kv_suffix_layers) : 0;
        if (shared_layers < 0 || shared_layers > topology.num_hidden_layers) {
            throw std::invalid_argument("descriptor shared KV suffix is out of range");
        }
        const int shared_start = topology.num_hidden_layers - shared_layers;
        const auto scheduled_is_sliding = [&](int layer) {
            if (scheduled_sliding.empty()) return false;
            const size_t index = scheduled_sliding.size() == 1 ? 0 :
                scheduled_sliding.size() == static_cast<size_t>(physical_layer_count)
                    ? static_cast<size_t>(layer % physical_layer_count)
                    : static_cast<size_t>(layer);
            return scheduled_sliding[index];
        };
        for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
            const bool is_sliding = scheduled_is_sliding(layer);
            const AttentionVariant* variant = is_sliding
                ? (descriptor_.sliding_attention_variant.has_value()
                    ? &*descriptor_.sliding_attention_variant : nullptr)
                : (descriptor_.full_attention_variant.has_value()
                    ? &*descriptor_.full_attention_variant : nullptr);
            const int layer_query_heads = variant && variant->query_heads.has_value()
                ? integer_value(metadata, *variant->query_heads) : query_heads;
            const int layer_kv_heads = variant && variant->key_value_heads.has_value()
                ? integer_value(metadata, *variant->key_value_heads) : kv_heads;
            const int layer_head_dim = variant && variant->head_dim.has_value()
                ? integer_value(metadata, *variant->head_dim, hidden, layer_query_heads)
                : head_dim;
            const double layer_rope_theta = !scheduled_rope_theta.empty()
                ? scheduled_rope_theta[static_cast<size_t>(layer)]
                : variant && variant->rope_theta.has_value()
                    ? number_value(metadata, *variant->rope_theta) : rope_theta;
            const double layer_rotary_fraction = variant && variant->rotary_fraction.has_value()
                ? number_value(metadata, *variant->rotary_fraction) : rotary_fraction;
            const int layer_sliding_window = variant && variant->sliding_window.has_value()
                ? integer_value(metadata, *variant->sliding_window) : sliding_window;
            if (is_sliding && layer_sliding_window <= 0) {
                throw std::invalid_argument("descriptor sliding attention has invalid window");
            }
        const int scheduled_layer_kv_heads = scheduled_kv_heads.empty()
            ? layer_kv_heads : scheduled_kv_heads[static_cast<size_t>(layer)];
            AttentionSpec attention;
            attention.query_heads = layer_query_heads;
            attention.key_value_heads = scheduled_layer_kv_heads;
            attention.head_dim = layer_head_dim;
            attention.query_norm = {descriptor_.query_norm_enabled
                ? topology.numerical_policy.norm_eps : 0.0f,
                descriptor_.query_norm_kind};
            attention.key_norm = {descriptor_.key_norm_enabled
                ? topology.numerical_policy.norm_eps : 0.0f,
                descriptor_.key_norm_kind};
            attention.pattern = FullCausalPattern{};
            attention.query_scale = topology.numerical_policy.attention_multiplier;
            attention.output_gate = {descriptor_.attention_gate_kind,
                                     descriptor_.attention_gate_packed_with_query};
            if (descriptor_.attention_state_kind == "latent") {
                if (!descriptor_.latent_rank.has_value()) {
                    throw std::invalid_argument("latent attention descriptor has no latent rank");
                }
                const int latent_rank = integer_value(metadata, *descriptor_.latent_rank);
                const int rope_head_dim = descriptor_.latent_rope_head_dim.has_value()
                    ? integer_value(metadata, *descriptor_.latent_rope_head_dim) : 0;
                const int nope_head_dim = descriptor_.latent_nope_head_dim.has_value()
                    ? integer_value(metadata, *descriptor_.latent_nope_head_dim) : latent_rank;
                attention.state = LatentAttentionStateSpec{
                    latent_rank, rope_head_dim, nope_head_dim,
                    descriptor_.latent_decoupled_rope};
            } else if (descriptor_.attention_state_kind != "ordinary_kv") {
                throw std::invalid_argument("descriptor has unsupported attention state kind: " +
                                            descriptor_.attention_state_kind);
            }
            attention.state_storage.key = parse_state_scalar(descriptor_.state_key_storage);
            attention.state_storage.value = parse_state_scalar(descriptor_.state_value_storage);
            attention.state_storage.latent = parse_state_scalar(descriptor_.state_latent_storage);
            attention.state_storage.rotary = parse_state_scalar(descriptor_.state_rotary_storage);
            attention.state_storage.recurrent = parse_state_scalar(
                descriptor_.state_recurrent_storage);
            attention.state_storage.granularity = parse_state_granularity(
                descriptor_.state_storage_granularity);
            attention.state_storage.paged = descriptor_.state_paged;
            if (descriptor_.attention_key_value_source == "external_memory") {
                if (!descriptor_.attention_memory_slot.has_value()) {
                    throw std::invalid_argument(
                        "external attention descriptor has no memory slot");
                }
                attention.sources.key_value = AttentionSourceKind::ExternalMemory;
                attention.sources.memory_slot = integer_value(
                    metadata, *descriptor_.attention_memory_slot);
                if (attention.sources.memory_slot < 0) {
                    throw std::invalid_argument(
                        "external attention descriptor has an invalid memory slot");
                }
            } else if (descriptor_.attention_key_value_source != "current_sequence") {
                throw std::invalid_argument(
                    "descriptor has unsupported attention key/value source: " +
                    descriptor_.attention_key_value_source);
            }
            if (is_sliding) {
                attention.pattern = SlidingWindowPattern{layer_sliding_window};
            }
            if (shared_layers > 0) {
                const int group = has_attention_variants ? (is_sliding ? 1 : 0) : 0;
                const bool is_shared = layer >= shared_start;
                bool publishes = false;
                if (!is_shared) {
                    publishes = true;
                    for (int later = layer + 1; later < shared_start; ++later) {
                        if ((has_attention_variants && scheduled_is_sliding(later)) == is_sliding) {
                            publishes = false;
                            break;
                        }
                    }
                }
                attention.kv_sharing = KvSharingSpec{group, publishes};
            }
            if (layer_rope_theta == 0.0) {
                attention.position = NoPositionEncodingSpec{};
            } else if (layer < static_cast<int>(disabled.size()) &&
                disabled[static_cast<size_t>(layer)] == 0) {
                attention.position = NoPositionEncodingSpec{};
            } else if (position_kind == "none") {
                attention.position = NoPositionEncodingSpec{};
            } else if (position_kind == "alibi") {
                attention.position = NoPositionEncodingSpec{};
                attention.bias = AlibiBiasSpec{alibi_slopes};
            } else if (position_kind == "relative_bias") {
                attention.position = NoPositionEncodingSpec{};
                attention.bias = RelativePositionBiasSpec{
                    relative_bucket_count, relative_max_distance, descriptor_.relative_bidirectional};
            } else if (position_kind == "rope") {
                RopePositionSpec rope{layer_rope_theta, layer_rotary_fraction, scaling};
                if (!mrope_sections.empty()) {
                    attention.position = MultiAxisRopeSpec{
                        rope, {mrope_sections[0], mrope_sections[1], mrope_sections[2]},
                        mrope_interleaved, 3};
                } else {
                    attention.position = rope;
                }
            } else {
                throw std::invalid_argument("descriptor has unsupported position kind: " + position_kind);
            }
            topology.attention_layouts[static_cast<size_t>(layer)] = std::move(attention);
        }
        topology.shared_kv_group_count = shared_layers > 0
            ? (has_attention_variants ? 2 : 1) : 0;
        topology.has_split_attention_norms = descriptor_.split_attention_norms;
        topology.has_per_layer_input = descriptor_.per_layer_input_size.has_value();
        topology.per_layer_input_size = topology.has_per_layer_input
            ? integer_value(metadata, *descriptor_.per_layer_input_size) : 0;
        if (topology.has_per_layer_input && topology.per_layer_input_size <= 0) {
            throw std::invalid_argument("descriptor per-layer input size is invalid");
        }
        topology.feed_forward_activations.assign(
            static_cast<size_t>(topology.num_hidden_layers), descriptor_.feed_forward_activation);
        topology.validate();
        ArchitectureResolutionStages stages;
        stages.topology = [topology](const CheckpointView&) { return topology; };
        stages.graph = [this](ResolvedModel& model, const CheckpointView& view) {
            build_descriptor_graph(model, descriptor_, view.metadata);
        };
        stages.weights = [this](ResolvedModel& model, const CheckpointView&) {
            auto policy = descriptor_detail::create_naming_policy(descriptor_);
            descriptor_detail::build_weight_plan(model, descriptor_, *policy);
        };
        const bool tied_embeddings = descriptor_.tied_embeddings_field.has_value()
            ? boolean_value(metadata, descriptor_.tied_embeddings_field, descriptor_.tied_embeddings)
            : descriptor_.tied_embeddings;
        stages.capabilities = {true, true, false, tied_embeddings};
        stages.provenance.architecture_id = descriptor_.id;
        stages.provenance.source_format = metadata.is_gguf() ? "gguf" : "safetensors";
        stages.provenance.checkpoint_profile_id = descriptor_.id;
        stages.provenance.chat_template_id = descriptor_.chat_template;
        stages.provenance.profile = {descriptor_.id, "", {}, descriptor_.chat_template};
        ResolvedModel result = resolve_architecture_stages(checkpoint, std::move(stages));
        result.provenance.identity = descriptor_.id + "-" + result.topology.fingerprint();
        return result;
    }

private:
    Descriptor descriptor_;
};

std::unique_ptr<IArchitecture> make_descriptor_architecture(Descriptor descriptor) {
    return std::make_unique<DescriptorArchitecture>(std::move(descriptor));
}

} // namespace celeg::descriptor_detail
