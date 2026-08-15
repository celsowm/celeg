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
        CheckpointDimensions checkpoint_dimensions;
        ModelGraph graph;
        NumericalPolicy numerical_policy;
        const int intermediate = integer_value(
            metadata, descriptor_.dimensions.at("intermediate"));
        const int physical_layer_count = integer_value(metadata, descriptor_.dimensions.at("layers"));
        const int repeat_count = descriptor_.repeated_layers
            ? integer_value(metadata, descriptor_.repeat_count) : 1;
        if (physical_layer_count <= 0 || repeat_count <= 0) {
            throw std::invalid_argument("descriptor has invalid layer schedule");
        }
        const int layer_count = physical_layer_count * repeat_count;
        if (const auto mtp = descriptor_.dimensions.find("mtp_layers");
            mtp != descriptor_.dimensions.end()) {
            checkpoint_dimensions.mtp_num_hidden_layers = integer_value(metadata, mtp->second);
        }
        checkpoint_dimensions.vocab_size = integer_value(metadata, descriptor_.dimensions.at("vocab"));
        if (checkpoint_dimensions.vocab_size == 0 && metadata.is_gguf() && metadata.contains("tokenizer.ggml.tokens")) {
            checkpoint_dimensions.vocab_size = static_cast<int>(metadata.strings("tokenizer.ggml.tokens").size());
        }
        checkpoint_dimensions.max_position_embeddings = integer_value(metadata, descriptor_.dimensions.at("context"));
        const int conv_cache = descriptor_.convolution_cache.has_value()
            ? integer_value(metadata, *descriptor_.convolution_cache) : 0;
        const int conv_dim = descriptor_.convolution_channels.has_value()
            ? integer_value(metadata, *descriptor_.convolution_channels, hidden) : 0;
        const int kv_heads = integer_value(metadata, descriptor_.dimensions.at("kv_heads"));
        const int head_dim = integer_value(metadata, descriptor_.dimensions.at("head_dim"), hidden, query_heads);
        checkpoint_dimensions.token_policy.bos_token_id = token_value(metadata, descriptor_.bos, descriptor_.gguf_bos);
        checkpoint_dimensions.token_policy.eos_token_ids = eos_values(metadata, descriptor_);
        checkpoint_dimensions.token_policy.pad_token_id = token_value(metadata, descriptor_.pad, descriptor_.gguf_pad);
        const auto& numbers = descriptor_.numbers;
        numerical_policy.norm_eps = static_cast<float>(number_value(metadata, numbers.at("norm_eps")));
        numerical_policy.post_norm_eps = numbers.contains("post_norm_eps")
            ? static_cast<float>(number_value(metadata, numbers.at("post_norm_eps")))
            : numerical_policy.norm_eps;
        const double rope_theta = number_value(metadata, numbers.at("rope_theta"));
        const std::vector<float> scheduled_rope_theta = scaling_factor_values(
            metadata, descriptor_.rope_theta_schedule);
        if (!scheduled_rope_theta.empty() && scheduled_rope_theta.size() !=
            static_cast<size_t>(layer_count)) {
            throw std::invalid_argument("descriptor RoPE schedule length does not match layer schedule");
        }
        const std::string scaling_kind = scaling_kind_value(metadata, descriptor_);
        RopeScalingSpec scaling = parse_scaling_kind(scaling_kind);
        std::visit([&](auto& value) {
            using Scaling = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Scaling, NoRopeScaling>) {
                return;
            } else if constexpr (std::is_same_v<Scaling, LinearRopeScaling>) {
                value.factor = scaling_number_value(metadata, descriptor_.rope_scaling_factor, 1.0);
            } else if constexpr (std::is_same_v<Scaling, DynamicNtkRopeScaling>) {
                value.factor = scaling_number_value(metadata, descriptor_.rope_scaling_factor, 1.0);
                value.original_context = scaling_integer_value(
                    metadata, descriptor_.rope_scaling_original_context);
            } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
                value.factor = scaling_number_value(metadata, descriptor_.rope_scaling_factor, 1.0);
                value.attention_factor = scaling_number_value(
                    metadata, descriptor_.rope_scaling_attention_factor, 1.0);
                value.beta_fast = scaling_number_value(metadata, descriptor_.rope_scaling_beta_fast, 0.0);
                value.beta_slow = scaling_number_value(metadata, descriptor_.rope_scaling_beta_slow, 0.0);
            } else if constexpr (std::is_same_v<Scaling, LongRopeScaling>) {
                value.original_context = scaling_integer_value(
                    metadata, descriptor_.rope_scaling_original_context);
                value.short_factors = scaling_factor_values(
                    metadata, descriptor_.rope_scaling_short_factors);
                value.long_factors = scaling_factor_values(
                    metadata, descriptor_.rope_scaling_long_factors);
            } else if constexpr (std::is_same_v<Scaling, Llama3FrequencyScaling>) {
                value.factor = scaling_number_value(metadata, descriptor_.rope_scaling_factor, 1.0);
                value.original_context = scaling_integer_value(
                    metadata, descriptor_.rope_scaling_original_context);
                value.low_frequency_factor = scaling_number_value(
                    metadata, descriptor_.rope_scaling_low_frequency_factor, 1.0);
                value.high_frequency_factor = scaling_number_value(
                    metadata, descriptor_.rope_scaling_high_frequency_factor, 1.0);
            } else {
                static_assert(always_false_v<Scaling>, "unhandled RoPE scaling variant");
            }
        }, scaling);
        const double rotary_fraction = descriptor_.rotary_fraction.has_value()
            ? number_value(metadata, *descriptor_.rotary_fraction) : 1.0;
        numerical_policy.embedding_multiplier = static_cast<float>(
            number_value(metadata, numbers.at("embedding_multiplier"), hidden));
        numerical_policy.logits_multiplier = numbers.contains("logits_multiplier")
            ? static_cast<float>(number_value(metadata, numbers.at("logits_multiplier"))) : 1.0f;
        numerical_policy.attention_multiplier = static_cast<float>(number_value(metadata, numbers.at("attention_multiplier")));
        numerical_policy.residual_multiplier = static_cast<float>(number_value(metadata, numbers.at("residual_multiplier")));
        numerical_policy.logits_divisor = static_cast<float>(number_value(metadata, numbers.at("logits_divisor")));
        graph.hidden = hidden;
        graph.embedding_transform.multiplier = numerical_policy.embedding_multiplier;
        if (descriptor_.embedding_post_norm_kind) {
            graph.embedding_transform.post_norm = NormSpec{
                numerical_policy.norm_eps, *descriptor_.embedding_post_norm_kind};
        }
        graph.logits_divisor = numerical_policy.logits_divisor;
        graph.logits_multiplier = numerical_policy.logits_multiplier;
        graph.final_norm = {numerical_policy.norm_eps, descriptor_.final_norm_kind};
        graph.final_logit_softcap = descriptor_.final_logit_softcap.has_value()
            ? static_cast<float>(number_value(metadata, *descriptor_.final_logit_softcap)) : 0.0f;
        if (descriptor_.norm_after_physical_block) {
            graph.norm_after_layers = {physical_layer_count - 1};
        }
        graph.layers.resize(static_cast<size_t>(layer_count));
        for (int layer = 0; layer < layer_count; ++layer) {
            LayerSpec& semantic_layer = graph.layers[static_cast<size_t>(layer)];
            semantic_layer.mixer = AttentionSpec{};
            semantic_layer.feed_forward = DenseFeedForwardSpec{
                intermediate, descriptor_.feed_forward_activation};
            semantic_layer.operator_norm = {numerical_policy.norm_eps,
                                             descriptor_.operator_norm_kind};
            semantic_layer.feed_forward_norm = NormSpec{numerical_policy.norm_eps,
                                                 descriptor_.feed_forward_norm_kind};
            semantic_layer.residual.multiplier = numerical_policy.residual_multiplier;
        }
        const std::vector<std::string> scheduled_mixer = mixer_schedule_values(metadata, descriptor_);
        if (!scheduled_mixer.empty() && scheduled_mixer.size() !=
            static_cast<size_t>(layer_count)) {
            throw std::invalid_argument("descriptor mixer schedule length does not match layer schedule");
        }
        const std::vector<int> scheduled_kv_heads = field_integer_values(
            metadata, descriptor_.kv_heads_schedule);
        if (!scheduled_kv_heads.empty() && scheduled_kv_heads.size() !=
            static_cast<size_t>(layer_count)) {
            throw std::invalid_argument("descriptor KV-head schedule length does not match layer schedule");
        }
        for (int layer = 0; layer < layer_count; ++layer) {
            if (!scheduled_mixer.empty()) {
                const std::string& kind = scheduled_mixer[static_cast<size_t>(layer)];
                if (kind == descriptor_.convolution_value || kind == "conv" ||
                    kind == "short_convolution") {
                    graph.layers[static_cast<size_t>(layer)].mixer =
                        ShortConvolutionSpec{conv_cache, conv_dim, false};
                } else if (kind == "linear_attention" || kind == "gated_delta_net") {
                    graph.layers[static_cast<size_t>(layer)].mixer = GatedDeltaNetSpec{};
                } else if (kind == "mamba" || kind == "mamba2" || kind == "M") {
                    graph.layers[static_cast<size_t>(layer)].mixer = Mamba2Spec{};
                } else if (kind == "mlp_only" || kind == "MlpOnly" || kind == "-") {
                    graph.layers[static_cast<size_t>(layer)].mixer = MlpBlockSpec{
                        intermediate, descriptor_.feed_forward_activation};
                    graph.layers[static_cast<size_t>(layer)].feed_forward = std::monostate{};
                } else if (kind != "full_attention" && kind != "attention" && kind != "*") {
                    throw std::invalid_argument("descriptor has unsupported mixer kind: " + kind);
                }
            }
        }
        const int dense_layers = descriptor_.moe_dense_layers.has_value()
            ? integer_value(metadata, *descriptor_.moe_dense_layers) : layer_count;
        const int moe_experts = descriptor_.moe_experts.has_value()
            ? integer_value(metadata, *descriptor_.moe_experts) : 0;
        const bool has_moe = moe_experts > 0 && dense_layers < layer_count;
        int moe_intermediate = 0;
        int experts_per_token = 0;
        int shared_expert_intermediate = 0;
        bool normalize_topk = false;
        bool use_expert_bias = false;
        float routed_scaling_factor = 1.0f;
        int routing_group_count = 0;
        int routing_experts_per_group = 0;
        int routing_groups_per_token = 0;
        int routing_group_score_top_k = 0;
        bool router_softmax = false;
        if (has_moe) {
            moe_intermediate = integer_value(metadata, *descriptor_.moe_intermediate);
            experts_per_token = integer_value(metadata, *descriptor_.moe_experts_per_token);
            shared_expert_intermediate = descriptor_.moe_shared_intermediate.has_value()
                ? integer_value(metadata, *descriptor_.moe_shared_intermediate) : 0;
            normalize_topk = boolean_value(metadata, descriptor_.moe_normalize_topk, true);
            router_softmax = normalize_topk;
            use_expert_bias = boolean_value(metadata, descriptor_.moe_expert_bias, false);
            routed_scaling_factor = static_cast<float>(
                descriptor_.moe_routed_scaling.has_value()
                    ? number_value(metadata, *descriptor_.moe_routed_scaling) : 1.0);
            routing_group_count = descriptor_.moe_routing_group_count.has_value()
                ? integer_value(metadata, *descriptor_.moe_routing_group_count) : 0;
            routing_experts_per_group =
                descriptor_.moe_routing_experts_per_group.has_value()
                    ? integer_value(metadata, *descriptor_.moe_routing_experts_per_group) : 0;
            routing_groups_per_token =
                descriptor_.moe_routing_groups_per_token.has_value()
                    ? integer_value(metadata, *descriptor_.moe_routing_groups_per_token) : 0;
            routing_group_score_top_k =
                descriptor_.moe_routing_group_score_top_k.has_value()
                    ? integer_value(metadata, *descriptor_.moe_routing_group_score_top_k) : 0;
            for (int layer = dense_layers; layer < layer_count; ++layer) {
                MoeSelectionSpec selection = MoeTopKSelectionSpec{};
                if (routing_group_count > 0) {
                    selection = MoeGroupedTopKSelectionSpec{
                        routing_group_count,
                        routing_experts_per_group,
                        routing_groups_per_token,
                        routing_group_score_top_k};
                }
                std::optional<SharedExpertSpec> shared;
                if (shared_expert_intermediate > 0) {
                    shared = SharedExpertSpec{
                        shared_expert_intermediate,
                        MoeCombineOrder::RoutedThenShared};
                }
                graph.layers[static_cast<size_t>(layer)].feed_forward = MixtureOfExpertsSpec{
                    .intermediate_size = moe_intermediate,
                    .num_experts = moe_experts,
                    .experts_per_token = experts_per_token,
                    .normalize_topk = normalize_topk,
                    .use_expert_bias = use_expert_bias,
                    .routed_scaling_factor = routed_scaling_factor,
                    .selection = std::move(selection),
                    .shared = std::move(shared),
                    .router_softmax = router_softmax};
            }
        }
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
        for (int layer = 0; layer < layer_count; ++layer) {
            auto& mixer = graph.layers[static_cast<size_t>(layer)].mixer;
            if (std::holds_alternative<GatedDeltaNetSpec>(mixer)) {
                graph.layers[static_cast<size_t>(layer)].mixer =
                    GatedDeltaNetSpec{gated_conv_kernel, gated_key_dim, gated_value_dim,
                                      gated_key_heads, gated_value_heads};
            } else if (std::holds_alternative<Mamba2Spec>(mixer)) {
                graph.layers[static_cast<size_t>(layer)].mixer =
                    Mamba2Spec{gated_conv_kernel, mamba_intermediate, mamba_state_size,
                               mamba_time_step_rank, mamba_heads, mamba_head_dim,
                               mamba_groups, mamba_chunk_size, false, false};
            }
        }
        if (descriptor_.map_physical_layers) {
            checkpoint_dimensions.checkpoint_layer_for_layer.resize(static_cast<size_t>(layer_count));
            for (int layer = 0; layer < layer_count; ++layer) {
                checkpoint_dimensions.checkpoint_layer_for_layer[static_cast<size_t>(layer)] =
                    layer % physical_layer_count;
            }
        }
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
                scheduled_sliding.size() == 1 && layer_count > 1) {
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
            scheduled_sliding.size() != static_cast<size_t>(layer_count)) {
            throw std::invalid_argument("descriptor attention pattern length does not match layer schedule");
        }
        const int sliding_window = descriptor_.sliding_window.has_value()
            ? integer_value(metadata, *descriptor_.sliding_window) : 0;
        const bool has_attention_variants = descriptor_.full_attention_variant.has_value() ||
            descriptor_.sliding_attention_variant.has_value();
        const int shared_layers = descriptor_.shared_kv_suffix_layers.has_value()
            ? integer_value(metadata, *descriptor_.shared_kv_suffix_layers) : 0;
        if (shared_layers < 0 || shared_layers > layer_count) {
            throw std::invalid_argument("descriptor shared KV suffix is out of range");
        }
        const int shared_start = layer_count - shared_layers;
        if (descriptor_.double_wide_shared_suffix) {
            for (int layer = shared_start; layer < layer_count; ++layer) {
                if (auto* dense = std::get_if<DenseFeedForwardSpec>(
                        &graph.layers[static_cast<size_t>(layer)].feed_forward)) {
                    dense->intermediate_size *= 2;
                }
            }
        }
        const auto scheduled_is_sliding = [&](int layer) {
            if (scheduled_sliding.empty()) return false;
            const size_t index = scheduled_sliding.size() == 1 ? 0 :
                scheduled_sliding.size() == static_cast<size_t>(physical_layer_count)
                    ? static_cast<size_t>(layer % physical_layer_count)
                    : static_cast<size_t>(layer);
            return scheduled_sliding[index];
        };
        for (int layer = 0; layer < layer_count; ++layer) {
            if (!std::holds_alternative<AttentionSpec>(
                    graph.layers[static_cast<size_t>(layer)].mixer)) {
                continue;
            }
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
            attention.query_norm = descriptor_.query_norm_enabled
                ? std::optional<NormSpec>{NormSpec{numerical_policy.norm_eps,
                                                   descriptor_.query_norm_kind}}
                : std::nullopt;
            attention.key_norm = descriptor_.key_norm_enabled
                ? std::optional<NormSpec>{NormSpec{numerical_policy.norm_eps,
                                                   descriptor_.key_norm_kind}}
                : std::nullopt;
            attention.pattern = FullCausalPattern{};
            attention.query_scale = numerical_policy.attention_multiplier;
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
                const int slot = integer_value(metadata, *descriptor_.attention_memory_slot);
                if (slot < 0) {
                    throw std::invalid_argument(
                        "external attention descriptor has an invalid memory slot");
                }
                attention.key_value_source = ExternalMemorySource{slot};
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
                attention.kv_sharing = publishes
                    ? KvSharingSpec{SharedKvPublisher{group}}
                    : KvSharingSpec{SharedKvConsumer{group}};
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
            graph.layers[static_cast<size_t>(layer)].mixer = std::move(attention);
        }
        const bool has_per_layer_input = descriptor_.per_layer_input_size.has_value();
        const int per_layer_input_size = has_per_layer_input
            ? integer_value(metadata, *descriptor_.per_layer_input_size) : 0;
        if (has_per_layer_input && per_layer_input_size <= 0) {
            throw std::invalid_argument("descriptor per-layer input size is invalid");
        }
        ArchitectureResolutionStages stages;
        stages.checkpoint_dimensions = [checkpoint_dimensions](const CheckpointView&) {
            return checkpoint_dimensions;
        };
        stages.numerical_policy = [numerical_policy](const CheckpointView&) {
            return numerical_policy;
        };
        stages.graph = [this, graph](const CheckpointDimensions&,
                                     const NumericalPolicy& numerical_policy,
                                     const CheckpointView& view) {
            return finalize_descriptor_graph(graph, descriptor_, numerical_policy,
                                             view.metadata);
        };
        stages.weights = [this](ResolvedModel& model, const CheckpointView&) {
            auto policy = descriptor_detail::create_naming_policy(descriptor_);
            build_weight_plan_from_graph(model, *policy);
        };
        const bool tied_embeddings = descriptor_.tied_embeddings_field.has_value()
            ? boolean_value(metadata, descriptor_.tied_embeddings_field, descriptor_.tied_embeddings)
            : descriptor_.tied_embeddings;
        stages.capabilities = {true, true, false, tied_embeddings};
        stages.provenance.architecture_id = descriptor_.id;
        stages.provenance.source_format = metadata.is_gguf() ? "gguf" : "safetensors";
        stages.provenance.checkpoint_profile_id = descriptor_.id;
        stages.provenance.profile = {descriptor_.id, "", {}};
        ResolvedModel result = resolve_architecture_stages(checkpoint, std::move(stages));
        result.provenance.identity = descriptor_.id + "-" + result.graph.fingerprint();
        return result;
    }

private:
    Descriptor descriptor_;
};

std::unique_ptr<IArchitecture> make_descriptor_architecture(Descriptor descriptor) {
    return std::make_unique<DescriptorArchitecture>(std::move(descriptor));
}

}
