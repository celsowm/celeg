#include "canonical_internal.hpp"
#include "support.hpp"

#include <cmath>

namespace celeg::inference_detail {
namespace {

const TensorInventoryEntry* find_mamba_tensor(
    const InferenceInput& input,
    int layer,
    std::string_view suffix) {
    const auto candidates = mamba2_tensor_candidates(layer, suffix);
    const TensorInventoryEntry* found = nullptr;
    for (const auto& candidate : candidates) {
        if (const auto* tensor = input.inventory.find(candidate)) {
            if (found != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "multiple Mamba-2 tensor spellings are present for layer " +
                        std::to_string(layer));
            }
            found = tensor;
        }
    }
    return found;
}

AttentionSpec make_attention(
    const NormalizedModelMetadata& metadata,
    int query_heads,
    int key_value_heads,
    int head_dim,
    bool query_key_norm) {
    AttentionSpec attention;
    attention.query_heads = query_heads;
    attention.key_value_heads = key_value_heads;
    attention.head_dim = head_dim;
    attention.query_norm = {
        query_key_norm ? *metadata.norm_epsilon : 0.0f,
        query_key_norm ? NormWeightKind::Scale : NormWeightKind::None};
    attention.key_norm = attention.query_norm;
    attention.pattern = FullCausalPattern{};
    attention.query_scale = 1.0f;
    attention.position = RopePositionSpec{
        *metadata.rope_theta,
        *metadata.rotary_fraction,
        RopeScalingSpec{}};
    std::get<RopePositionSpec>(attention.position).pairing =
        *metadata.rope_pairing;
    attention.state_storage = AttentionStateStorageSpec{};
    if (*metadata.xsa_projection) {
        attention.output_transform = OrthogonalizeCurrentValueSpec{
            *metadata.xsa_minimum_norm_squared};
    }
    return attention;
}

void infer_fused_gated_delta(
    CanonicalInferenceContext& context,
    int layer,
    LayerSpec& semantic_layer,
    std::string_view layer_prefix,
    bool has_ffn) {
    const auto& input = context.input;
    const auto& m = input.metadata;

    const auto* qkv =
        input.inventory.find(std::string(layer_prefix) + "attn_qkv.weight");
    const auto* z =
        input.inventory.find(std::string(layer_prefix) + "attn_gate.weight");
    const auto* alpha =
        input.inventory.find(std::string(layer_prefix) + "ssm_alpha.weight");
    const auto* beta =
        input.inventory.find(std::string(layer_prefix) + "ssm_beta.weight");
    const auto* convolution =
        input.inventory.find(std::string(layer_prefix) + "ssm_conv1d.weight");
    const auto* dt_bias =
        input.inventory.find(std::string(layer_prefix) + "ssm_dt.bias");
    const auto* a_log =
        input.inventory.find(std::string(layer_prefix) + "ssm_a");
    const auto* norm =
        input.inventory.find(std::string(layer_prefix) + "ssm_norm.weight");
    const auto* output =
        input.inventory.find(std::string(layer_prefix) + "ssm_out.weight");

    const int key_dim = m.mamba_state_size.value_or(0);
    const int key_heads = m.mamba_group_count.value_or(0);
    const int value_heads = m.mamba_time_step_rank.value_or(0);
    const int value_width =
        output && output->shape.size() == 2
            ? static_cast<int>(output->shape[1])
            : 0;
    const int value_dim =
        value_heads > 0 && value_width % value_heads == 0
            ? value_width / value_heads
            : 0;
    const int qkv_width = 2 * key_heads * key_dim + value_width;

    if (!qkv || !z || !alpha || !beta || !convolution || !dt_bias ||
        !a_log || !norm || !output || key_dim <= 0 || key_heads <= 0 ||
        value_heads <= 0 || value_dim <= 0 ||
        !shape_is(*qkv, {qkv_width, *m.hidden_size}) ||
        !shape_is(*z, {value_width, *m.hidden_size}) ||
        !shape_is(*alpha, {value_heads, *m.hidden_size}) ||
        !shape_is(*beta, {value_heads, *m.hidden_size}) ||
        !shape_is(
            *convolution,
            {qkv_width, 1, m.mamba_conv_kernel.value_or(0)}) ||
        !shape_is(*dt_bias, {value_heads}) ||
        !shape_is(*a_log, {value_heads}) ||
        !shape_is(*norm, {value_dim}) ||
        !shape_is(*output, {*m.hidden_size, value_width})) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "fused recurrent linear-attention tensor shapes do not agree with "
            "geometry for layer " +
                std::to_string(layer));
    }

    semantic_layer.mixer = GatedDeltaNetSpec{
        m.mamba_conv_kernel.value_or(0),
        key_dim,
        value_dim,
        key_heads,
        value_heads,
        false,
        false,
        -5.0f,
        false,
        false};
    if (!has_ffn) semantic_layer.feed_forward = std::monostate{};
}

void infer_factorized_gated_delta(
    CanonicalInferenceContext& context,
    int layer,
    LayerSpec& semantic_layer,
    std::string_view model_layer_prefix,
    bool has_ffn) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    const std::string prefix =
        std::string(model_layer_prefix) + "attention.";

    const auto* q = input.inventory.find(prefix + "q_proj.weight");
    const auto* k = input.inventory.find(prefix + "k_proj.weight");
    const auto* v = input.inventory.find(prefix + "v_proj.weight");
    const auto* f = input.inventory.find(prefix + "f_proj.weight");
    const auto* b = input.inventory.find(prefix + "b_proj.weight");
    const auto* g = input.inventory.find(prefix + "g_proj.weight");
    const auto* qc = input.inventory.find(prefix + "q_conv1d.weight");
    const auto* kc = input.inventory.find(prefix + "k_conv1d.weight");
    const auto* vc = input.inventory.find(prefix + "v_conv1d.weight");
    const auto* dt = input.inventory.find(prefix + "dt_bias");
    const auto* a_log = input.inventory.find(prefix + "A_log");
    const auto* norm = input.inventory.find(prefix + "o_norm.weight");
    const auto* out = input.inventory.find(prefix + "o_proj.weight");

    const int heads =
        m.recurrent_key_heads.value_or(*m.query_heads.value_for(layer));
    const int key_dim =
        m.recurrent_key_dim.value_or(*m.head_dim.value_for(layer));
    const int value_dim =
        m.recurrent_value_dim.value_or(key_dim);
    const int width = heads * key_dim;
    const int conv_kernel = m.recurrent_conv_kernel.value_or(
        qc && qc->shape.size() == 3
            ? static_cast<int>(qc->shape[2])
            : 0);

    if (!q || !k || !v || !f || !b || !g || !qc || !kc || !vc ||
        !dt || !a_log || !norm || !out || heads <= 0 ||
        key_dim <= 0 || value_dim <= 0 ||
        q->shape != std::vector<std::int64_t>{width, *m.hidden_size} ||
        k->shape != std::vector<std::int64_t>{width, *m.hidden_size} ||
        v->shape != std::vector<std::int64_t>{width, *m.hidden_size} ||
        f->shape != std::vector<std::int64_t>{width, *m.hidden_size} ||
        b->shape != std::vector<std::int64_t>{heads, *m.hidden_size} ||
        g->shape != std::vector<std::int64_t>{width, *m.hidden_size} ||
        qc->shape != std::vector<std::int64_t>{width, 1, conv_kernel} ||
        kc->shape != std::vector<std::int64_t>{width, 1, conv_kernel} ||
        vc->shape != std::vector<std::int64_t>{width, 1, conv_kernel} ||
        dt->shape != std::vector<std::int64_t>{width} ||
        a_log->shape != std::vector<std::int64_t>{heads} ||
        norm->shape != std::vector<std::int64_t>{value_dim} ||
        out->shape !=
            std::vector<std::int64_t>{*m.hidden_size, width}) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "recurrent linear-attention tensor shapes do not agree with "
            "geometry for layer " +
                std::to_string(layer));
    }

    semantic_layer.mixer = GatedDeltaNetSpec{
        conv_kernel,
        key_dim,
        value_dim,
        heads,
        heads,
        true,
        m.recurrent_safe_decay.value_or(false),
        m.recurrent_decay_lower_bound.value_or(-5.0f),
        true,
        true};
    if (!has_ffn) semantic_layer.feed_forward = std::monostate{};
}

void infer_latent_attention(
    CanonicalInferenceContext& context,
    int layer,
    LayerSpec& semantic_layer,
    std::string_view model_layer_prefix,
    bool has_ffn) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    const std::string prefix =
        std::string(model_layer_prefix) + "attention.";

    const auto* q_a = input.inventory.find(prefix + "q_a_proj.weight");
    const auto* q_a_norm =
        input.inventory.find(prefix + "q_a_layernorm.weight");
    const auto* q_b = input.inventory.find(prefix + "q_b_proj.weight");
    const auto* kv_a =
        input.inventory.find(prefix + "kv_a_proj_with_mqa.weight");
    const auto* kv_a_norm =
        input.inventory.find(prefix + "kv_a_layernorm.weight");
    const auto* kv_b = input.inventory.find(prefix + "kv_b_proj.weight");
    const auto* out = input.inventory.find(prefix + "dense.weight");
    if (!out) {
        out = input.inventory.find(prefix + "o_proj.weight");
    }
    const auto* gate = input.inventory.find(prefix + "g_proj.weight");

    const int q_rank = m.latent_query_rank.value_or(0);
    const int kv_rank = m.latent_kv_rank.value_or(0);
    const int nope = m.latent_query_nope_dim.value_or(0);
    const int rope = m.latent_query_rope_dim.value_or(0);
    const int value_dim = m.latent_value_head_dim.value_or(0);
    const int heads = *m.query_heads.value_for(layer);

    const std::vector<std::int64_t> head_gate_shape = {
        heads,
        *m.hidden_size};
    const std::vector<std::int64_t> element_gate_shape = {
        heads * value_dim,
        *m.hidden_size};
    const bool head_wise_gate =
        gate && gate->shape == head_gate_shape;
    const bool element_wise_gate =
        gate && gate->shape == element_gate_shape;

    if (!q_a || !q_a_norm || !q_b || !kv_a || !kv_a_norm || !kv_b ||
        !out || !gate || q_rank <= 0 || kv_rank <= 0 ||
        nope <= 0 || rope <= 0 || value_dim <= 0 ||
        q_a->shape !=
            std::vector<std::int64_t>{q_rank, *m.hidden_size} ||
        q_a_norm->shape != std::vector<std::int64_t>{q_rank} ||
        q_b->shape !=
            std::vector<std::int64_t>{heads * (nope + rope), q_rank} ||
        kv_a->shape !=
            std::vector<std::int64_t>{kv_rank + rope, *m.hidden_size} ||
        kv_a_norm->shape != std::vector<std::int64_t>{kv_rank} ||
        kv_b->shape !=
            std::vector<std::int64_t>{
                heads * (nope + value_dim),
                kv_rank} ||
        out->shape !=
            std::vector<std::int64_t>{
                *m.hidden_size,
                heads * value_dim} ||
        (!head_wise_gate && !element_wise_gate)) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "factorized latent-attention tensor shapes do not agree with "
            "geometry for layer " +
                std::to_string(layer));
    }

    AttentionSpec attention =
        make_attention(m, heads, 1, value_dim, false);
    attention.query_heads = heads;
    attention.key_value_heads = 1;
    attention.head_dim = value_dim;
    attention.query_norm = {0.0f, NormWeightKind::None};
    attention.key_norm = {0.0f, NormWeightKind::None};
    attention.output_gate = {
        AttentionGateKind::Sigmoid,
        false,
        head_wise_gate
            ? AttentionGateGranularity::HeadWise
            : AttentionGateGranularity::ElementWise};
    attention.state = LatentAttentionStateSpec{
        kv_rank,
        rope,
        nope,
        true,
        true,
        q_rank,
        value_dim,
        NormSpec{*m.norm_epsilon, NormWeightKind::Scale},
        NormSpec{*m.norm_epsilon, NormWeightKind::Scale}};
    attention.query_scale =
        std::sqrt(static_cast<float>(value_dim) /
                  static_cast<float>(nope + rope));

    semantic_layer.mixer = std::move(attention);
    if (!has_ffn) semantic_layer.feed_forward = std::monostate{};
}

void infer_mamba2(
    CanonicalInferenceContext& context,
    int layer,
    LayerSpec& semantic_layer) {
    const auto& input = context.input;
    const auto& m = input.metadata;

    const auto* input_projection =
        find_mamba_tensor(input, layer, "in_proj.weight");
    const auto* convolution =
        find_mamba_tensor(input, layer, "conv1d.weight");
    const auto* convolution_bias =
        find_mamba_tensor(input, layer, "conv1d.bias");
    const auto* dt_bias =
        find_mamba_tensor(input, layer, "dt_bias");
    const auto* a_log =
        find_mamba_tensor(input, layer, "A_log");
    const auto* d =
        find_mamba_tensor(input, layer, "D");
    const auto* norm =
        find_mamba_tensor(input, layer, "norm.weight");
    const auto* output_projection =
        find_mamba_tensor(input, layer, "out_proj.weight");

    if (!input_projection || !convolution || !convolution_bias ||
        !dt_bias || !a_log || !d || !norm || !output_projection) {
        fail(
            ResolutionFailureKind::MissingTensorRole,
            "Mamba-2 tensor grammar is incomplete for layer " +
                std::to_string(layer));
    }

    const int inner = m.mamba_intermediate.value_or(
        static_cast<int>(output_projection->shape.at(1)));
    const int heads = m.mamba_num_heads.value_or(
        static_cast<int>(dt_bias->shape.at(0)));
    const int head_dim = m.mamba_head_dim.value_or(
        heads > 0 && inner % heads == 0
            ? inner / heads
            : 0);
    const int state_size = m.mamba_state_size.value_or(0);
    const int group_count = m.mamba_group_count.value_or(0);
    const int conv_kernel = m.mamba_conv_kernel.value_or(
        convolution->shape.size() == 3
            ? static_cast<int>(convolution->shape.at(2))
            : 0);
    const int time_step_rank =
        m.mamba_time_step_rank.value_or(heads);
    const int chunk_size = m.mamba_chunk_size.value_or(0);
    const int conv_dim =
        inner + 2 * group_count * state_size;

    if (inner <= 0 || heads <= 0 || head_dim <= 0 ||
        state_size <= 0 || group_count <= 0 ||
        heads % group_count != 0 || conv_kernel <= 0 ||
        conv_dim <= 0 ||
        input_projection->shape !=
            std::vector<std::int64_t>{
                2 * inner + 2 * group_count * state_size + heads,
                *m.hidden_size} ||
        convolution->shape !=
            std::vector<std::int64_t>{conv_dim, 1, conv_kernel} ||
        convolution_bias->shape !=
            std::vector<std::int64_t>{conv_dim} ||
        dt_bias->shape != std::vector<std::int64_t>{heads} ||
        a_log->shape != std::vector<std::int64_t>{heads} ||
        d->shape != std::vector<std::int64_t>{heads} ||
        norm->shape != std::vector<std::int64_t>{inner} ||
        output_projection->shape !=
            std::vector<std::int64_t>{*m.hidden_size, inner}) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "Mamba-2 tensor shapes do not agree with recurrent geometry for "
            "layer " +
                std::to_string(layer));
    }

    semantic_layer.mixer = Mamba2Spec{
        conv_kernel,
        inner,
        state_size,
        time_step_rank,
        heads,
        head_dim,
        group_count,
        chunk_size,
        true,
        false};
    semantic_layer.feed_forward = std::monostate{};
}

void infer_standard_attention(
    CanonicalInferenceContext& context,
    int layer,
    LayerSpec& semantic_layer,
    std::string_view layer_prefix,
    bool has_ffn) {
    const auto& input = context.input;
    const auto& m = input.metadata;

    const auto query_heads = m.query_heads.value_for(layer);
    const auto key_value_heads = m.key_value_heads.value_for(layer);
    const auto explicit_head_dim = m.head_dim.value_for(layer);
    if (!query_heads.has_value() || !key_value_heads.has_value()) {
        fail(
            ResolutionFailureKind::MissingRequiredMetadata,
            "automatic resolution requires attention geometry for layer " +
                std::to_string(layer));
    }
    if (*query_heads <= 0) {
        fail(
            ResolutionFailureKind::ConflictingInferenceFacts,
            "query_heads is non-positive for layer " +
                std::to_string(layer));
    }

    const int head_dim =
        explicit_head_dim.value_or(*m.hidden_size / *query_heads);
    if (*key_value_heads <= 0 ||
        *query_heads % *key_value_heads != 0 ||
        head_dim <= 0 || head_dim % 2 != 0 ||
        *key_value_heads * head_dim > *m.hidden_size) {
        fail(
            ResolutionFailureKind::ConflictingInferenceFacts,
            "attention head geometry is not a valid GQA layout for layer " +
                std::to_string(layer));
    }

    const auto has_tensor = [&](std::string_view name) {
        return input.inventory.find(name) != nullptr;
    };
    const bool has_query_norm =
        *m.query_key_norm ||
        has_tensor(std::string(layer_prefix) + "attn_q_norm.weight") ||
        has_tensor(
            "model.layers." + std::to_string(layer) +
            ".self_attn.q_layernorm.weight");
    const bool has_key_norm =
        *m.query_key_norm ||
        has_tensor(std::string(layer_prefix) + "attn_k_norm.weight") ||
        has_tensor(
            "model.layers." + std::to_string(layer) +
            ".self_attn.k_layernorm.weight");
    if (has_query_norm != has_key_norm) {
        fail(
            ResolutionFailureKind::ConflictingInferenceFacts,
            "query/key normalization evidence is incomplete for layer " +
                std::to_string(layer));
    }

    AttentionSpec attention = make_attention(
        m,
        *query_heads,
        *key_value_heads,
        head_dim,
        has_query_norm);

    const auto q_candidates =
        attention_tensor_candidates(layer, "q_proj.weight");
    const TensorInventoryEntry* query = nullptr;
    for (const std::string& name : q_candidates) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (query != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "multiple query projections are present for layer " +
                        std::to_string(layer));
            }
            query = candidate;
        }
    }
    if (query &&
        shape_is(
            *query,
            {2 * attention.query_width(), *m.hidden_size})) {
        attention.output_gate = {
            AttentionGateKind::Sigmoid,
            true,
            AttentionGateGranularity::ElementWise};
    }

    semantic_layer.mixer = std::move(attention);
    if (!has_ffn) semantic_layer.feed_forward = std::monostate{};
}

} // namespace

void infer_layer_semantics(CanonicalInferenceContext& context) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& graph = context.facts.graph;
    auto& numerical_policy = context.facts.numerical_policy;

    for (int layer = 0; layer < context.layer_count; ++layer) {
        LayerSpec& semantic_layer =
            graph.layers[static_cast<size_t>(layer)];
        const auto has_tensor = [&](std::string_view name) {
            return input.inventory.find(name) != nullptr;
        };

        const std::string layer_prefix =
            "blk." + std::to_string(layer) + ".";
        const std::string model_layer_prefix =
            "model.layers." + std::to_string(layer) + ".";

        const bool has_kda =
            has_tensor(model_layer_prefix + "attention.q_proj.weight") &&
            has_tensor(model_layer_prefix + "attention.f_proj.weight") &&
            has_tensor(model_layer_prefix + "attention.q_conv1d.weight");
        const bool has_mla =
            has_tensor(model_layer_prefix + "attention.q_a_proj.weight");
        const bool has_fused_gated_delta =
            has_tensor(layer_prefix + "attn_qkv.weight") &&
            has_tensor(layer_prefix + "attn_gate.weight") &&
            has_tensor(layer_prefix + "ssm_alpha.weight") &&
            has_tensor(layer_prefix + "ssm_beta.weight") &&
            has_tensor(layer_prefix + "ssm_conv1d.weight") &&
            has_tensor(layer_prefix + "ssm_dt.bias") &&
            has_tensor(layer_prefix + "ssm_a") &&
            has_tensor(layer_prefix + "ssm_norm.weight") &&
            has_tensor(layer_prefix + "ssm_out.weight");
        const bool has_attention =
            has_tensor(layer_prefix + "attn_q.weight") ||
            has_tensor(
                "model.layers." + std::to_string(layer) +
                ".self_attn.q_proj.weight") ||
            has_tensor(
                "model.language_model.layers." + std::to_string(layer) +
                ".self_attn.q_proj.weight") ||
            has_tensor(
                "transformer.h." + std::to_string(layer) +
                ".attn.q_proj.weight") ||
            has_mla;
        const bool has_mamba =
            find_mamba_tensor(input, layer, "in_proj.weight") != nullptr;
        const bool has_shortconv =
            has_tensor(layer_prefix + "shortconv.in_proj.weight") ||
            has_tensor(
                "model.layers." + std::to_string(layer) +
                ".conv.in_proj.weight") ||
            has_tensor(
                "model.language_model.layers." + std::to_string(layer) +
                ".conv.in_proj.weight");
        const bool has_ffn =
            find_mamba_tensor(input, layer, "in_proj.weight") == nullptr &&
            (has_tensor(layer_prefix + "ffn_up.weight") ||
             has_tensor(
                 "model.layers." + std::to_string(layer) +
                 ".mlp.up_proj.weight") ||
             has_tensor(
                 "model.language_model.layers." + std::to_string(layer) +
                 ".mlp.up_proj.weight") ||
             has_tensor(
                 "transformer.h." + std::to_string(layer) +
                 ".mlp.w_up.weight") ||
             has_tensor(
                 "model.layers." + std::to_string(layer) +
                 ".feed_forward.w1.weight") ||
             (context.has_moe &&
              has_tensor(
                  model_layer_prefix +
                  "mlp.experts.0.gate_proj.weight")));

        if ((has_mamba || has_fused_gated_delta) && has_attention) {
            fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "layer has mutually exclusive attention and recurrent tensor "
                "grammars: " +
                    std::to_string(layer));
        }

        if (has_fused_gated_delta) {
            infer_fused_gated_delta(
                context,
                layer,
                semantic_layer,
                layer_prefix,
                has_ffn);
            continue;
        }
        if (has_kda) {
            infer_factorized_gated_delta(
                context,
                layer,
                semantic_layer,
                model_layer_prefix,
                has_ffn);
            continue;
        }
        if (has_mla) {
            infer_latent_attention(
                context,
                layer,
                semantic_layer,
                model_layer_prefix,
                has_ffn);
            continue;
        }
        if (has_mamba) {
            infer_mamba2(context, layer, semantic_layer);
            continue;
        }
        if (!has_attention) {
            if (has_shortconv) {
                if (!m.shortconv_cache.has_value() ||
                    *m.shortconv_cache <= 0) {
                    fail(
                        ResolutionFailureKind::MissingRequiredMetadata,
                        "short-convolution layer has no positive cache length");
                }
                semantic_layer.mixer = ShortConvolutionSpec{
                    *m.shortconv_cache,
                    *m.hidden_size,
                    false};
                if (!has_ffn) semantic_layer.feed_forward = std::monostate{};
                continue;
            }
            if (!has_ffn) {
                fail(
                    ResolutionFailureKind::UnsupportedGraphPrimitive,
                    "layer has no recognized mixer or FFN tensor grammar: " +
                        std::to_string(layer));
            }
            semantic_layer.mixer = MlpBlockSpec{
                context.intermediate_sizes.at(
                    static_cast<size_t>(layer)),
                ActivationKind::Relu2};
            semantic_layer.feed_forward = std::monostate{};
            continue;
        }

        infer_standard_attention(
            context,
            layer,
            semantic_layer,
            layer_prefix,
            has_ffn);
    }

    if (m.attention_multiplier.has_value()) {
        numerical_policy.attention_multiplier =
            *m.attention_multiplier;
    } else {
        int attention_head_dim = 0;
        for (int layer = 0; layer < context.layer_count; ++layer) {
            if (std::holds_alternative<AttentionSpec>(
                    graph.layers[static_cast<size_t>(layer)].mixer)) {
                attention_head_dim = std::get<AttentionSpec>(
                    graph.layers[static_cast<size_t>(layer)].mixer)
                                         .head_dim;
                break;
            }
        }
        numerical_policy.attention_multiplier =
            attention_head_dim > 0
                ? 1.0f /
                      std::sqrt(
                          static_cast<float>(attention_head_dim))
                : 1.0f;
    }

    for (LayerSpec& semantic_layer : graph.layers) {
        if (auto* attention =
                std::get_if<AttentionSpec>(&semantic_layer.mixer);
            attention != nullptr && attention->query_heads > 0) {
            attention->query_scale *=
                numerical_policy.attention_multiplier;
        }
    }
    graph.validate();
}

} // namespace celeg::inference_detail
