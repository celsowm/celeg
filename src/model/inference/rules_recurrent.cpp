#include "rules.hpp"

#include "support.hpp"

namespace celeg::inference_detail {
namespace {

/// Recognizes the fused gated-delta-net grammar (GGUF-mapped fused
/// attn_qkv/ssm_* projections with a single recurrent state).
class FusedGatedDeltaRule final : public ILayerInferenceRule {
public:
    std::string_view id() const override { return "fused_gated_delta"; }
    int specificity() const override { return 6; }
    MixerFamily family() const override { return MixerFamily::Recurrent; }

    bool probe(const CanonicalInferenceContext& context, int layer)
        const override {
        const auto& inventory = context.input.inventory;
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        return inventory.find(prefix + "attn_qkv.weight") != nullptr &&
            inventory.find(prefix + "attn_gate.weight") != nullptr &&
            inventory.find(prefix + "ssm_alpha.weight") != nullptr &&
            inventory.find(prefix + "ssm_beta.weight") != nullptr &&
            inventory.find(prefix + "ssm_conv1d.weight") != nullptr &&
            inventory.find(prefix + "ssm_dt.bias") != nullptr &&
            inventory.find(prefix + "ssm_a") != nullptr &&
            inventory.find(prefix + "ssm_norm.weight") != nullptr &&
            inventory.find(prefix + "ssm_out.weight") != nullptr;
    }

    void resolve(CanonicalInferenceContext& context, int layer)
        const override {
        const auto& input = context.input;
        const auto& m = input.metadata;
        LayerSpec& semantic_layer =
            context.facts.graph.layers[static_cast<size_t>(layer)];
        const std::string layer_prefix =
            "blk." + std::to_string(layer) + ".";

        const auto* qkv =
            input.inventory.find(layer_prefix + "attn_qkv.weight");
        const auto* z =
            input.inventory.find(layer_prefix + "attn_gate.weight");
        const auto* alpha =
            input.inventory.find(layer_prefix + "ssm_alpha.weight");
        const auto* beta =
            input.inventory.find(layer_prefix + "ssm_beta.weight");
        const auto* convolution =
            input.inventory.find(layer_prefix + "ssm_conv1d.weight");
        const auto* dt_bias =
            input.inventory.find(layer_prefix + "ssm_dt.bias");
        const auto* a_log =
            input.inventory.find(layer_prefix + "ssm_a");
        const auto* norm =
            input.inventory.find(layer_prefix + "ssm_norm.weight");
        const auto* output =
            input.inventory.find(layer_prefix + "ssm_out.weight");

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
            false,
            !input.is_gguf()};
        if (!layer_has_feed_forward(context, layer)) {
            semantic_layer.feed_forward = std::monostate{};
        }

        const GatedDeltaNetSpec& spec = std::get<GatedDeltaNetSpec>(
            semantic_layer.mixer);
        const int resolved_qkv_width = spec.qkv_width();
        auto& bindings = context.facts.bindings;

        const auto bind = [&](TensorRole role,
                              std::string_view suffix,
                              std::initializer_list<std::int64_t> shape) {
            const auto* tensor = find_unique(
                input.inventory,
                {layer_prefix + std::string(suffix)},
                role,
                layer,
                shape,
                {});
            add_binding(bindings, role, layer, *tensor, {});
        };

        bind(
            TensorRole::GatedDeltaNetQkv,
            "attn_qkv.weight",
            {resolved_qkv_width, *m.hidden_size});
        bind(
            TensorRole::GatedDeltaNetZ,
            "attn_gate.weight",
            {spec.value_width(), *m.hidden_size});
        bind(
            TensorRole::GatedDeltaNetAlpha,
            "ssm_alpha.weight",
            {spec.value_heads, *m.hidden_size});
        bind(
            TensorRole::GatedDeltaNetBeta,
            "ssm_beta.weight",
            {spec.value_heads, *m.hidden_size});
        bind(
            TensorRole::GatedDeltaNetDtBias,
            "ssm_dt.bias",
            {spec.value_heads});
        bind(TensorRole::GatedDeltaNetALog, "ssm_a", {spec.value_heads});
        bind(
            TensorRole::GatedDeltaNetConv,
            "ssm_conv1d.weight",
            {resolved_qkv_width, 1, spec.conv_kernel});
        bind(
            TensorRole::GatedDeltaNetNorm,
            "ssm_norm.weight",
            {spec.value_head_dim});
        bind(
            TensorRole::GatedDeltaNetOutput,
            "ssm_out.weight",
            {*m.hidden_size, spec.value_width()});
    }
};

/// Recognizes the factorized gated-delta-net grammar (separate
/// q/k/v/f/b/g projections with per-stream convolutions under an
/// attention prefix).
class FactorizedGatedDeltaRule final : public ILayerInferenceRule {
public:
    std::string_view id() const override { return "factorized_gated_delta"; }
    int specificity() const override { return 5; }
    MixerFamily family() const override { return MixerFamily::Recurrent; }

    bool probe(const CanonicalInferenceContext& context, int layer)
        const override {
        const auto& inventory = context.input.inventory;
        const std::string prefix =
            "model.layers." + std::to_string(layer) + ".attention.";
        return inventory.find(prefix + "q_proj.weight") != nullptr &&
            inventory.find(prefix + "f_proj.weight") != nullptr &&
            inventory.find(prefix + "q_conv1d.weight") != nullptr;
    }

    void resolve(CanonicalInferenceContext& context, int layer)
        const override {
        const auto& input = context.input;
        const auto& m = input.metadata;
        LayerSpec& semantic_layer =
            context.facts.graph.layers[static_cast<size_t>(layer)];
        const std::string prefix =
            "model.layers." + std::to_string(layer) + ".attention.";

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
            true,
            !input.is_gguf()};
        if (!layer_has_feed_forward(context, layer)) {
            semantic_layer.feed_forward = std::monostate{};
        }

        const GatedDeltaNetSpec& spec = std::get<GatedDeltaNetSpec>(
            semantic_layer.mixer);
        auto& bindings = context.facts.bindings;

        const auto bind = [&](TensorRole role,
                              std::string_view suffix,
                              std::initializer_list<std::int64_t> shape) {
            const auto* tensor = find_unique(
                input.inventory,
                {prefix + std::string(suffix)},
                role,
                layer,
                shape,
                {});
            add_binding(bindings, role, layer, *tensor, {});
        };

        bind(
            TensorRole::GatedDeltaNetQuery,
            "q_proj.weight",
            {spec.key_heads * spec.key_head_dim, *m.hidden_size});
        bind(
            TensorRole::GatedDeltaNetKey,
            "k_proj.weight",
            {spec.key_heads * spec.key_head_dim, *m.hidden_size});
        bind(
            TensorRole::GatedDeltaNetValue,
            "v_proj.weight",
            {spec.value_width(), *m.hidden_size});
        bind(
            TensorRole::GatedDeltaNetDecay,
            "f_proj.weight",
            {spec.decay_width(), *m.hidden_size});
        bind(
            TensorRole::GatedDeltaNetOutputGate,
            "g_proj.weight",
            {spec.value_width(), *m.hidden_size});
        bind(
            TensorRole::GatedDeltaNetQueryConv,
            "q_conv1d.weight",
            {spec.key_heads * spec.key_head_dim, 1, spec.conv_kernel});
        bind(
            TensorRole::GatedDeltaNetKeyConv,
            "k_conv1d.weight",
            {spec.key_heads * spec.key_head_dim, 1, spec.conv_kernel});
        bind(
            TensorRole::GatedDeltaNetValueConv,
            "v_conv1d.weight",
            {spec.value_width(), 1, spec.conv_kernel});
        bind(
            TensorRole::GatedDeltaNetBeta,
            "b_proj.weight",
            {spec.value_heads, *m.hidden_size});
        bind(
            TensorRole::GatedDeltaNetDtBias,
            "dt_bias",
            {spec.decay_width()});
        bind(TensorRole::GatedDeltaNetALog, "A_log", {spec.value_heads});
        bind(
            TensorRole::GatedDeltaNetNorm,
            "o_norm.weight",
            {spec.value_head_dim});
        bind(
            TensorRole::GatedDeltaNetOutput,
            "o_proj.weight",
            {*m.hidden_size, spec.value_width()});
    }
};

/// Recognizes the Mamba-2 grammar (ssm/mixer projections in any known
/// spelling convention). Mamba-2 layers never carry a feed-forward axis.
class Mamba2Rule final : public ILayerInferenceRule {
public:
    std::string_view id() const override { return "mamba2"; }
    int specificity() const override { return 3; }
    MixerFamily family() const override { return MixerFamily::Recurrent; }

    bool probe(const CanonicalInferenceContext& context, int layer)
        const override {
        return find_mamba_tensor(context.input, layer, "in_proj.weight") !=
            nullptr;
    }

    void resolve(CanonicalInferenceContext& context, int layer)
        const override {
        const auto& input = context.input;
        const auto& m = input.metadata;
        LayerSpec& semantic_layer =
            context.facts.graph.layers[static_cast<size_t>(layer)];

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
            false,
            !input.is_gguf()};
        semantic_layer.feed_forward = std::monostate{};

        const Mamba2Spec& spec = std::get<Mamba2Spec>(semantic_layer.mixer);
        auto& bindings = context.facts.bindings;

        const auto bind = [&](TensorRole role,
                              std::string_view suffix,
                              std::initializer_list<std::int64_t> shape) {
            const auto* tensor = find_unique(
                input.inventory,
                mamba2_tensor_candidates(layer, suffix),
                role,
                layer,
                shape,
                {});
            add_binding(bindings, role, layer, *tensor, {});
        };

        const int resolved_conv_dim = spec.intermediate_size +
            2 * spec.group_count * spec.state_size;
        bind(
            TensorRole::Mamba2Input,
            "in_proj.weight",
            {2 * spec.intermediate_size +
                 2 * spec.group_count * spec.state_size + spec.num_heads,
             *m.hidden_size});
        bind(
            TensorRole::Mamba2Conv,
            "conv1d.weight",
            {resolved_conv_dim, 1, spec.conv_kernel});
        bind(TensorRole::Mamba2ConvBias, "conv1d.bias", {resolved_conv_dim});
        bind(TensorRole::Mamba2DtBias, "dt_bias", {spec.num_heads});
        bind(TensorRole::Mamba2ALog, "A_log", {spec.num_heads});
        bind(TensorRole::Mamba2D, "D", {spec.num_heads});
        bind(
            TensorRole::Mamba2Norm,
            "norm.weight",
            {spec.intermediate_size});
        bind(
            TensorRole::Mamba2Output,
            "out_proj.weight",
            {*m.hidden_size, spec.intermediate_size});
    }
};

}

std::unique_ptr<ILayerInferenceRule> make_fused_gated_delta_rule() {
    return std::make_unique<FusedGatedDeltaRule>();
}

std::unique_ptr<ILayerInferenceRule> make_factorized_gated_delta_rule() {
    return std::make_unique<FactorizedGatedDeltaRule>();
}

std::unique_ptr<ILayerInferenceRule> make_mamba2_rule() {
    return std::make_unique<Mamba2Rule>();
}

}
