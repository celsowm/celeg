#include "../rules.hpp"

#include "../support.hpp"

#include <string>

namespace celeg::inference_detail {
namespace {

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

        const int inner = m.mamba2.intermediate.value_or(
            static_cast<int>(output_projection->shape.at(1)));
        const int heads = m.mamba2.num_heads.value_or(
            static_cast<int>(dt_bias->shape.at(0)));
        const int head_dim = m.mamba2.head_dim.value_or(
            heads > 0 && inner % heads == 0
                ? inner / heads
                : 0);
        const int state_size = m.mamba2.state_size.value_or(0);
        const int group_count = m.mamba2.group_count.value_or(0);
        const int conv_kernel = m.mamba2.conv_kernel.value_or(
            convolution->shape.size() == 3
                ? static_cast<int>(convolution->shape.at(2))
                : 0);
        const int time_step_rank =
            m.mamba2.time_step_rank.value_or(heads);
        const int chunk_size = m.mamba2.chunk_size.value_or(0);
        const int conv_dim =
            inner + 2 * group_count * state_size;

        if (inner <= 0 || heads <= 0 || head_dim <= 0 ||
            state_size <= 0 || group_count <= 0 ||
            heads % group_count != 0 || conv_kernel <= 0 ||
            conv_dim <= 0 ||
            input_projection->shape !=
                std::vector<std::int64_t>{
                    2 * inner + 2 * group_count * state_size + heads,
                    *m.core.hidden_size} ||
            convolution->shape !=
                std::vector<std::int64_t>{conv_dim, 1, conv_kernel} ||
            convolution_bias->shape !=
                std::vector<std::int64_t>{conv_dim} ||
            dt_bias->shape != std::vector<std::int64_t>{heads} ||
            a_log->shape != std::vector<std::int64_t>{heads} ||
            d->shape != std::vector<std::int64_t>{heads} ||
            norm->shape != std::vector<std::int64_t>{inner} ||
            output_projection->shape !=
                std::vector<std::int64_t>{*m.core.hidden_size, inner}) {
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
            m.mamba2.decay_encoding == DecayParameterEncoding::LogA};
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
             *m.core.hidden_size});
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
            {*m.core.hidden_size, spec.intermediate_size});
    }
};

}

std::unique_ptr<ILayerInferenceRule> make_mamba2_rule() {
    return std::make_unique<Mamba2Rule>();
}

}
