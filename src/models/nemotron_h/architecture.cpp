#include "celeg/models/nemotron_h/architecture.hpp"

#include "celeg/model/weight_plan.hpp"
#include "naming_policy.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace celeg::detail {
namespace {

constexpr std::string_view kDefaultPattern =
    "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-";

int integer(const CheckpointMetadata& m, std::string_view json,
            std::string_view gguf) {
    return static_cast<int>(m.integer_for(json, gguf));
}

int integer_or(const CheckpointMetadata& m, std::string_view json,
               std::string_view gguf, int fallback) {
    return static_cast<int>(m.integer_for_or(json, gguf, fallback));
}

int first_nonzero(const CheckpointMetadata& m, std::string_view key,
                  int fallback) {
    if (!m.contains(key)) return fallback;
    for (const int64_t value : m.integers(key)) {
        if (value != 0) return static_cast<int>(value);
    }
    return fallback;
}

double number_or(const CheckpointMetadata& m, std::string_view json,
                 std::string_view gguf, double fallback) {
    return m.number_for_or(json, gguf, fallback);
}

std::shared_ptr<const ITensorNamingPolicy> policy() {
    static const auto value = std::make_shared<const NemotronHTensorNamingPolicy>();
    return value;
}

bool contains_layer_tensor(const CheckpointView& checkpoint, int layer,
                           std::string_view suffix) {
    const std::string a = "backbone.layers." + std::to_string(layer) + "." +
                          std::string(suffix);
    const std::string gguf = "blk." + std::to_string(layer) + "." + std::string(suffix);
    return checkpoint.repository->contains(a) || checkpoint.repository->contains("model." + a) ||
           checkpoint.repository->contains(gguf);
}

std::string layer_pattern(const CheckpointView& checkpoint, int count) {
    if (!checkpoint.metadata.is_gguf()) {
        const std::string pattern = checkpoint.metadata.string_or(
            "hybrid_override_pattern", std::string(kDefaultPattern));
        if (static_cast<int>(pattern.size()) != count) {
            throw std::runtime_error("Nemotron-H hybrid pattern length mismatch");
        }
        return pattern;
    }
    std::string pattern;
    pattern.reserve(static_cast<size_t>(count));
    for (int layer = 0; layer < count; ++layer) {
        if (contains_layer_tensor(checkpoint, layer, "ssm_in") ||
            contains_layer_tensor(checkpoint, layer, "mixer.in_proj.weight")) pattern += 'M';
        else if (contains_layer_tensor(checkpoint, layer, "attn_q") ||
                 contains_layer_tensor(checkpoint, layer, "mixer.q_proj.weight")) pattern += '*';
        else if (contains_layer_tensor(checkpoint, layer, "ffn_up") ||
                 contains_layer_tensor(checkpoint, layer, "mixer.up_proj.weight") ||
                 contains_layer_tensor(checkpoint, layer, "mixer.down_proj.weight")) pattern += '-';
        else throw std::runtime_error("cannot infer Nemotron-H GGUF block " + std::to_string(layer));
    }
    return pattern;
}

RuntimeTopology decode_topology(const CheckpointView& checkpoint) {
    const auto& m = checkpoint.metadata;
    RuntimeTopology t;
    t.hidden = integer(m, "hidden_size", "nemotron_h.embedding_length");
    t.intermediate = m.is_gguf()
        ? first_nonzero(m, "nemotron_h.feed_forward_length", 12544)
        : integer(m, "intermediate_size", "nemotron_h.feed_forward_length");
    t.dense_intermediate = t.intermediate;
    t.max_feed_forward_intermediate = t.intermediate;
    t.num_hidden_layers = integer(m, "num_hidden_layers", "nemotron_h.block_count");
    t.vocab_size = integer(m, "vocab_size", "nemotron_h.vocab_size");
    t.max_position_embeddings = integer_or(m, "max_position_embeddings",
                                           "nemotron_h.context_length", 262144);
    t.token_policy.bos_token_id = integer_or(m, "bos_token_id", "tokenizer.ggml.bos_token_id", 1);
    t.token_policy.eos_token_ids = {integer(m, "eos_token_id", "tokenizer.ggml.eos_token_id")};
    t.token_policy.pad_token_id = integer_or(m, "pad_token_id", "tokenizer.ggml.padding_token_id", 0);
    t.numerical_policy.norm_eps = static_cast<float>(number_or(
        m, "rms_norm_eps", "nemotron_h.attention.layer_norm_epsilon", 1.0e-5));
    t.numerical_policy.attention_multiplier = 1.0f / std::sqrt(128.0f);
    t.mixer_kinds.resize(static_cast<size_t>(t.num_hidden_layers));
    t.feed_forward_kinds.assign(static_cast<size_t>(t.num_hidden_layers), FeedForwardKind::Dense);
    t.feed_forward_intermediates.assign(static_cast<size_t>(t.num_hidden_layers), t.intermediate);
    t.feed_forward_activations.assign(static_cast<size_t>(t.num_hidden_layers), ActivationKind::Relu2);
    t.attention_layouts.resize(static_cast<size_t>(t.num_hidden_layers));
    t.attention_slot_for_layer.assign(static_cast<size_t>(t.num_hidden_layers), -1);
    t.mamba2_layouts.resize(static_cast<size_t>(t.num_hidden_layers));
    t.mlp_only_layouts.resize(static_cast<size_t>(t.num_hidden_layers));

    const std::string pattern = layer_pattern(checkpoint, t.num_hidden_layers);
    const int query_heads = integer(m, "num_attention_heads", "nemotron_h.attention.head_count");
    const int kv_heads = m.is_gguf()
        ? first_nonzero(m, "nemotron_h.attention.head_count_kv", 8)
        : integer(m, "num_key_value_heads", "attention.head_count_kv");
    const int attention_dim = integer_or(m, "head_dim", "nemotron_h.attention.key_length", 128);
    const int mamba_heads = integer(m, "mamba_num_heads", "nemotron_h.ssm.time_step_rank");
    const int mamba_inner = integer_or(m, "mamba_num_heads", "nemotron_h.ssm.inner_size", 0);
    const int mamba_dim = m.is_gguf()
        ? (mamba_inner / mamba_heads)
        : integer(m, "mamba_head_dim", "nemotron_h.ssm.embedding_length");
    const int state_size = integer(m, "ssm_state_size", "nemotron_h.ssm.state_size");
    const int groups = integer(m, "n_groups", "nemotron_h.ssm.group_count");
    const int kernel = integer(m, "conv_kernel", "nemotron_h.ssm.conv_kernel");
    const int chunk = integer_or(m, "chunk_size", "nemotron_h.ssm.chunk_size", 256);
    t.mamba2_intermediate = mamba_heads * mamba_dim;
    for (int layer = 0; layer < t.num_hidden_layers; ++layer) {
        switch (pattern[static_cast<size_t>(layer)]) {
        case 'M':
            t.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::Mamba2;
            ++t.mamba2_layer_count;
            t.mamba2_layouts[static_cast<size_t>(layer)] = {
                kernel, mamba_heads * mamba_dim, state_size, mamba_heads, mamba_heads,
                mamba_dim, groups, chunk, true, false};
            break;
        case '*':
            t.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::Attention;
            t.attention_slot_for_layer[static_cast<size_t>(layer)] = t.attention_layer_count++;
            t.layer_for_attention_slot.push_back(layer);
            t.attention_layouts[static_cast<size_t>(layer)] = {
                query_heads, kv_heads, attention_dim, false, AttentionMaskKind::Causal, 0,
                0.0, 1.0, {}, t.numerical_policy.attention_multiplier,
                PositionalEncodingKind::None};
            break;
        case '-':
            t.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::MlpOnly;
            ++t.mlp_only_layer_count;
            t.mlp_only_layouts[static_cast<size_t>(layer)] = {
                t.intermediate, ActivationKind::Relu2};
            break;
        default: throw std::runtime_error("unsupported Nemotron-H block type");
        }
    }
    t.validate();
    return t;
}

void build_graph(ResolvedModel& model) {
    const auto& t = model.topology;
    model.graph.final_norm.epsilon = t.numerical_policy.norm_eps;
    model.graph.layers.reserve(static_cast<size_t>(t.num_hidden_layers));
    for (int layer = 0; layer < t.num_hidden_layers; ++layer) {
        LayerSpec spec;
        spec.operator_norm.epsilon = t.numerical_policy.norm_eps;
        spec.feed_forward_norm.epsilon = t.numerical_policy.norm_eps;
        spec.feed_forward = DenseFeedForwardSpec{t.intermediate, ActivationKind::Relu2};
        spec.execute_feed_forward = false;
        switch (t.mixer_kinds[static_cast<size_t>(layer)]) {
        case MixerKind::Mamba2: spec.mixer = t.mamba2_layouts[static_cast<size_t>(layer)]; break;
        case MixerKind::Attention: spec.mixer = t.attention_layout(layer); break;
        case MixerKind::MlpOnly: spec.mixer = t.mlp_only_layouts[static_cast<size_t>(layer)]; break;
        default: throw std::runtime_error("invalid Nemotron-H graph schedule");
        }
        model.graph.layers.push_back(std::move(spec));
    }
}

void build_weights(ResolvedModel& model) {
    const auto& t = model.topology;
    auto add = [&model](TensorRole role, int layer, std::vector<int64_t> shape) {
        model.weight_plan.requests.push_back({role, layer, -1, std::move(shape)});
    };
    add(TensorRole::TokenEmbedding, -1, {t.vocab_size, t.hidden});
    add(TensorRole::LanguageModelHead, -1, {t.vocab_size, t.hidden});
    add(TensorRole::FinalNorm, -1, {t.hidden});
    for (int layer = 0; layer < t.num_hidden_layers; ++layer) {
        add(TensorRole::AttentionInputNorm, layer, {t.hidden});
        const MixerKind kind = t.mixer_kinds[static_cast<size_t>(layer)];
        if (kind == MixerKind::Attention) {
            const auto& a = t.attention_layout(layer);
            add(TensorRole::AttentionQuery, layer, {a.query_width(), t.hidden});
            add(TensorRole::AttentionKey, layer, {a.key_value_width(), t.hidden});
            add(TensorRole::AttentionValue, layer, {a.key_value_width(), t.hidden});
            add(TensorRole::AttentionOutput, layer, {t.hidden, a.query_width()});
        } else if (kind == MixerKind::Mamba2) {
            const auto& s = t.mamba2_layouts[static_cast<size_t>(layer)];
            const int conv_dim = s.intermediate_size + 2 * s.group_count * s.state_size;
            add(TensorRole::Mamba2Input, layer,
                {2 * s.intermediate_size + 2 * s.group_count * s.state_size + s.num_heads,
                 t.hidden});
            add(TensorRole::Mamba2Conv, layer, {conv_dim, 1, s.conv_kernel});
            add(TensorRole::Mamba2ConvBias, layer, {conv_dim});
            add(TensorRole::Mamba2DtBias, layer, {s.num_heads});
            add(TensorRole::Mamba2ALog, layer, {s.num_heads});
            add(TensorRole::Mamba2D, layer, {s.num_heads});
            add(TensorRole::Mamba2Norm, layer, {s.intermediate_size});
            add(TensorRole::Mamba2Output, layer, {t.hidden, s.intermediate_size});
        } else if (kind == MixerKind::MlpOnly) {
            const int width = t.mlp_only_layouts[static_cast<size_t>(layer)].intermediate_size;
            add(TensorRole::FfnUp, layer, {width, t.hidden});
            add(TensorRole::FfnDown, layer, {t.hidden, width});
        }
    }
    resolve_weight_plan(model, *policy());
}

class NemotronHArchitecture final : public IArchitecture {
public:
    std::string_view id() const override { return "nemotron_h"; }
    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        const bool supported = metadata.architecture_type() == "nemotron_h";
        return {supported, supported ? 250 : 0,
                supported ? "Nemotron-H checkpoint" : "not Nemotron-H"};
    }
    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        if (!probe(checkpoint.metadata).supported) {
            throw std::runtime_error("Nemotron-H architecture cannot resolve checkpoint");
        }
        ArchitectureResolutionStages stages;
        stages.topology = [](const CheckpointView& view) { return decode_topology(view); };
        stages.graph = [](ResolvedModel& model, const CheckpointView&) { build_graph(model); };
        stages.weights = [](ResolvedModel& model, const CheckpointView&) { build_weights(model); };
        stages.capabilities = {true, true, false, false};
        stages.provenance = {"nemotron_h", checkpoint.metadata.is_gguf() ? "gguf" : "safetensors",
                             {"nemotron_h", "", {}, "nemotron3-nano-4b"},
                             "nemotron3-nano-4b", "nemotron-h-instruct", {}};
        ResolvedModel result = resolve_architecture_stages(checkpoint, std::move(stages));
        result.provenance.identity = "nemotron3-nano-4b-" + result.topology.fingerprint();
        return result;
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_nemotron_h_architecture() {
    return std::make_unique<NemotronHArchitecture>();
}

void register_nemotron_h_architecture(ArchitectureCatalog& catalog) {
    catalog.add(make_nemotron_h_architecture());
}

} // namespace celeg::detail
