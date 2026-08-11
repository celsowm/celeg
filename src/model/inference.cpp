#include "celeg/model/inference.hpp"

#include "inference/support.hpp"

#include <regex>
#include <sstream>
#include <unordered_set>

namespace celeg {
namespace {

void require_positive(const std::optional<int>& value, std::string_view name) {
    if (!value.has_value()) {
        inference_detail::fail(ResolutionFailureKind::MissingRequiredMetadata,
                               "automatic resolution requires metadata: " +
                                   std::string(name));
    }
    if (*value <= 0) {
        inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                               "automatic resolution received a non-positive " +
                                   std::string(name));
    }
}

} // namespace

ResolutionError::ResolutionError(ResolutionFailureKind kind, std::string message,
                                 std::vector<EvidenceItem> evidence)
    : std::runtime_error(std::move(message)), kind_(kind), evidence_(std::move(evidence)) {}

CanonicalModelFacts infer_canonical_model_facts(const InferenceInput& input) {
    const auto& m = input.metadata;
    require_positive(m.hidden_size, "hidden_size");
    require_positive(m.layer_count, "layer_count");
    require_positive(m.vocab_size, "vocab_size");
    require_positive(m.context_length, "context_length");
    if (!m.bos_token_id.has_value() || !m.pad_token_id.has_value() ||
        *m.bos_token_id < 0 || *m.pad_token_id < 0 || m.eos_token_ids.empty()) {
        inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                               "token IDs are incomplete");
    }

    const TensorInventoryEntry* embedding = nullptr;
    for (const std::string& name : {std::string("transformer.wte.weight"),
                                    std::string("model.embed_tokens.weight"),
                                    std::string("tok_embeddings.weight"),
                                    std::string("token_embd.weight")}) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (embedding != nullptr) {
                inference_detail::fail(ResolutionFailureKind::AmbiguousTensorBinding,
                                       "multiple token-embedding candidates are present");
            }
            embedding = candidate;
        }
    }
    if (embedding == nullptr) {
        inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                               "automatic resolution could not find token embedding");
    }
    if (!inference_detail::shape_is(*embedding, {*m.vocab_size, *m.hidden_size})) {
        inference_detail::fail(ResolutionFailureKind::ShapeConstraintViolation,
                               "token embedding shape does not agree with normalized metadata");
    }

    std::unordered_set<int> layers;
    const std::regex layer_pattern(R"((?:transformer\.h|model\.layers|layers|blk)\.(\d+)\.)");
    for (const auto& entry : input.inventory.entries()) {
        std::smatch match;
        if (std::regex_search(entry.name, match, layer_pattern)) {
            layers.insert(std::stoi(match[1].str()));
        }
    }
    if (layers.size() != static_cast<size_t>(*m.layer_count)) {
        inference_detail::fail(ResolutionFailureKind::IncompleteLayerSchedule,
                               "tensor inventory layer count does not agree with metadata");
    }
    for (int layer = 0; layer < *m.layer_count; ++layer) {
        if (!layers.contains(layer)) {
            inference_detail::fail(
                ResolutionFailureKind::IncompleteLayerSchedule,
                "tensor inventory has a gap at layer " + std::to_string(layer));
        }
    }

    CanonicalModelFacts facts;
    facts.resolution_mode = "automatic";
    facts.tied_embeddings = m.tied_embeddings.value_or(false);
    facts.evidence = m.evidence;
    RuntimeTopology& topology = facts.topology;
    topology.hidden = *m.hidden_size;
    std::vector<int> intermediate_sizes;
    intermediate_sizes.reserve(static_cast<size_t>(*m.layer_count));
    for (int layer = 0; layer < *m.layer_count; ++layer) {
        const std::optional<int> value = m.intermediate_size.value_for(layer);
        const std::string index = std::to_string(layer);
        const TensorInventoryEntry* ffn_up = input.inventory.find(
            "blk." + index + ".ffn_up.weight");
        if (!ffn_up) {
            ffn_up = input.inventory.find(
                "model.layers." + index + ".mlp.up_proj.weight");
        }
        if (!value.has_value() || *value <= 0) {
            if (ffn_up && ffn_up->shape.size() == 2 && ffn_up->shape[0] > 0) {
                intermediate_sizes.push_back(static_cast<int>(ffn_up->shape[0]));
            } else {
                // Mixer-only layers deliberately carry a positive compiled
                // placeholder; their FFN execution and weight requests are
                // disabled by the per-layer schedule below.
                intermediate_sizes.push_back(1);
            }
        } else {
            intermediate_sizes.push_back(*value);
        }
    }
    topology.intermediate = *std::max_element(intermediate_sizes.begin(), intermediate_sizes.end());
    topology.dense_intermediate = topology.intermediate;
    topology.max_feed_forward_intermediate = topology.intermediate;
    topology.num_hidden_layers = *m.layer_count;
    topology.vocab_size = *m.vocab_size;
    topology.max_position_embeddings = *m.context_length;
    topology.mixer_kinds.assign(static_cast<size_t>(*m.layer_count), MixerKind::Attention);
    topology.feed_forward_kinds.assign(static_cast<size_t>(*m.layer_count),
                                        FeedForwardKind::Dense);
    topology.execute_feed_forward.assign(static_cast<size_t>(*m.layer_count), true);
    topology.feed_forward_intermediates = intermediate_sizes;
    topology.feed_forward_activations.assign(static_cast<size_t>(*m.layer_count),
                                             ActivationKind::SwiGLU);
    topology.attention_layouts.resize(static_cast<size_t>(*m.layer_count));
    topology.mamba2_layouts.resize(static_cast<size_t>(*m.layer_count));
    topology.mlp_only_layouts.resize(static_cast<size_t>(*m.layer_count));
    topology.attention_slot_for_layer.assign(static_cast<size_t>(*m.layer_count), -1);
    topology.layer_for_attention_slot.reserve(static_cast<size_t>(*m.layer_count));
    topology.attention_layer_count = 0;
    topology.conv_cache = m.shortconv_cache.value_or(0);
    topology.conv_dim = *m.hidden_size;
    topology.num_dense_layers = *m.layer_count;
    topology.shared_kv_group_count = 0;
    topology.token_policy = {*m.bos_token_id, m.eos_token_ids, *m.pad_token_id};
    topology.numerical_policy.norm_eps = *m.norm_epsilon;
    topology.numerical_policy.attention_multiplier = 0.125f;

    const auto make_attention = [&](int query_heads, int key_value_heads, int head_dim,
                                    bool query_key_norm) {
        AttentionSpec attention;
        attention.query_heads = query_heads;
        attention.key_value_heads = key_value_heads;
        attention.head_dim = head_dim;
        attention.query_norm = {query_key_norm ? *m.norm_epsilon : 0.0f,
                                query_key_norm ? NormWeightKind::Scale
                                                : NormWeightKind::None};
        attention.key_norm = attention.query_norm;
        attention.pattern = FullCausalPattern{};
        attention.query_scale = 1.0f;
        attention.position = RopePositionSpec{*m.rope_theta, *m.rotary_fraction,
                                              RopeScalingSpec{}};
        std::get<RopePositionSpec>(attention.position).pairing = *m.rope_pairing;
        attention.state_storage = AttentionStateStorageSpec{};
        if (*m.xsa_projection) {
            attention.output_transform = OrthogonalizeCurrentValueSpec{
                *m.xsa_minimum_norm_squared};
        }
        return attention;
    };
    for (int layer = 0; layer < *m.layer_count; ++layer) {
        const auto has_tensor = [&](std::string_view name) {
            return input.inventory.find(name) != nullptr;
        };
        const auto find_mamba_tensor = [&](std::string_view suffix) {
            const auto candidates = inference_detail::mamba2_tensor_candidates(layer, suffix);
            const TensorInventoryEntry* found = nullptr;
            for (const auto& candidate : candidates) {
                if (const auto* tensor = input.inventory.find(candidate)) {
                    if (found != nullptr) {
                        inference_detail::fail(
                            ResolutionFailureKind::AmbiguousTensorBinding,
                            "multiple Mamba-2 tensor spellings are present for layer " +
                                std::to_string(layer));
                    }
                    found = tensor;
                }
            }
            return found;
        };
        const std::string layer_prefix = "blk." + std::to_string(layer) + ".";
        const bool has_attention =
            has_tensor(layer_prefix + "attn_q.weight") ||
            has_tensor("model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight") ||
            has_tensor("transformer.h." + std::to_string(layer) + ".attn.q_proj.weight");
        const bool has_mamba = find_mamba_tensor("in_proj.weight") != nullptr;
        const bool has_shortconv = has_tensor(layer_prefix + "shortconv.in_proj.weight") ||
            has_tensor("model.layers." + std::to_string(layer) + ".conv.in_proj.weight");
        const bool has_ffn =
            find_mamba_tensor("in_proj.weight") == nullptr &&
            (has_tensor(layer_prefix + "ffn_up.weight") ||
             has_tensor("model.layers." + std::to_string(layer) + ".mlp.up_proj.weight"));
        const auto query_heads = m.query_heads.value_for(layer);
        const auto key_value_heads = m.key_value_heads.value_for(layer);
        const auto explicit_head_dim = m.head_dim.value_for(layer);
        if (has_mamba && has_attention) {
            inference_detail::fail(ResolutionFailureKind::ConflictingInferenceFacts,
                                   "layer has both attention and Mamba-2 tensor grammars: " +
                                       std::to_string(layer));
        }
        if (has_mamba) {
            const auto* input_projection = find_mamba_tensor("in_proj.weight");
            const auto* convolution = find_mamba_tensor("conv1d.weight");
            const auto* convolution_bias = find_mamba_tensor("conv1d.bias");
            const auto* dt_bias = find_mamba_tensor("dt_bias");
            const auto* a_log = find_mamba_tensor("A_log");
            const auto* d = find_mamba_tensor("D");
            const auto* norm = find_mamba_tensor("norm.weight");
            const auto* output_projection = find_mamba_tensor("out_proj.weight");
            if (!input_projection || !convolution || !convolution_bias || !dt_bias ||
                !a_log || !d || !norm || !output_projection) {
                inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                                       "Mamba-2 tensor grammar is incomplete for layer " +
                                           std::to_string(layer));
            }
            const int inner = m.mamba_intermediate.value_or(
                static_cast<int>(output_projection->shape.at(1)));
            const int heads = m.mamba_num_heads.value_or(
                static_cast<int>(dt_bias->shape.at(0)));
            const int head_dim = m.mamba_head_dim.value_or(
                heads > 0 && inner % heads == 0 ? inner / heads : 0);
            const int state_size = m.mamba_state_size.value_or(0);
            const int group_count = m.mamba_group_count.value_or(0);
            const int conv_kernel = m.mamba_conv_kernel.value_or(
                convolution->shape.size() == 3 ? static_cast<int>(convolution->shape.at(2)) : 0);
            const int time_step_rank = m.mamba_time_step_rank.value_or(heads);
            const int chunk_size = m.mamba_chunk_size.value_or(0);
            const int conv_dim = inner + 2 * group_count * state_size;
            if (inner <= 0 || heads <= 0 || head_dim <= 0 || state_size <= 0 ||
                group_count <= 0 || heads % group_count != 0 || conv_kernel <= 0 ||
                conv_dim <= 0 || input_projection->shape !=
                    std::vector<std::int64_t>{2 * inner + 2 * group_count * state_size + heads,
                                               *m.hidden_size} ||
                convolution->shape != std::vector<std::int64_t>{conv_dim, 1, conv_kernel} ||
                convolution_bias->shape != std::vector<std::int64_t>{conv_dim} ||
                dt_bias->shape != std::vector<std::int64_t>{heads} ||
                a_log->shape != std::vector<std::int64_t>{heads} ||
                d->shape != std::vector<std::int64_t>{heads} ||
                norm->shape != std::vector<std::int64_t>{inner} ||
                output_projection->shape != std::vector<std::int64_t>{*m.hidden_size, inner}) {
                inference_detail::fail(ResolutionFailureKind::ShapeConstraintViolation,
                                       "Mamba-2 tensor shapes do not agree with recurrent geometry for layer " +
                                           std::to_string(layer));
            }
            topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::Mamba2;
            topology.mamba2_layouts[static_cast<size_t>(layer)] =
                Mamba2Spec{conv_kernel, inner, state_size, time_step_rank, heads, head_dim,
                           group_count, chunk_size, true, false};
            ++topology.mamba2_layer_count;
            topology.mamba2_intermediate = std::max(topology.mamba2_intermediate, inner);
            topology.execute_feed_forward[static_cast<size_t>(layer)] = false;
            continue;
        }
        if (!has_attention) {
            if (has_shortconv) {
                if (!m.shortconv_cache.has_value() || *m.shortconv_cache <= 0) {
                    inference_detail::fail(
                        ResolutionFailureKind::MissingRequiredMetadata,
                        "short-convolution layer has no positive cache length");
                }
                topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::ShortConvolution;
                ++topology.conv_layer_count;
                topology.execute_feed_forward[static_cast<size_t>(layer)] = has_ffn;
                continue;
            }
            if (!has_ffn) {
                inference_detail::fail(ResolutionFailureKind::UnsupportedGraphPrimitive,
                                       "layer has no recognized mixer or FFN tensor grammar: " +
                                           std::to_string(layer));
            }
            topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::MlpOnly;
            topology.mlp_only_layouts[static_cast<size_t>(layer)] =
                MlpBlockSpec{intermediate_sizes.at(static_cast<size_t>(layer)),
                             ActivationKind::Relu2};
            ++topology.mlp_only_layer_count;
            topology.execute_feed_forward[static_cast<size_t>(layer)] = false;
            continue;
        }
        if (!query_heads.has_value() || !key_value_heads.has_value()) {
            inference_detail::fail(
                ResolutionFailureKind::MissingRequiredMetadata,
                "automatic resolution requires attention geometry for layer " +
                    std::to_string(layer));
        }
        if (*query_heads <= 0) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "query_heads is non-positive for layer " + std::to_string(layer));
        }
        const int head_dim = explicit_head_dim.value_or(*m.hidden_size / *query_heads);
        if (*key_value_heads <= 0 || *query_heads % *key_value_heads != 0 ||
            head_dim <= 0 || head_dim % 2 != 0 ||
            *key_value_heads * head_dim > *m.hidden_size) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "attention head geometry is not a valid GQA layout for layer " +
                    std::to_string(layer));
        }
        const bool has_query_norm =
            *m.query_key_norm || has_tensor(layer_prefix + "attn_q_norm.weight");
        const bool has_key_norm =
            *m.query_key_norm || has_tensor(layer_prefix + "attn_k_norm.weight");
        if (has_query_norm != has_key_norm) {
            inference_detail::fail(ResolutionFailureKind::ConflictingInferenceFacts,
                                   "query/key normalization evidence is incomplete for layer " +
                                       std::to_string(layer));
        }
        topology.attention_slot_for_layer[static_cast<size_t>(layer)] =
            topology.attention_layer_count++;
        topology.layer_for_attention_slot.push_back(layer);
        topology.attention_layouts[static_cast<size_t>(layer)] =
            make_attention(*query_heads, *key_value_heads, head_dim, has_query_norm);
        topology.execute_feed_forward[static_cast<size_t>(layer)] = has_ffn;
    }
    topology.validate();

    const auto add_global = [&](TensorRole role, const TensorInventoryEntry& tensor) {
        inference_detail::add_binding(
            facts.bindings, role, -1, tensor,
            {{EvidenceKind::TensorName, tensor.name, std::string(tensor_role_name(role))}});
    };
    add_global(TensorRole::TokenEmbedding, *embedding);

    const std::vector<std::string> head_names = {
        "lm_head.weight", "transformer.lm_head.weight", "output.weight"};
    const TensorInventoryEntry* head = nullptr;
    for (const std::string& name : head_names) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (head != nullptr) {
                inference_detail::fail(ResolutionFailureKind::AmbiguousTensorBinding,
                                       "multiple language-model heads are present");
            }
            head = candidate;
        }
    }
    if (head == nullptr) {
        if (input.source_format == CheckpointSourceFormat::Gguf &&
            !m.tied_embeddings.has_value()) {
            facts.tied_embeddings = true;
            facts.evidence.push_back({EvidenceKind::FormatGuarantee, "output.weight",
                                      "GGUF omits an independent language-model head"});
        } else if (!facts.tied_embeddings) {
            inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                                   "untied checkpoint has no language-model head");
        }
        head = embedding;
        facts.evidence.push_back({EvidenceKind::Derived, embedding->name,
                                  "language-model head is tied to token embedding"});
    } else if (!inference_detail::shape_is(*head, {*m.vocab_size, *m.hidden_size})) {
        inference_detail::fail(ResolutionFailureKind::ShapeConstraintViolation,
                               "language-model head shape does not agree with normalized metadata");
    }
    add_global(TensorRole::LanguageModelHead, *head);

    const std::vector<std::string> final_norm_names = {
        "transformer.ln_f.weight", "model.norm.weight", "norm.weight", "output_norm.weight",
        "token_embd_norm.weight"};
    const TensorInventoryEntry* final_norm = nullptr;
    for (const auto& name : final_norm_names) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (final_norm != nullptr) {
                inference_detail::fail(ResolutionFailureKind::AmbiguousTensorBinding,
                                       "multiple final norms are present");
            }
            final_norm = candidate;
        }
    }
    if (final_norm == nullptr) {
        inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                               "automatic resolution could not find final norm");
    }
    if (!inference_detail::shape_is(*final_norm, {*m.hidden_size})) {
        inference_detail::fail(ResolutionFailureKind::ShapeConstraintViolation,
                               "final norm shape mismatch");
    }
    add_global(TensorRole::FinalNorm, *final_norm);

    for (int layer = 0; layer < *m.layer_count; ++layer) {
        const std::string index = std::to_string(layer);
        const std::string layer_prefix = "blk." + index + ".";
        const auto has_tensor = [&](std::string_view name) {
            return input.inventory.find(name) != nullptr;
        };
        const int query_head_count = *m.query_heads.value_for(layer);
        const int key_value_head_count = *m.key_value_heads.value_for(layer);
        const int layer_head_dim = m.head_dim.value_for(layer).value_or(
            *m.hidden_size / query_head_count);
        const int layer_intermediate = intermediate_sizes.at(static_cast<size_t>(layer));
        const std::vector<std::string> norm_candidates = {
            "transformer.h." + index + ".ln_1.weight",
            "model.layers." + index + ".input_layernorm.weight",
            "model.layers." + index + ".self_attn_layer_norm.weight",
            "blk." + index + ".attn_norm.weight",
        };
        const std::vector<std::string> ffn_norm_candidates = {
            "transformer.h." + index + ".ln_2.weight",
            "model.layers." + index + ".post_attention_layernorm.weight",
            "blk." + index + ".ffn_norm.weight",
        };
        const auto* attention_norm = inference_detail::find_unique(
            input.inventory, norm_candidates, TensorRole::AttentionInputNorm, layer,
            {*m.hidden_size}, {});
        inference_detail::add_binding(facts.bindings, TensorRole::AttentionInputNorm, layer,
                                      *attention_norm, {});
        if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Mamba2) {
            const auto bind_mamba = [&](TensorRole role, std::string_view suffix,
                                        std::initializer_list<std::int64_t> shape) {
                const auto* tensor = inference_detail::find_unique(
                    input.inventory,
                    inference_detail::mamba2_tensor_candidates(layer, suffix),
                    role, layer, shape, {});
                inference_detail::add_binding(facts.bindings, role, layer, *tensor, {});
            };
            const Mamba2Spec& spec = topology.mamba2_layouts.at(static_cast<size_t>(layer));
            const int conv_dim = spec.intermediate_size +
                2 * spec.group_count * spec.state_size;
            bind_mamba(TensorRole::Mamba2Input, "in_proj.weight",
                       {2 * spec.intermediate_size + 2 * spec.group_count * spec.state_size +
                        spec.num_heads, *m.hidden_size});
            bind_mamba(TensorRole::Mamba2Conv, "conv1d.weight",
                       {conv_dim, 1, spec.conv_kernel});
            bind_mamba(TensorRole::Mamba2ConvBias, "conv1d.bias", {conv_dim});
            bind_mamba(TensorRole::Mamba2DtBias, "dt_bias", {spec.num_heads});
            bind_mamba(TensorRole::Mamba2ALog, "A_log", {spec.num_heads});
            bind_mamba(TensorRole::Mamba2D, "D", {spec.num_heads});
            bind_mamba(TensorRole::Mamba2Norm, "norm.weight", {spec.intermediate_size});
            bind_mamba(TensorRole::Mamba2Output, "out_proj.weight",
                       {*m.hidden_size, spec.intermediate_size});
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::ShortConvolution) {
            const auto* input_projection = inference_detail::find_unique(
                input.inventory,
                inference_detail::shortconv_tensor_candidates(layer, "in_proj.weight"),
                TensorRole::ShortConvInput, layer,
                {3 * *m.hidden_size, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::ShortConvInput,
                                          layer, *input_projection, {});
            const auto* kernel = inference_detail::find_unique(
                input.inventory,
                inference_detail::shortconv_tensor_candidates(layer, "conv.weight"),
                TensorRole::ShortConvKernel, layer,
                {*m.hidden_size, 1, topology.conv_cache}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::ShortConvKernel,
                                          layer, *kernel, {});
            const auto* output_projection = inference_detail::find_unique(
                input.inventory,
                inference_detail::shortconv_tensor_candidates(layer, "out_proj.weight"),
                TensorRole::ShortConvOutput, layer,
                {*m.hidden_size, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::ShortConvOutput,
                                          layer, *output_projection, {});
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
            const auto* q = inference_detail::find_unique(
                input.inventory,
                inference_detail::attention_tensor_candidates(layer, "q_proj.weight"),
                TensorRole::AttentionQuery, layer,
                {query_head_count * layer_head_dim, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::AttentionQuery,
                                          layer, *q, {});
            const auto* k = inference_detail::find_unique(
                input.inventory,
                inference_detail::attention_tensor_candidates(layer, "k_proj.weight"),
                TensorRole::AttentionKey, layer,
                {key_value_head_count * layer_head_dim, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::AttentionKey,
                                          layer, *k, {});
            const auto* v = inference_detail::find_unique(
                input.inventory,
                inference_detail::attention_tensor_candidates(layer, "v_proj.weight"),
                TensorRole::AttentionValue, layer,
                {key_value_head_count * layer_head_dim, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::AttentionValue,
                                          layer, *v, {});
            const auto* o = inference_detail::find_unique(
                input.inventory,
                inference_detail::attention_tensor_candidates(layer, "o_proj.weight"),
                TensorRole::AttentionOutput, layer,
                {*m.hidden_size, query_head_count * layer_head_dim}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::AttentionOutput,
                                          layer, *o, {});
            if (has_tensor(layer_prefix + "attn_q_norm.weight")) {
                const auto* q_norm = inference_detail::find_unique(
                    input.inventory,
                    {layer_prefix + "attn_q_norm.weight"},
                    TensorRole::AttentionQueryNorm, layer,
                    {layer_head_dim}, {});
                inference_detail::add_binding(facts.bindings, TensorRole::AttentionQueryNorm,
                                              layer, *q_norm, {});
            }
            if (has_tensor(layer_prefix + "attn_k_norm.weight")) {
                const auto* k_norm = inference_detail::find_unique(
                    input.inventory,
                    {layer_prefix + "attn_k_norm.weight"},
                    TensorRole::AttentionKeyNorm, layer,
                    {layer_head_dim}, {});
                inference_detail::add_binding(facts.bindings, TensorRole::AttentionKeyNorm,
                                              layer, *k_norm, {});
            }
        }
        const MixerKind mixer = topology.mixer_kinds[static_cast<size_t>(layer)];
        if (mixer == MixerKind::MlpOnly) {
            const auto* up = inference_detail::find_unique(
                input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                    "w_up.weight"), TensorRole::FfnUp, layer,
                {layer_intermediate, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnUp, layer, *up, {});
            const auto* down = inference_detail::find_unique(
                input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                    "w_down.weight"), TensorRole::FfnDown, layer,
                {*m.hidden_size, layer_intermediate}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnDown, layer, *down, {});
        } else if (topology.execute_feed_forward.at(static_cast<size_t>(layer))) {
            const auto* ffn_norm = inference_detail::find_unique(
                input.inventory, ffn_norm_candidates, TensorRole::FfnInputNorm, layer,
                {*m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnInputNorm, layer,
                                          *ffn_norm, {});
            const auto* gate = inference_detail::find_unique(
                input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                    "w_gate.weight"), TensorRole::FfnGate, layer,
                {layer_intermediate, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnGate, layer, *gate, {});
            const auto* up = inference_detail::find_unique(
                input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                    "w_up.weight"), TensorRole::FfnUp, layer,
                {layer_intermediate, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnUp, layer, *up, {});
            const auto* down = inference_detail::find_unique(
                input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                    "w_down.weight"), TensorRole::FfnDown, layer,
                {*m.hidden_size, layer_intermediate}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnDown, layer, *down, {});
        }
    }
    facts.bindings = BindingSolver{}.solve(facts.bindings.values);
    facts.validate();
    return facts;
}

std::string CanonicalModelFacts::fingerprint() const {
    std::ostringstream out;
    out << resolution_mode << ':' << source_format << ':' << topology.fingerprint()
        << ":tied=" << tied_embeddings;
    for (const auto& binding : bindings.values) {
        out << ':' << static_cast<int>(binding.role) << ':' << binding.layer << ':'
            << binding.expert << ':' << binding.source_name;
        for (const auto dimension : binding.shape) out << ':' << dimension;
    }
    return out.str();
}

void CanonicalModelFacts::validate() const {
    topology.validate();
    bindings.validate();
    const auto require = [&](TensorRole role, int layer) {
        if (bindings.find(role, layer) == nullptr) {
            inference_detail::fail(
                ResolutionFailureKind::MissingTensorRole,
                "canonical facts omit required tensor role " +
                    std::string(tensor_role_name(role)));
        }
    };
    require(TensorRole::TokenEmbedding, -1);
    require(TensorRole::LanguageModelHead, -1);
    require(TensorRole::FinalNorm, -1);
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        require(TensorRole::AttentionInputNorm, layer);
        if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
            require(TensorRole::AttentionQuery, layer);
            require(TensorRole::AttentionKey, layer);
            require(TensorRole::AttentionValue, layer);
            require(TensorRole::AttentionOutput, layer);
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] ==
                   MixerKind::ShortConvolution) {
            require(TensorRole::ShortConvInput, layer);
            require(TensorRole::ShortConvKernel, layer);
            require(TensorRole::ShortConvOutput, layer);
        } else {
            if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Mamba2) {
                require(TensorRole::Mamba2Input, layer);
                require(TensorRole::Mamba2Conv, layer);
                require(TensorRole::Mamba2ConvBias, layer);
                require(TensorRole::Mamba2DtBias, layer);
                require(TensorRole::Mamba2ALog, layer);
                require(TensorRole::Mamba2D, layer);
                require(TensorRole::Mamba2Norm, layer);
                require(TensorRole::Mamba2Output, layer);
            } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::MlpOnly) {
                require(TensorRole::FfnUp, layer);
                require(TensorRole::FfnDown, layer);
            } else {
                inference_detail::fail(ResolutionFailureKind::UnsupportedGraphPrimitive,
                                       "automatic resolution has no binding contract for mixer");
            }
        }
        if (topology.mixer_kinds[static_cast<size_t>(layer)] != MixerKind::MlpOnly &&
            (topology.execute_feed_forward.empty() ||
             topology.execute_feed_forward.at(static_cast<size_t>(layer)))) {
            require(TensorRole::FfnInputNorm, layer);
            require(TensorRole::FfnGate, layer);
            require(TensorRole::FfnUp, layer);
            require(TensorRole::FfnDown, layer);
        }
    }
}

} // namespace celeg
