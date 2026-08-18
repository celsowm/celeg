from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def edit(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if old in text:
        file.write_text(text.replace(old, new), encoding="utf-8")
        return
    if new in text:
        return
    raise RuntimeError(f"expected fragment not found in {path}: {old[:120]!r}")


def edit_all(path: str, replacements: list[tuple[str, str]]) -> None:
    for old, new in replacements:
        edit(path, old, new)


edit(
    "include/celeg/model/graph.hpp",
    "struct LayerSpec {\n    NormSpec operator_norm;\n    std::optional<NormSpec> post_attention_norm;\n    std::optional<NormSpec> pre_feed_forward_norm;\n    std::optional<NormSpec> post_feed_forward_norm;\n    MixerSpec mixer;\n    std::optional<NormSpec> feed_forward_norm;\n    FeedForwardSpec feed_forward;\n    ResidualSpec residual;\n    float layer_scalar = 1.0f;\n};",
    "struct SublayerNormSpec {\n    std::optional<NormSpec> before;\n    std::optional<NormSpec> after;\n\n    friend bool operator==(const SublayerNormSpec&, const SublayerNormSpec&) = default;\n};\n\nstruct LayerSpec {\n    SublayerNormSpec mixer_norm;\n    MixerSpec mixer;\n    SublayerNormSpec feed_forward_norm;\n    FeedForwardSpec feed_forward;\n    ResidualSpec residual;\n    float layer_scalar = 1.0f;\n};",
)

edit(
    "include/celeg/model/program.hpp",
    "    NormSpec operator_norm;\n    std::optional<NormSpec> post_attention_norm;\n    std::optional<NormSpec> feed_forward_norm;\n    std::optional<NormSpec> post_feed_forward_norm;\n    ResidualSpec residual;",
    "    SublayerNormSpec mixer_norm;\n    SublayerNormSpec feed_forward_norm;\n    ResidualSpec residual;",
)

edit_all(
    "src/model/program_builder.cpp",
    [
        ("        compiled.operator_norm = layer.operator_norm;\n        compiled.post_attention_norm = layer.post_attention_norm;\n        compiled.feed_forward_norm = layer.feed_forward_norm;\n        compiled.post_feed_forward_norm = layer.post_feed_forward_norm;",
         "        compiled.mixer_norm = layer.mixer_norm;\n        compiled.feed_forward_norm = layer.feed_forward_norm;"),
    ],
)

edit(
    "src/model/resolved.cpp",
    "        append_norm(out, layer.operator_norm);\n        append_optional_norm(out, layer.post_attention_norm);\n        append_optional_norm(out, layer.pre_feed_forward_norm);\n        append_optional_norm(out, layer.post_feed_forward_norm);\n        append_mixer(out, layer);\n        append_optional_norm(out, layer.feed_forward_norm);\n        append_feed_forward(out, layer);",
    "        append_optional_norm(out, layer.mixer_norm.before);\n        append_optional_norm(out, layer.mixer_norm.after);\n        append_mixer(out, layer);\n        append_optional_norm(out, layer.feed_forward_norm.before);\n        append_optional_norm(out, layer.feed_forward_norm.after);\n        append_feed_forward(out, layer);",
)

edit(
    "src/model/resolved.cpp",
    "        if (!(layer.operator_norm.epsilon > 0.0f) ||\n            !std::isfinite(layer.residual.multiplier) ||\n            !std::isfinite(layer.layer_scalar)) {\n            throw std::runtime_error(\"resolved model graph has invalid layer policy\");\n        }\n        layer.operator_norm.validate();\n        if (layer.feed_forward_norm) layer.feed_forward_norm->validate();\n        if (layer.post_attention_norm) layer.post_attention_norm->validate();\n        if (layer.pre_feed_forward_norm) layer.pre_feed_forward_norm->validate();\n        if (layer.post_feed_forward_norm) layer.post_feed_forward_norm->validate();",
    "        if (!std::isfinite(layer.residual.multiplier) ||\n            !std::isfinite(layer.layer_scalar)) {\n            throw std::runtime_error(\"resolved model graph has invalid layer policy\");\n        }\n        if (layer.mixer_norm.before) layer.mixer_norm.before->validate();\n        if (layer.mixer_norm.after) layer.mixer_norm.after->validate();\n        if (layer.feed_forward_norm.before) layer.feed_forward_norm.before->validate();\n        if (layer.feed_forward_norm.after) layer.feed_forward_norm.after->validate();",
)

edit(
    "src/model/descriptor/graph_builder.cpp",
    "        layer.operator_norm = {epsilon, descriptor.operator_norm_kind};\n        layer.feed_forward_norm = {epsilon, descriptor.feed_forward_norm_kind};\n        if (descriptor.split_attention_norms) {\n            layer.post_attention_norm = {post_epsilon, descriptor.operator_norm_kind};\n            layer.pre_feed_forward_norm = {epsilon, descriptor.feed_forward_norm_kind};\n            layer.post_feed_forward_norm = {post_epsilon, descriptor.feed_forward_norm_kind};\n        }",
    "        layer.mixer_norm.before = NormSpec{epsilon, descriptor.operator_norm_kind};\n        layer.feed_forward_norm.before = NormSpec{epsilon, descriptor.feed_forward_norm_kind};\n        if (descriptor.split_attention_norms) {\n            layer.mixer_norm.after = NormSpec{post_epsilon, descriptor.operator_norm_kind};\n            layer.feed_forward_norm.after = NormSpec{post_epsilon, descriptor.feed_forward_norm_kind};\n        }",
)

edit(
    "src/model/weight_plan.cpp",
    "        add_norm_request(model, TensorRole::AttentionInputNorm, layer_index,\n                         std::optional<NormSpec>{layer.operator_norm}, {graph.hidden}, physical_layer);\n        append_mixer(model, layer, layer_index, physical_layer);\n        const bool has_feed_forward =\n            !std::holds_alternative<std::monostate>(layer.feed_forward);\n        if (has_feed_forward) {\n            add_norm_request(model, TensorRole::FfnInputNorm, layer_index,\n                             layer.feed_forward_norm, {graph.hidden}, physical_layer);\n        }",
    "        add_norm_request(model, TensorRole::AttentionInputNorm, layer_index,\n                         layer.mixer_norm.before, {graph.hidden}, physical_layer);\n        add_norm_request(model, TensorRole::AttentionPostNorm, layer_index,\n                         layer.mixer_norm.after, {graph.hidden}, physical_layer);\n        append_mixer(model, layer, layer_index, physical_layer);\n        const bool has_feed_forward =\n            !std::holds_alternative<std::monostate>(layer.feed_forward);\n        if (has_feed_forward) {\n            add_norm_request(model, TensorRole::FfnInputNorm, layer_index,\n                             layer.feed_forward_norm.before, {graph.hidden}, physical_layer);\n            add_norm_request(model, TensorRole::FfnOutputNorm, layer_index,\n                             layer.feed_forward_norm.after, {graph.hidden}, physical_layer);\n        }",
)

edit(
    "src/model/weight_plan.cpp",
    "        add_norm_request(model, TensorRole::AttentionPostNorm, layer_index,\n                         layer.post_attention_norm, {graph.hidden}, physical_layer);\n        add_norm_request(model, TensorRole::FfnOutputNorm, layer_index,\n                         layer.post_feed_forward_norm, {graph.hidden}, physical_layer);\n",
    "",
)

edit(
    "include/celeg/model/inference.hpp",
    "struct InferredRopePosition {\n    double theta = 0.0;\n    float rotary_fraction = 1.0f;\n    RopePairingKind pairing = RopePairingKind::SplitHalf;\n\n    friend bool operator==(const InferredRopePosition&, const InferredRopePosition&) = default;\n};",
    "struct InferredRopePosition {\n    double theta = 0.0;\n    float rotary_fraction = 1.0f;\n    RopePairingKind pairing = RopePairingKind::SplitHalf;\n    RopeScalingSpec scaling;\n\n    friend bool operator==(const InferredRopePosition&, const InferredRopePosition&) = default;\n};",
)

edit(
    "include/celeg/model/inference.hpp",
    "struct AttentionFacts {\n    LayerScopedValue<int> query_heads;\n    LayerScopedValue<int> key_value_heads;\n    LayerScopedValue<int> head_dim;\n    std::optional<float> attention_multiplier;\n    InferredPositionEncoding position_encoding = UnresolvedPositionEncoding{};\n    std::optional<bool> query_key_norm;\n    std::optional<bool> xsa_projection;\n    std::optional<float> xsa_minimum_norm_squared;\n};",
    "struct AttentionFacts {\n    LayerScopedValue<int> query_heads;\n    LayerScopedValue<int> key_value_heads;\n    LayerScopedValue<int> head_dim;\n    LayerScopedValue<std::string> layer_type;\n    LayerScopedValue<int> sliding_window;\n    std::optional<float> attention_multiplier;\n    InferredPositionEncoding position_encoding = UnresolvedPositionEncoding{};\n    std::optional<bool> query_key_norm;\n    std::optional<bool> xsa_projection;\n    std::optional<float> xsa_minimum_norm_squared;\n};\n\nstruct NormTopologyFacts {\n    LayerScopedValue<bool> mixer_before;\n    LayerScopedValue<bool> mixer_after;\n    LayerScopedValue<bool> feed_forward_before;\n    LayerScopedValue<bool> feed_forward_after;\n    LayerScopedValue<std::string> layer_layout;\n};",
)

edit(
    "include/celeg/model/inference.hpp",
    "    AttentionFacts attention;\n    LatentAttentionFacts latent_attention;",
    "    AttentionFacts attention;\n    NormTopologyFacts norms;\n    LatentAttentionFacts latent_attention;",
)

edit(
    "src/checkpoint/metadata.cpp",
    "        bool all_strings = true;\n        bool all_numbers = true;\n        bool all_integers = true;\n        std::vector<std::string> strings;\n        std::vector<double> numbers;\n        std::vector<int64_t> integers;",
    "        bool all_strings = true;\n        bool all_numbers = true;\n        bool all_integers = true;\n        bool all_bools = true;\n        std::vector<std::string> strings;\n        std::vector<double> numbers;\n        std::vector<int64_t> integers;\n        std::vector<int64_t> booleans;",
)
edit(
    "src/checkpoint/metadata.cpp",
    "        for (const Json& item : value.as_array()) {\n            if (!item.is_string()) all_strings = false;\n            else strings.push_back(item.as_string());\n            double special_number = 0.0;",
    "        for (const Json& item : value.as_array()) {\n            if (!item.is_string()) all_strings = false;\n            else strings.push_back(item.as_string());\n            if (!item.is_bool()) all_bools = false;\n            else booleans.push_back(item.as_bool() ? 1 : 0);\n            double special_number = 0.0;",
)
edit(
    "src/checkpoint/metadata.cpp",
    "        if (all_strings) metadata.values[prefix] = std::move(strings);\n        else if (all_numbers && all_integers) metadata.values[prefix] = std::move(integers);",
    "        if (all_strings) metadata.values[prefix] = std::move(strings);\n        else if (all_bools) metadata.values[prefix] = std::move(booleans);\n        else if (all_numbers && all_integers) metadata.values[prefix] = std::move(integers);",
)

edit(
    "src/model/inference/metadata.cpp",
    "void validate_scoped_alias(const LayerScopedValue<int>& value,\n                           const std::optional<int>& layer_count,\n                           std::string_view fact) {",
    "template <typename T>\nvoid validate_scoped_alias(const LayerScopedValue<T>& value,\n                           const std::optional<int>& layer_count,\n                           std::string_view fact) {",
)

insert_anchor = "template <typename T>\nstd::optional<T> aliases(const CheckpointMetadata& metadata,"
insert_text = r'''LayerScopedValue<std::string> scoped_string_aliases(
    const CheckpointMetadata& metadata,
    std::initializer_list<std::string_view> keys,
    std::vector<EvidenceItem>& evidence,
    std::string_view fact) {
    LayerScopedValue<std::string> result;
    std::string source;
    const auto consider = [&](std::string_view key) {
        if (!metadata.contains(key)) return;
        const MetadataValue& value = metadata.value(key);
        if (const auto* scalar_value = std::get_if<std::string>(&value)) {
            if (result.global && *result.global != *scalar_value) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "conflicting metadata aliases for " + std::string(fact));
            }
            result.global = *scalar_value;
            source = key;
            return;
        }
        if (const auto* values = std::get_if<std::vector<std::string>>(&value)) {
            std::vector<std::optional<std::string>> schedule;
            schedule.reserve(values->size());
            for (const std::string& item : *values) schedule.push_back(item);
            if (!result.per_layer.empty() && result.per_layer != schedule) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "conflicting layer-scoped metadata for " + std::string(fact));
            }
            result.per_layer = std::move(schedule);
            source = key;
            return;
        }
        inference_detail::fail(
            ResolutionFailureKind::ConflictingMetadata,
            "metadata key has an incompatible type: " + std::string(key));
    };
    for (const std::string_view key : keys) {
        consider(key);
        consider("text_config." + std::string(key));
    }
    if (result.has_value()) {
        evidence.push_back({EvidenceKind::AliasMetadata, source,
                            std::string(fact) + (result.per_layer.empty()
                                ? " = " + *result.global
                                : " = layer-scoped schedule")});
    }
    return result;
}

'''
file = ROOT / "src/model/inference/metadata.cpp"
text = file.read_text(encoding="utf-8")
if insert_text not in text:
    if insert_anchor not in text:
        raise RuntimeError("metadata aliases insertion anchor not found")
    file.write_text(text.replace(insert_anchor, insert_text + insert_anchor), encoding="utf-8")

edit(
    "src/model/inference/metadata.cpp",
    "    result.attention.head_dim = scoped_aliases<int>(metadata, {\"head_dim\"}, result.evidence, \"head_dim\",\n                                                    \"attention.key_length\");",
    "    result.attention.head_dim = scoped_aliases<int>(metadata, {\"head_dim\"}, result.evidence, \"head_dim\",\n                                                    \"attention.key_length\");\n    result.attention.layer_type = scoped_string_aliases(\n        metadata, {\"layer_types\", \"attention_types\"}, result.evidence, \"layer_type\");\n    result.attention.sliding_window = scoped_aliases<int>(\n        metadata, {\"sliding_window\", \"sliding_window_size\"}, result.evidence,\n        \"sliding_window\", \"attention.sliding_window\");\n    result.norms.mixer_before = scoped_aliases<bool>(\n        metadata, {\"use_pre_attn_norm\", \"use_pre_attention_norm\"}, result.evidence,\n        \"mixer_norm.before\");\n    result.norms.mixer_after = scoped_aliases<bool>(\n        metadata, {\"use_post_attn_norm\", \"use_post_attention_norm\"}, result.evidence,\n        \"mixer_norm.after\");\n    result.norms.feed_forward_before = scoped_aliases<bool>(\n        metadata, {\"use_pre_mlp_norm\", \"use_pre_ffn_norm\"}, result.evidence,\n        \"feed_forward_norm.before\");\n    result.norms.feed_forward_after = scoped_aliases<bool>(\n        metadata, {\"use_post_mlp_norm\", \"use_post_ffn_norm\"}, result.evidence,\n        \"feed_forward_norm.after\");\n    result.norms.layer_layout = scoped_string_aliases(\n        metadata, {\"layer_layouts\"}, result.evidence, \"layer_layout\");",
)

edit(
    "src/model/inference/metadata.cpp",
    "    validate_scoped_alias(result.attention.head_dim, result.core.layer_count, \"head_dim\");\n",
    "    validate_scoped_alias(result.attention.head_dim, result.core.layer_count, \"head_dim\");\n    validate_scoped_alias(result.attention.layer_type, result.core.layer_count, \"layer_type\");\n    validate_scoped_alias(result.attention.sliding_window, result.core.layer_count, \"sliding_window\");\n    validate_scoped_alias(result.norms.mixer_before, result.core.layer_count, \"mixer_norm.before\");\n    validate_scoped_alias(result.norms.mixer_after, result.core.layer_count, \"mixer_norm.after\");\n    validate_scoped_alias(result.norms.feed_forward_before, result.core.layer_count, \"feed_forward_norm.before\");\n    validate_scoped_alias(result.norms.feed_forward_after, result.core.layer_count, \"feed_forward_norm.after\");\n    validate_scoped_alias(result.norms.layer_layout, result.core.layer_count, \"layer_layout\");\n",
)

edit(
    "src/model/inference/metadata.cpp",
    "    if (!result.attention.query_key_norm.has_value()) result.attention.query_key_norm = false;\n",
    "",
)

edit(
    "src/model/inference/metadata.cpp",
    "        result.attention.position_encoding = InferredRopePosition{\n            *rope_theta,\n            rotary_fraction.value_or(1.0f),\n            pairing};",
    "        RopeScalingSpec scaling = NoRopeScaling{};\n        const std::string rope_type = metadata.string_or(\n            \"rope_scaling.rope_type\", metadata.string_or(\"rope_scaling.type\", {}));\n        if (!rope_type.empty()) {\n            if (rope_type == \"yarn\") {\n                const double factor = metadata.number_or(\"rope_scaling.factor\", 1.0);\n                const double attention_factor = metadata.number_or(\"rope_scaling.attention_factor\",\n                    0.1 * std::log(std::max(1.0, factor)) + 1.0);\n                const double beta_fast = metadata.number_or(\"rope_scaling.beta_fast\", 32.0);\n                const double beta_slow = metadata.number_or(\"rope_scaling.beta_slow\", 1.0);\n                scaling = YarnRopeScaling{factor, attention_factor, beta_fast, beta_slow};\n            } else if (rope_type == \"linear\") {\n                scaling = LinearRopeScaling{metadata.number_or(\"rope_scaling.factor\", 1.0)};\n            } else if (rope_type != \"default\" && rope_type != \"none\") {\n                inference_detail::fail(\n                    ResolutionFailureKind::UnsupportedSemanticFeature,\n                    \"unsupported RoPE scaling type: \" + rope_type);\n            }\n        }\n        result.attention.position_encoding = InferredRopePosition{\n            *rope_theta,\n            rotary_fraction.value_or(1.0f),\n            pairing,\n            std::move(scaling)};",
)

edit(
    "src/model/inference/support.hpp",
    "std::vector<std::string> mamba2_tensor_candidates(int layer,\n                                                  std::string_view suffix);",
    "std::vector<std::string> mamba2_tensor_candidates(int layer,\n                                                  std::string_view suffix);\nstd::vector<std::string> norm_tensor_candidates(int layer, TensorRole role);\nbool has_any_tensor(const TensorInventory& inventory,\n                    const std::vector<std::string>& candidates);",
)

support_anchor = "void add_binding(TensorRoleBindings& bindings, TensorRole role, int layer,"
support_text = r'''std::vector<std::string> norm_tensor_candidates(int layer, TensorRole role) {
    const std::string index = std::to_string(layer);
    switch (role) {
    case TensorRole::AttentionInputNorm:
        return {
            "transformer.h." + index + ".ln_1.weight",
            "model.layers." + index + ".input_layernorm.weight",
            "model.layers." + index + ".self_attn_layer_norm.weight",
            "model.language_model.layers." + index + ".input_layernorm.weight",
            "model.language_model.layers." + index + ".operator_norm.weight",
            "model.layers." + index + ".operator_norm.weight",
            "model.layers." + index + ".pre_attn_norm.weight",
            "model.language_model.layers." + index + ".pre_attn_norm.weight",
            "backbone.layers." + index + ".norm.weight",
            "blk." + index + ".attn_norm.weight",
        };
    case TensorRole::AttentionPostNorm:
        return {
            "model.layers." + index + ".post_attn_norm.weight",
            "model.language_model.layers." + index + ".post_attn_norm.weight",
            "model.layers." + index + ".post_attention_norm.weight",
            "model.language_model.layers." + index + ".post_attention_norm.weight",
            "blk." + index + ".post_attention_norm.weight",
        };
    case TensorRole::FfnInputNorm:
        return {
            "transformer.h." + index + ".ln_2.weight",
            "model.layers." + index + ".post_attention_layernorm.weight",
            "model.language_model.layers." + index + ".post_attention_layernorm.weight",
            "model.layers." + index + ".pre_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".pre_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".ffn_norm.weight",
            "model.layers." + index + ".ffn_norm.weight",
            "model.layers." + index + ".pre_mlp_norm.weight",
            "model.language_model.layers." + index + ".pre_mlp_norm.weight",
            "blk." + index + ".ffn_norm.weight",
        };
    case TensorRole::FfnOutputNorm:
        return {
            "model.layers." + index + ".post_mlp_norm.weight",
            "model.language_model.layers." + index + ".post_mlp_norm.weight",
            "model.layers." + index + ".post_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".post_feedforward_layernorm.weight",
            "model.layers." + index + ".post_feed_forward_norm.weight",
            "model.language_model.layers." + index + ".post_feed_forward_norm.weight",
            "blk." + index + ".post_ffw_norm.weight",
        };
    default:
        throw std::invalid_argument("tensor role is not a sublayer norm role");
    }
}

bool has_any_tensor(const TensorInventory& inventory,
                    const std::vector<std::string>& candidates) {
    for (const std::string& candidate : candidates) {
        if (inventory.find(candidate) != nullptr) return true;
    }
    return false;
}

'''
file = ROOT / "src/model/inference/support.cpp"
text = file.read_text(encoding="utf-8")
if support_text not in text:
    if support_anchor not in text:
        raise RuntimeError("support insertion anchor not found")
    file.write_text(text.replace(support_anchor, support_text + support_anchor), encoding="utf-8")

edit(
    "src/model/inference/support.cpp",
    "#include <string>\n",
    "#include <stdexcept>\n#include <string>\n",
)

edit(
    "src/model/inference/support.cpp",
    "    if (suffix == \"o_proj.weight\") gguf_suffix = \"attn_output.weight\";",
    "    if (suffix == \"o_proj.weight\") gguf_suffix = \"attn_output.weight\";\n    if (suffix == \"q_norm.weight\") gguf_suffix = \"attn_q_norm.weight\";\n    if (suffix == \"k_norm.weight\") gguf_suffix = \"attn_k_norm.weight\";",
)
edit(
    "src/model/inference/support.cpp",
    "    if (suffix == \"o_proj.weight\") {\n        result.push_back(\"model.language_model.layers.\" + index + \".self_attn.out_proj.weight\");\n        result.push_back(\"model.layers.\" + index + \".self_attn.out_proj.weight\");\n    }",
    "    if (suffix == \"o_proj.weight\") {\n        result.push_back(\"model.language_model.layers.\" + index + \".self_attn.out_proj.weight\");\n        result.push_back(\"model.layers.\" + index + \".self_attn.out_proj.weight\");\n    } else if (suffix == \"q_norm.weight\" || suffix == \"k_norm.weight\") {\n        result.push_back(\"model.language_model.layers.\" + index + \".self_attn.\" + std::string(suffix));\n        result.push_back(\"model.layers.\" + index + \".self_attn.\" + std::string(suffix));\n    }",
)

edit(
    "src/model/inference/global_facts.cpp",
    "        semantic_layer.operator_norm = {\n            numerical_policy.norm_eps,\n            NormWeightKind::Scale};\n        semantic_layer.feed_forward_norm = {\n            numerical_policy.norm_eps,\n            NormWeightKind::Scale};",
    "        const auto layout = m.norms.layer_layout.value_for(layer);\n        const auto layout_flag = [&](bool before) -> std::optional<bool> {\n            if (!layout) return std::nullopt;\n            if (*layout == \"decoder_prenorm\" || *layout == \"prenorm\") return before;\n            if (*layout == \"decoder_postnorm\" || *layout == \"postnorm\") return !before;\n            if (*layout == \"decoder_sandwich\" || *layout == \"sandwich\") return true;\n            fail(ResolutionFailureKind::UnsupportedSemanticFeature,\n                 \"unsupported layer layout token: \" + *layout);\n        };\n        const auto resolve_norm = [&](const LayerScopedValue<bool>& metadata_flag,\n                                      TensorRole role, bool before) -> std::optional<NormSpec> {\n            const std::optional<bool> explicit_flag = metadata_flag.value_for(layer);\n            const std::optional<bool> layout_default = layout_flag(before);\n            const bool tensor_present = has_any_tensor(\n                context.input.inventory, norm_tensor_candidates(layer, role));\n            const std::optional<bool> selected = explicit_flag.has_value()\n                ? explicit_flag : layout_default;\n            if (selected.has_value()) {\n                if (!*selected && tensor_present) {\n                    fail(ResolutionFailureKind::ConflictingInferenceFacts,\n                         \"normalization metadata disables a tensor-backed norm for layer \" +\n                             std::to_string(layer));\n                }\n                if (!*selected) return std::nullopt;\n                return NormSpec{numerical_policy.norm_eps, NormWeightKind::Scale};\n            }\n            return tensor_present\n                ? std::optional<NormSpec>{NormSpec{numerical_policy.norm_eps, NormWeightKind::Scale}}\n                : std::nullopt;\n        };\n        semantic_layer.mixer_norm.before = resolve_norm(\n            m.norms.mixer_before, TensorRole::AttentionInputNorm, true);\n        semantic_layer.mixer_norm.after = resolve_norm(\n            m.norms.mixer_after, TensorRole::AttentionPostNorm, false);\n        semantic_layer.feed_forward_norm.before = resolve_norm(\n            m.norms.feed_forward_before, TensorRole::FfnInputNorm, true);\n        semantic_layer.feed_forward_norm.after = resolve_norm(\n            m.norms.feed_forward_after, TensorRole::FfnOutputNorm, false);",
)

edit(
    "src/model/inference/rules_attention.cpp",
    "    int head_dim,\n    bool query_key_norm) {",
    "    int layer,\n    int head_dim,\n    bool query_key_norm) {",
)
edit(
    "src/model/inference/rules_attention.cpp",
    "    attention.pattern = FullCausalPattern{};\n    attention.query_scale = 1.0f;",
    "    const auto layer_type = metadata.attention.layer_type.value_for(layer);\n    if (!layer_type || *layer_type == \"full_attention\" || *layer_type == \"full\") {\n        attention.pattern = FullCausalPattern{};\n    } else if (*layer_type == \"sliding_attention\" || *layer_type == \"sliding_window\" ||\n               *layer_type == \"sliding\") {\n        const auto window = metadata.attention.sliding_window.value_for(layer);\n        if (!window || *window <= 0) {\n            fail(ResolutionFailureKind::UnsupportedSemanticFeature,\n                 \"sliding attention requires a positive sliding_window for layer \" +\n                     std::to_string(layer));\n        }\n        attention.pattern = SlidingWindowPattern{*window};\n    } else {\n        fail(ResolutionFailureKind::UnsupportedSemanticFeature,\n             \"unsupported attention layer type token: \" + *layer_type);\n    }\n    attention.query_scale = 1.0f;",
)
edit(
    "src/model/inference/rules_attention.cpp",
    "            RopePositionSpec rope{position.theta, position.rotary_fraction, RopeScalingSpec{}};",
    "            RopePositionSpec rope{position.theta, position.rotary_fraction, position.scaling};",
)
edit(
    "src/model/inference/rules_attention.cpp",
    "        const bool has_query_norm =\n            *m.attention.query_key_norm ||\n            has_tensor(layer_prefix + \"attn_q_norm.weight\") ||\n            has_tensor(\n                \"model.layers.\" + index + \".self_attn.q_layernorm.weight\");\n        const bool has_key_norm =\n            *m.attention.query_key_norm ||\n            has_tensor(layer_prefix + \"attn_k_norm.weight\") ||\n            has_tensor(\n                \"model.layers.\" + index + \".self_attn.k_layernorm.weight\");\n        if (has_query_norm != has_key_norm) {",
    "        const bool tensor_query_norm =\n            has_any_tensor(input.inventory, attention_tensor_candidates(layer, \"q_norm.weight\")) ||\n            has_tensor(\"model.layers.\" + index + \".self_attn.q_layernorm.weight\");\n        const bool tensor_key_norm =\n            has_any_tensor(input.inventory, attention_tensor_candidates(layer, \"k_norm.weight\")) ||\n            has_tensor(\"model.layers.\" + index + \".self_attn.k_layernorm.weight\");\n        if (tensor_query_norm != tensor_key_norm) {",
)
edit(
    "src/model/inference/rules_attention.cpp",
    "                \"query/key normalization evidence is incomplete for layer \" +\n                    std::to_string(layer));\n        }\n\n        AttentionSpec attention = make_attention(\n            m,\n            *query_heads,\n            *key_value_heads,\n            head_dim,\n            has_query_norm);",
    "                \"query/key normalization evidence is incomplete for layer \" +\n                    std::to_string(layer));\n        }\n        if (m.attention.query_key_norm.has_value() &&\n            !*m.attention.query_key_norm && tensor_query_norm) {\n            fail(ResolutionFailureKind::ConflictingInferenceFacts,\n                 \"query/key norm metadata conflicts with checkpoint tensors for layer \" +\n                     std::to_string(layer));\n        }\n        const bool has_query_key_norm = m.attention.query_key_norm.value_or(tensor_query_norm);\n\n        AttentionSpec attention = make_attention(\n            m,\n            *query_heads,\n            *key_value_heads,\n            layer,\n            head_dim,\n            has_query_key_norm);",
)
edit(
    "src/model/inference/rules_attention.cpp",
    "            make_attention(m, heads, 1, value_dim, false);",
    "            make_attention(m, heads, 1, layer, value_dim, false);",
)

edit(
    "src/model/inference/rules_attention.cpp",
    "        if (has_tensor(layer_prefix + \"attn_q_norm.weight\")) {\n            const auto* q_norm = find_unique(\n                input.inventory,\n                {layer_prefix + \"attn_q_norm.weight\"},",
    "        if (has_query_key_norm) {\n            auto q_norm_candidates = attention_tensor_candidates(layer, \"q_norm.weight\");\n            q_norm_candidates.push_back(\"model.layers.\" + index + \".self_attn.q_layernorm.weight\");\n            const auto* q_norm = find_unique(\n                input.inventory,\n                q_norm_candidates,",
)
edit(
    "src/model/inference/rules_attention.cpp",
    "            add_binding(\n                bindings,\n                TensorRole::AttentionQueryNorm,\n                layer,\n                *q_norm,\n                {});\n        } else if (has_tensor(q_norm_alias)) {\n            const auto* q_norm = find_unique(\n                input.inventory,\n                {q_norm_alias},\n                TensorRole::AttentionQueryNorm,\n                layer,\n                {layer_head_dim},\n                {});\n            add_binding(\n                bindings,\n                TensorRole::AttentionQueryNorm,\n                layer,\n                *q_norm,\n                {});\n        }\n\n        if (has_tensor(layer_prefix + \"attn_k_norm.weight\")) {\n            const auto* k_norm = find_unique(\n                input.inventory,\n                {layer_prefix + \"attn_k_norm.weight\"},",
    "            add_binding(\n                bindings,\n                TensorRole::AttentionQueryNorm,\n                layer,\n                *q_norm,\n                {});\n\n            auto k_norm_candidates = attention_tensor_candidates(layer, \"k_norm.weight\");\n            k_norm_candidates.push_back(\"model.layers.\" + index + \".self_attn.k_layernorm.weight\");\n            const auto* k_norm = find_unique(\n                input.inventory,\n                k_norm_candidates,",
)
edit(
    "src/model/inference/rules_attention.cpp",
    "            add_binding(\n                bindings,\n                TensorRole::AttentionKeyNorm,\n                layer,\n                *k_norm,\n                {});\n        } else if (has_tensor(k_norm_alias)) {\n            const auto* k_norm = find_unique(\n                input.inventory,\n                {k_norm_alias},\n                TensorRole::AttentionKeyNorm,\n                layer,\n                {layer_head_dim},\n                {});\n            add_binding(\n                bindings,\n                TensorRole::AttentionKeyNorm,\n                layer,\n                *k_norm,\n                {});\n        }",
    "            add_binding(\n                bindings,\n                TensorRole::AttentionKeyNorm,\n                layer,\n                *k_norm,\n                {});\n        }",
)
edit(
    "src/model/inference/rules_attention.cpp",
    "        const std::string q_norm_alias =\n            \"model.layers.\" + index + \".self_attn.q_layernorm.weight\";\n        const std::string k_norm_alias =\n            \"model.layers.\" + index + \".self_attn.k_layernorm.weight\";\n\n",
    "",
)

edit(
    "src/model/inference/layer_bindings.cpp",
    "void resolve_layer_feed_forward(CanonicalInferenceContext& context,\n                                int layer,\n                                const std::vector<std::string>& ffn_norm_candidates) {",
    "void resolve_layer_feed_forward(CanonicalInferenceContext& context, int layer) {",
)
edit(
    "src/model/inference/layer_bindings.cpp",
    "    const auto* ffn_norm = find_unique(\n        input.inventory,\n        ffn_norm_candidates,\n        TensorRole::FfnInputNorm,\n        layer,\n        {*m.core.hidden_size},\n        {});\n    add_binding(\n        facts.bindings,\n        TensorRole::FfnInputNorm,\n        layer,\n        *ffn_norm,\n        {});",
    "    const auto bind_norm = [&](TensorRole role, const std::optional<NormSpec>& spec) {\n        if (!spec || spec->weightless()) return;\n        const auto* tensor = find_unique(\n            input.inventory, norm_tensor_candidates(layer, role), role, layer,\n            {*m.core.hidden_size}, {});\n        add_binding(facts.bindings, role, layer, *tensor, {});\n    };\n    bind_norm(TensorRole::FfnInputNorm, semantic_layer.feed_forward_norm.before);\n    bind_norm(TensorRole::FfnOutputNorm, semantic_layer.feed_forward_norm.after);",
)

old_loop = r'''        const std::vector<std::string> norm_candidates = {
            "transformer.h." + index + ".ln_1.weight",
            "model.layers." + index + ".input_layernorm.weight",
            "model.layers." + index + ".self_attn_layer_norm.weight",
            "model.language_model.layers." + index + ".input_layernorm.weight",
            "model.language_model.layers." + index + ".operator_norm.weight",
            "model.layers." + index + ".operator_norm.weight",
            "backbone.layers." + index + ".norm.weight",
            "blk." + index + ".attn_norm.weight",
        };
        const std::vector<std::string> ffn_norm_candidates = {
            "transformer.h." + index + ".ln_2.weight",
            "model.layers." + index + ".post_attention_layernorm.weight",
            "model.language_model.layers." + index + ".post_attention_layernorm.weight",
            "model.language_model.layers." + index + ".ffn_norm.weight",
            "model.layers." + index + ".ffn_norm.weight",
            "blk." + index + ".ffn_norm.weight",
            "blk." + index + ".post_attention_norm.weight",
        };

        const auto* attention_norm = find_unique(
            input.inventory,
            norm_candidates,
            TensorRole::AttentionInputNorm,
            layer,
            {*m.core.hidden_size},
            {});
        add_binding(
            facts.bindings,
            TensorRole::AttentionInputNorm,
            layer,
            *attention_norm,
            {});

        const ILayerInferenceRule& rule =
            select_layer_inference_rule(rules, context, layer);
        rule.resolve(context, layer);
        resolve_layer_feed_forward(context, layer, ffn_norm_candidates);'''
new_loop = r'''        const LayerSpec& semantic_layer = facts.graph.layers[static_cast<size_t>(layer)];
        const auto bind_norm = [&](TensorRole role, const std::optional<NormSpec>& spec) {
            if (!spec || spec->weightless()) return;
            const auto* tensor = find_unique(
                input.inventory, norm_tensor_candidates(layer, role), role, layer,
                {*m.core.hidden_size}, {});
            add_binding(facts.bindings, role, layer, *tensor, {});
        };
        bind_norm(TensorRole::AttentionInputNorm, semantic_layer.mixer_norm.before);
        bind_norm(TensorRole::AttentionPostNorm, semantic_layer.mixer_norm.after);

        const ILayerInferenceRule& rule =
            select_layer_inference_rule(rules, context, layer);
        rule.resolve(context, layer);
        resolve_layer_feed_forward(context, layer);'''
edit("src/model/inference/layer_bindings.cpp", old_loop, new_loop)

edit(
    "tests/architecture_resolution_test.cpp",
    "            CELEG_TEST_CHECK(fixture_model.graph.layers[2].post_attention_norm.has_value());",
    "            CELEG_TEST_CHECK(fixture_model.graph.layers[2].mixer_norm.after.has_value());",
)

for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts or path == Path(__file__).resolve():
        continue
    if path.suffix not in {".cpp", ".cu", ".cuh", ".hpp", ".h"}:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace(".post_attention_norm", ".mixer_norm.after")
    text = text.replace(".post_feed_forward_norm", ".feed_forward_norm.after")
    text = text.replace(".pre_feed_forward_norm", ".feed_forward_norm.before")
    path.write_text(text, encoding="utf-8")

print("automatic semantic model refactor staged")
