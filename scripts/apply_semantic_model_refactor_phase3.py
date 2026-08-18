from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if old in text:
        file.write_text(text.replace(old, new), encoding="utf-8")
        return
    if new in text:
        return
    raise RuntimeError(f"missing fragment in {path}: {old[:120]!r}")


def replace_region(path: str, start: str, end: str, replacement: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    begin = text.find(start)
    finish = text.find(end, begin)
    if begin < 0 or finish < 0:
        raise RuntimeError(f"region not found in {path}")
    file.write_text(text[:begin] + replacement + text[finish:], encoding="utf-8")


replace(
    "src/model/descriptor/architecture.cpp",
    "            semantic_layer.operator_norm = {numerical_policy.norm_eps,\n                                             descriptor_.operator_norm_kind};\n            semantic_layer.feed_forward_norm = NormSpec{numerical_policy.norm_eps,\n                                                 descriptor_.feed_forward_norm_kind};",
    "            semantic_layer.mixer_norm.before = NormSpec{\n                numerical_policy.norm_eps, descriptor_.operator_norm_kind};\n            semantic_layer.feed_forward_norm.before = NormSpec{\n                numerical_policy.norm_eps, descriptor_.feed_forward_norm_kind};",
)

support_declarations = '''std::vector<std::string> norm_tensor_candidates(int layer, TensorRole role);\nbool has_any_tensor(const TensorInventory& inventory,\n                    const std::vector<std::string>& candidates);\n\n'''
replace_region(
    "src/model/inference/support.hpp",
    "std::vector<std::string> norm_tensor_candidates(int layer, TensorRole role);",
    "const TensorInventoryEntry* find_mamba_tensor",
    support_declarations,
)

support_implementations = '''std::vector<std::string> norm_tensor_candidates(int layer, TensorRole role) {\n    const std::string index = std::to_string(layer);\n    switch (role) {\n    case TensorRole::AttentionInputNorm:\n        return {\n            "transformer.h." + index + ".ln_1.weight",\n            "model.layers." + index + ".input_layernorm.weight",\n            "model.layers." + index + ".self_attn_layer_norm.weight",\n            "model.language_model.layers." + index + ".input_layernorm.weight",\n            "model.language_model.layers." + index + ".operator_norm.weight",\n            "model.layers." + index + ".operator_norm.weight",\n            "model.layers." + index + ".pre_attn_norm.weight",\n            "model.language_model.layers." + index + ".pre_attn_norm.weight",\n            "backbone.layers." + index + ".norm.weight",\n            "blk." + index + ".attn_norm.weight",\n        };\n    case TensorRole::AttentionPostNorm:\n        return {\n            "model.layers." + index + ".post_attn_norm.weight",\n            "model.language_model.layers." + index + ".post_attn_norm.weight",\n            "model.layers." + index + ".post_attention_norm.weight",\n            "model.language_model.layers." + index + ".post_attention_norm.weight",\n            "blk." + index + ".post_attention_norm.weight",\n        };\n    case TensorRole::FfnInputNorm:\n        return {\n            "transformer.h." + index + ".ln_2.weight",\n            "model.layers." + index + ".post_attention_layernorm.weight",\n            "model.language_model.layers." + index + ".post_attention_layernorm.weight",\n            "model.layers." + index + ".pre_feedforward_layernorm.weight",\n            "model.language_model.layers." + index + ".pre_feedforward_layernorm.weight",\n            "model.language_model.layers." + index + ".ffn_norm.weight",\n            "model.layers." + index + ".ffn_norm.weight",\n            "model.layers." + index + ".pre_mlp_norm.weight",\n            "model.language_model.layers." + index + ".pre_mlp_norm.weight",\n            "blk." + index + ".ffn_norm.weight",\n        };\n    case TensorRole::FfnOutputNorm:\n        return {\n            "model.layers." + index + ".post_mlp_norm.weight",\n            "model.language_model.layers." + index + ".post_mlp_norm.weight",\n            "model.layers." + index + ".post_feedforward_layernorm.weight",\n            "model.language_model.layers." + index + ".post_feedforward_layernorm.weight",\n            "model.layers." + index + ".post_feed_forward_norm.weight",\n            "model.language_model.layers." + index + ".post_feed_forward_norm.weight",\n            "blk." + index + ".post_ffw_norm.weight",\n        };\n    default:\n        throw std::invalid_argument("tensor role is not a sublayer norm role");\n    }\n}\n\nbool has_any_tensor(const TensorInventory& inventory,\n                    const std::vector<std::string>& candidates) {\n    for (const std::string& candidate : candidates) {\n        if (inventory.find(candidate) != nullptr) return true;\n    }\n    return false;\n}\n\n'''
replace_region(
    "src/model/inference/support.cpp",
    "std::vector<std::string> norm_tensor_candidates(int layer, TensorRole role) {",
    "void add_binding(TensorRoleBindings& bindings",
    support_implementations,
)

print("semantic inference helper cleanup staged")
