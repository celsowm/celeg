from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_optional(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if old in text:
        file.write_text(text.replace(old, new), encoding="utf-8")


def replace_region(path: str, start: str, end: str, replacement: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    begin = text.find(start)
    finish = text.find(end, begin)
    if begin < 0 or finish < 0:
        raise RuntimeError(f"region not found in {path}: {start!r} -> {end!r}")
    file.write_text(text[:begin] + replacement + text[finish:], encoding="utf-8")


replace_optional(
    "src/model/inference/rules_attention.cpp",
    "    int layer,\n    int layer,\n    int layer,\n",
    "    int layer,\n",
)
replace_optional(
    "src/model/inference/rules_attention.cpp",
    "    int layer,\n    int layer,\n",
    "    int layer,\n",
)

scoped_facts = '''    result.attention.layer_type = scoped_string_aliases(\n        metadata, {"layer_types", "attention_types"}, result.evidence, "layer_type");\n    result.attention.sliding_window = scoped_aliases<int>(\n        metadata, {"sliding_window", "sliding_window_size"}, result.evidence,\n        "sliding_window", "attention.sliding_window");\n    result.norms.mixer_before = scoped_aliases<bool>(\n        metadata, {"use_pre_attn_norm", "use_pre_attention_norm"}, result.evidence,\n        "mixer_norm.before");\n    result.norms.mixer_after = scoped_aliases<bool>(\n        metadata, {"use_post_attn_norm", "use_post_attention_norm"}, result.evidence,\n        "mixer_norm.after");\n    result.norms.feed_forward_before = scoped_aliases<bool>(\n        metadata, {"use_pre_mlp_norm", "use_pre_ffn_norm"}, result.evidence,\n        "feed_forward_norm.before");\n    result.norms.feed_forward_after = scoped_aliases<bool>(\n        metadata, {"use_post_mlp_norm", "use_post_ffn_norm"}, result.evidence,\n        "feed_forward_norm.after");\n    result.norms.layer_layout = scoped_string_aliases(\n        metadata, {"layer_layouts"}, result.evidence, "layer_layout");\n'''
replace_region(
    "src/model/inference/metadata.cpp",
    "    result.attention.layer_type = scoped_string_aliases(",
    "    result.mamba2.intermediate = aliases<int>(",
    scoped_facts,
)

scoped_validation = '''    validate_scoped_alias(result.attention.query_heads, result.core.layer_count, "query_heads");\n    validate_scoped_alias(result.attention.key_value_heads, result.core.layer_count, "key_value_heads");\n    validate_scoped_alias(result.attention.head_dim, result.core.layer_count, "head_dim");\n    validate_scoped_alias(result.attention.layer_type, result.core.layer_count, "layer_type");\n    validate_scoped_alias(result.attention.sliding_window, result.core.layer_count, "sliding_window");\n    validate_scoped_alias(result.norms.mixer_before, result.core.layer_count, "mixer_norm.before");\n    validate_scoped_alias(result.norms.mixer_after, result.core.layer_count, "mixer_norm.after");\n    validate_scoped_alias(result.norms.feed_forward_before, result.core.layer_count, "feed_forward_norm.before");\n    validate_scoped_alias(result.norms.feed_forward_after, result.core.layer_count, "feed_forward_norm.after");\n    validate_scoped_alias(result.norms.layer_layout, result.core.layer_count, "layer_layout");\n\n'''
replace_region(
    "src/model/inference/metadata.cpp",
    "    validate_scoped_alias(result.attention.query_heads, result.core.layer_count, \"query_heads\");",
    "    const std::vector<int> eos = token_list(metadata, \"eos_token_id\");",
    scoped_validation,
)

print("semantic schedule deduplication staged")
