from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src/model/inference/metadata.cpp"
text = path.read_text(encoding="utf-8")

old = '''    result.core.norm_epsilon = aliases<float>(
        metadata, {"rms_norm_eps", "rms_norm_epsilon", "layer_norm_epsilon"},
        result.evidence, "norm_epsilon", "attention.layer_norm_rms_epsilon");'''
new = '''    result.core.norm_epsilon = aliases<float>(
        metadata, {"rms_norm_eps", "rms_norm_epsilon", "layer_norm_epsilon", "norm_eps"},
        result.evidence, "norm_epsilon", "attention.layer_norm_rms_epsilon");'''
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise RuntimeError("norm epsilon alias anchor not found")

anchor = '''NormalizedModelMetadata normalize_model_metadata(const CheckpointMetadata& metadata) {
    reject_unknown_semantic_metadata(metadata);
    NormalizedModelMetadata result;'''
replacement = '''NormalizedModelMetadata normalize_model_metadata(const CheckpointMetadata& metadata) {
    reject_unknown_semantic_metadata(metadata);
    const auto require_supported_rms_norm = [&](std::string_view key) {
        for (const std::string& candidate : {std::string(key), "text_config." + std::string(key)}) {
            if (!metadata.contains(candidate)) continue;
            const MetadataValue& value = metadata.value(candidate);
            const auto* name = std::get_if<std::string>(&value);
            if (name == nullptr) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "normalization metadata has an incompatible type: " + candidate);
            }
            if (*name != "rmsnorm" && *name != "rms_norm" && *name != "rms") {
                inference_detail::fail(
                    ResolutionFailureKind::UnsupportedSemanticFeature,
                    "automatic resolution only represents RMS normalization for metadata key: " +
                        candidate);
            }
        }
    };
    require_supported_rms_norm("norm_type");
    require_supported_rms_norm("qk_norm_type");
    NormalizedModelMetadata result;'''
if anchor in text:
    text = text.replace(anchor, replacement, 1)
elif replacement not in text:
    raise RuntimeError("normalization type validation anchor not found")

path.write_text(text, encoding="utf-8")
print("generic RMSNorm metadata aliases normalized")
