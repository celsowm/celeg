from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_region(path: str, start: str, end: str, replacement: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    begin = text.find(start)
    finish = text.find(end, begin)
    if begin < 0 or finish < 0:
        raise RuntimeError(f"region not found in {path}: {start!r} -> {end!r}")
    file.write_text(text[:begin] + replacement + text[finish:], encoding="utf-8")


scoped_string = '''LayerScopedValue<std::string> scoped_string_aliases(\n    const CheckpointMetadata& metadata,\n    std::initializer_list<std::string_view> keys,\n    std::vector<EvidenceItem>& evidence,\n    std::string_view fact) {\n    LayerScopedValue<std::string> result;\n    std::string source;\n    const auto consider_scalar = [&](const std::string& value, std::string_view key) {\n        if (result.global && *result.global != value) {\n            inference_detail::fail(\n                ResolutionFailureKind::ConflictingMetadata,\n                "conflicting metadata aliases for " + std::string(fact));\n        }\n        if (!result.per_layer.empty()) {\n            for (const auto& layer_value : result.per_layer) {\n                if (layer_value && *layer_value != value) {\n                    inference_detail::fail(\n                        ResolutionFailureKind::ConflictingMetadata,\n                        "conflicting global and layer-scoped metadata for " +\n                            std::string(fact));\n                }\n            }\n        }\n        result.global = value;\n        source = key;\n    };\n    const auto consider_vector = [&](const std::vector<std::string>& values,\n                                     std::string_view key) {\n        std::vector<std::optional<std::string>> schedule;\n        schedule.reserve(values.size());\n        for (const std::string& value : values) schedule.push_back(value);\n        if (!result.per_layer.empty() && result.per_layer != schedule) {\n            inference_detail::fail(\n                ResolutionFailureKind::ConflictingMetadata,\n                "conflicting layer-scoped metadata for " + std::string(fact));\n        }\n        if (result.global) {\n            for (const auto& layer_value : schedule) {\n                if (layer_value && *layer_value != *result.global) {\n                    inference_detail::fail(\n                        ResolutionFailureKind::ConflictingMetadata,\n                        "conflicting global and layer-scoped metadata for " +\n                            std::string(fact));\n                }\n            }\n        }\n        result.per_layer = std::move(schedule);\n        source = key;\n    };\n    const auto consider = [&](std::string_view key) {\n        if (!metadata.contains(key)) return;\n        const MetadataValue& value = metadata.value(key);\n        if (const auto* scalar_value = std::get_if<std::string>(&value)) {\n            consider_scalar(*scalar_value, key);\n            return;\n        }\n        if (const auto* values = std::get_if<std::vector<std::string>>(&value)) {\n            consider_vector(*values, key);\n            return;\n        }\n        inference_detail::fail(\n            ResolutionFailureKind::ConflictingMetadata,\n            "metadata key has an incompatible type: " + std::string(key));\n    };\n    for (const std::string_view key : keys) {\n        consider(key);\n        consider("text_config." + std::string(key));\n    }\n    if (result.has_value()) {\n        evidence.push_back({EvidenceKind::AliasMetadata, source,\n                            std::string(fact) + (result.per_layer.empty()\n                                ? " = " + *result.global\n                                : " = layer-scoped schedule")});\n    }\n    return result;\n}\n\n'''
replace_region(
    "src/model/inference/metadata.cpp",
    "LayerScopedValue<std::string> scoped_string_aliases(",
    "template <typename T>\nstd::optional<T> aliases",
    scoped_string,
)

print("semantic metadata precedence hardened")
