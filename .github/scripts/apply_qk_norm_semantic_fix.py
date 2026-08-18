from pathlib import Path

metadata_path = Path("src/model/inference/metadata.cpp")
metadata = metadata_path.read_text(encoding="utf-8")
anchor = '''    result.attention.query_key_norm = aliases<bool>(
        metadata, {"qk_norm", "query_key_norm", "use_qk_norm"}, result.evidence,
        "query_key_norm");
'''
validation = '''    if (result.attention.query_key_norm.value_or(false)) {
        std::optional<std::string> qk_norm_type;
        std::string qk_norm_type_source;
        for (const std::string_view key : {
                 std::string_view("qk_norm_type"),
                 std::string_view("text_config.qk_norm_type")}) {
            if (!metadata.contains(key)) continue;
            const MetadataValue& raw = metadata.value(key);
            const auto* value = std::get_if<std::string>(&raw);
            if (value == nullptr) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "Q/K normalization type metadata is not a string: " +
                        std::string(key));
            }
            if (qk_norm_type.has_value() && *qk_norm_type != *value) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "conflicting Q/K normalization type metadata");
            }
            qk_norm_type = *value;
            qk_norm_type_source = std::string(key);
        }
        if (qk_norm_type.has_value()) {
            if (*qk_norm_type != "rmsnorm") {
                inference_detail::fail(
                    ResolutionFailureKind::UnsupportedSemanticFeature,
                    "unsupported Q/K normalization mathematics: " + *qk_norm_type);
            }
            result.evidence.push_back({EvidenceKind::ExplicitMetadata,
                                       qk_norm_type_source,
                                       "query_key_norm = rmsnorm"});
        }
    }
'''
if "unsupported Q/K normalization mathematics" not in metadata:
    if metadata.count(anchor) != 1:
        raise SystemExit("query_key_norm assignment shape changed")
    metadata = metadata.replace(anchor, anchor + validation)
    metadata_path.write_text(metadata, encoding="utf-8")

cmake_path = Path("CMakeLists.txt")
cmake = cmake_path.read_text(encoding="utf-8")
cmake_anchor = '''    add_executable(checkpoint_capabilities_test tests/checkpoint_capabilities_test.cpp)
    target_include_directories(checkpoint_capabilities_test PRIVATE include tests)
    add_test(NAME checkpoint_capabilities_test COMMAND checkpoint_capabilities_test)
'''
cmake_fixed = '''    add_executable(checkpoint_capabilities_test tests/checkpoint_capabilities_test.cpp)
    target_include_directories(checkpoint_capabilities_test PRIVATE include tests)
    target_link_libraries(checkpoint_capabilities_test PRIVATE celeg_base)
    add_test(NAME checkpoint_capabilities_test COMMAND checkpoint_capabilities_test)
'''
if "target_link_libraries(checkpoint_capabilities_test PRIVATE celeg_base)" not in cmake:
    if cmake.count(cmake_anchor) != 1:
        raise SystemExit("checkpoint_capabilities_test CMake shape changed")
    cmake = cmake.replace(cmake_anchor, cmake_fixed)
    cmake_path.write_text(cmake, encoding="utf-8")
