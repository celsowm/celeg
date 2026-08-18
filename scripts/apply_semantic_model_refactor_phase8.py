from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"missing fragment in {path}: {old[:120]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_region(path: str, start: str, end: str, replacement: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    begin = text.find(start)
    finish = text.find(end, begin)
    if begin < 0 or finish < 0:
        raise RuntimeError(f"region not found in {path}")
    file.write_text(text[:begin] + replacement + text[finish:], encoding="utf-8")


attention_candidates = '''std::vector<std::string> attention_tensor_candidates(int layer,\n                                                      std::string_view suffix) {\n    const std::string index = std::to_string(layer);\n    std::string gguf_suffix;\n    if (suffix == "q_proj.weight") gguf_suffix = "attn_q.weight";\n    else if (suffix == "k_proj.weight") gguf_suffix = "attn_k.weight";\n    else if (suffix == "v_proj.weight") gguf_suffix = "attn_v.weight";\n    else if (suffix == "o_proj.weight") gguf_suffix = "attn_output.weight";\n    else if (suffix == "q_norm.weight") gguf_suffix = "attn_q_norm.weight";\n    else if (suffix == "k_norm.weight") gguf_suffix = "attn_k_norm.weight";\n    else gguf_suffix = std::string(suffix);\n    std::vector<std::string> result = {\n        "transformer.h." + index + ".attn." + std::string(suffix),\n        "model.language_model.layers." + index + ".self_attn." + std::string(suffix),\n        "model.layers." + index + ".self_attn." + std::string(suffix),\n        "model.layers." + index + ".attention." + std::string(suffix),\n        "layers." + index + ".attention." + std::string(suffix),\n        "blk." + index + "." + gguf_suffix,\n    };\n    if (suffix == "o_proj.weight") {\n        result.push_back("model.language_model.layers." + index + ".self_attn.out_proj.weight");\n        result.push_back("model.layers." + index + ".self_attn.out_proj.weight");\n    }\n    return result;\n}\n\n'''
replace_region(
    "src/model/inference/support.cpp",
    "std::vector<std::string> attention_tensor_candidates(int layer,",
    "std::vector<std::string> feed_forward_tensor_candidates(int layer,",
    attention_candidates,
)

replace_once(
    "tests/automatic_inference_test.cpp",
    '''    void add(std::string name, std::vector<int64_t> shape) {\n        shapes_.emplace(std::move(name), std::move(shape));\n    }''',
    '''    void add(std::string name, std::vector<int64_t> shape) {\n        shapes_.emplace(std::move(name), std::move(shape));\n    }\n\n    void remove(std::string_view name) {\n        shapes_.erase(std::string(name));\n    }''')

semantic_tests = r'''
    auto identity_a_metadata = metadata();
    identity_a_metadata.values["model_type"] = std::string("unknown_semantic_clone_a");
    celeg::CheckpointView identity_a_checkpoint;
    identity_a_checkpoint.metadata = identity_a_metadata;
    identity_a_checkpoint.repository = repository();
    const celeg::ResolvedModel identity_a =
        catalog.select(identity_a_checkpoint.metadata).resolve(identity_a_checkpoint);

    auto identity_b_metadata = metadata();
    identity_b_metadata.values["model_type"] = std::string("poisoned_unrelated_identity");
    celeg::CheckpointView identity_b_checkpoint;
    identity_b_checkpoint.metadata = identity_b_metadata;
    identity_b_checkpoint.repository = repository();
    const celeg::ResolvedModel identity_b =
        catalog.select(identity_b_checkpoint.metadata).resolve(identity_b_checkpoint);
    CELEG_TEST_CHECK(identity_a.provenance.identity == identity_b.provenance.identity);
    CELEG_TEST_CHECK(identity_a.weight_plan.requests.size() == identity_b.weight_plan.requests.size());
    for (size_t i = 0; i < identity_a.weight_plan.requests.size(); ++i) {
        const auto& left = identity_a.weight_plan.requests[i];
        const auto& right = identity_b.weight_plan.requests[i];
        CELEG_TEST_CHECK(left.role == right.role);
        CELEG_TEST_CHECK(left.layer == right.layer);
        CELEG_TEST_CHECK(left.shape == right.shape);
    }

    auto scheduled_attention = metadata();
    scheduled_attention.values["layer_types"] =
        std::vector<std::string>{"sliding_attention", "full_attention"};
    scheduled_attention.values["sliding_window"] = int64_t(7);
    celeg::CheckpointView scheduled_checkpoint;
    scheduled_checkpoint.metadata = scheduled_attention;
    scheduled_checkpoint.repository = repository();
    const celeg::ResolvedModel scheduled_model =
        catalog.select(scheduled_checkpoint.metadata).resolve(scheduled_checkpoint);
    CELEG_TEST_CHECK(std::get<celeg::AttentionSpec>(scheduled_model.graph.layers[0].mixer)
                         .sliding_window_size() == 7);
    CELEG_TEST_CHECK(std::get<celeg::AttentionSpec>(scheduled_model.graph.layers[1].mixer)
                         .sliding_window_size() == 0);

    auto truncated_schedule = metadata();
    truncated_schedule.values["layer_types"] =
        std::vector<std::string>{"full_attention"};
    bool truncated_schedule_rejected = false;
    try { (void)celeg::normalize_model_metadata(truncated_schedule); }
    catch (const celeg::ResolutionError& error) {
        truncated_schedule_rejected =
            error.kind() == celeg::ResolutionFailureKind::IncompleteLayerSchedule;
    }
    CELEG_TEST_CHECK(truncated_schedule_rejected);

    auto unknown_layer_type = metadata();
    unknown_layer_type.values["layer_types"] =
        std::vector<std::string>{"unknown_attention_math", "full_attention"};
    celeg::CheckpointView unknown_layer_checkpoint;
    unknown_layer_checkpoint.metadata = unknown_layer_type;
    unknown_layer_checkpoint.repository = repository();
    bool unknown_layer_rejected = false;
    try {
        (void)catalog.select(unknown_layer_checkpoint.metadata).resolve(unknown_layer_checkpoint);
    } catch (const celeg::ResolutionError& error) {
        unknown_layer_rejected =
            error.kind() == celeg::ResolutionFailureKind::UnsupportedSemanticFeature;
    }
    CELEG_TEST_CHECK(unknown_layer_rejected);

    auto sliding_without_window = metadata();
    sliding_without_window.values["layer_types"] =
        std::vector<std::string>{"sliding_attention", "full_attention"};
    celeg::CheckpointView sliding_without_window_checkpoint;
    sliding_without_window_checkpoint.metadata = sliding_without_window;
    sliding_without_window_checkpoint.repository = repository();
    bool missing_window_rejected = false;
    try {
        (void)catalog.select(sliding_without_window_checkpoint.metadata)
            .resolve(sliding_without_window_checkpoint);
    } catch (const celeg::ResolutionError& error) {
        missing_window_rejected =
            error.kind() == celeg::ResolutionFailureKind::UnsupportedSemanticFeature;
    }
    CELEG_TEST_CHECK(missing_window_rejected);

    auto postnorm_metadata = metadata();
    postnorm_metadata.values["layer_layouts"] =
        std::vector<std::string>{"postnorm", "postnorm"};
    auto postnorm_repository = repository();
    for (int layer = 0; layer < 2; ++layer) {
        const std::string transformer = "transformer.h." + std::to_string(layer);
        postnorm_repository->remove(transformer + ".ln_1.weight");
        postnorm_repository->remove(transformer + ".ln_2.weight");
        const std::string model_layer = "model.layers." + std::to_string(layer);
        postnorm_repository->add(model_layer + ".post_attn_norm.weight", {8});
        postnorm_repository->add(model_layer + ".post_mlp_norm.weight", {8});
    }
    celeg::CheckpointView postnorm_checkpoint;
    postnorm_checkpoint.metadata = postnorm_metadata;
    postnorm_checkpoint.repository = postnorm_repository;
    const celeg::ResolvedModel postnorm_model =
        catalog.select(postnorm_checkpoint.metadata).resolve(postnorm_checkpoint);
    for (const auto& layer : postnorm_model.graph.layers) {
        CELEG_TEST_CHECK(!layer.mixer_norm.before.has_value());
        CELEG_TEST_CHECK(layer.mixer_norm.after.has_value());
        CELEG_TEST_CHECK(!layer.feed_forward_norm.before.has_value());
        CELEG_TEST_CHECK(layer.feed_forward_norm.after.has_value());
    }

    auto contradictory_norm = metadata();
    contradictory_norm.values["use_pre_attn_norm"] = false;
    celeg::CheckpointView contradictory_norm_checkpoint;
    contradictory_norm_checkpoint.metadata = contradictory_norm;
    contradictory_norm_checkpoint.repository = repository();
    bool contradictory_norm_rejected = false;
    try {
        (void)catalog.select(contradictory_norm_checkpoint.metadata)
            .resolve(contradictory_norm_checkpoint);
    } catch (const celeg::ResolutionError& error) {
        contradictory_norm_rejected =
            error.kind() == celeg::ResolutionFailureKind::ConflictingInferenceFacts;
    }
    CELEG_TEST_CHECK(contradictory_norm_rejected);

    auto required_norm_metadata = metadata();
    required_norm_metadata.values["use_pre_attn_norm"] = true;
    auto missing_norm_repository = repository();
    missing_norm_repository->remove("transformer.h.0.ln_1.weight");
    celeg::CheckpointView missing_norm_checkpoint;
    missing_norm_checkpoint.metadata = required_norm_metadata;
    missing_norm_checkpoint.repository = missing_norm_repository;
    bool missing_norm_rejected = false;
    try {
        (void)catalog.select(missing_norm_checkpoint.metadata).resolve(missing_norm_checkpoint);
    } catch (const celeg::ResolutionError& error) {
        missing_norm_rejected =
            error.kind() == celeg::ResolutionFailureKind::MissingTensorRole;
    }
    CELEG_TEST_CHECK(missing_norm_rejected);

    auto qk_metadata = metadata();
    qk_metadata.values["use_qk_norm"] = true;
    auto qk_repository = repository();
    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = "transformer.h." + std::to_string(layer) + ".attn.";
        qk_repository->add(prefix + "q_norm.weight", {2});
        qk_repository->add(prefix + "k_norm.weight", {2});
    }
    celeg::CheckpointView qk_checkpoint;
    qk_checkpoint.metadata = qk_metadata;
    qk_checkpoint.repository = qk_repository;
    const celeg::ResolvedModel qk_model =
        catalog.select(qk_checkpoint.metadata).resolve(qk_checkpoint);
    CELEG_TEST_CHECK(std::get<celeg::AttentionSpec>(qk_model.graph.layers[0].mixer)
                         .query_norm.has_value());
    CELEG_TEST_CHECK(std::get<celeg::AttentionSpec>(qk_model.graph.layers[0].mixer)
                         .key_norm.has_value());

    auto qk_false_metadata = metadata();
    qk_false_metadata.values["use_qk_norm"] = false;
    celeg::CheckpointView qk_false_checkpoint;
    qk_false_checkpoint.metadata = qk_false_metadata;
    qk_false_checkpoint.repository = qk_repository;
    bool qk_false_rejected = false;
    try {
        (void)catalog.select(qk_false_checkpoint.metadata).resolve(qk_false_checkpoint);
    } catch (const celeg::ResolutionError& error) {
        qk_false_rejected =
            error.kind() == celeg::ResolutionFailureKind::ConflictingInferenceFacts;
    }
    CELEG_TEST_CHECK(qk_false_rejected);

    auto yarn_metadata = metadata();
    yarn_metadata.values["rope_scaling.rope_type"] = std::string("yarn");
    yarn_metadata.values["rope_scaling.factor"] = 4.0;
    celeg::CheckpointView yarn_checkpoint;
    yarn_checkpoint.metadata = yarn_metadata;
    yarn_checkpoint.repository = repository();
    const celeg::ResolvedModel yarn_model =
        catalog.select(yarn_checkpoint.metadata).resolve(yarn_checkpoint);
    const auto* yarn_rope = std::get<celeg::AttentionSpec>(yarn_model.graph.layers[0].mixer)
                                .rope_position();
    CELEG_TEST_CHECK(yarn_rope != nullptr);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::YarnRopeScaling>(yarn_rope->scaling));
'''

replace_once(
    "tests/automatic_inference_test.cpp",
    "    CELEG_TEST_CHECK(celeg::explain_resolution(ling_checkpoint).failures.empty());\n    return 0;",
    "    CELEG_TEST_CHECK(celeg::explain_resolution(ling_checkpoint).failures.empty());\n" +
        semantic_tests + "\n    return 0;",
)

print("semantic inference regression coverage staged")
