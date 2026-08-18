from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if old in text:
        file.write_text(text.replace(old, new, 1), encoding="utf-8")
        return
    if new in text:
        return
    raise RuntimeError(f"missing fragment in {path}: {old[:120]!r}")


replace(
    "src/model/inference/metadata.cpp",
    '        "qk_norm", "query_key_norm", "use_qk_norm", "xsa_projection",',
    '        "qk_norm", "query_key_norm", "use_qk_norm", "qk_norm_type", "xsa_projection",',
)

replace(
    "src/model/inference.cpp",
    '''        out << ':' << static_cast<int>(binding.role) << ':' << binding.layer << ':'
            << binding.expert << ':' << binding.source_name;''',
    '''        out << ':' << static_cast<int>(binding.role) << ':' << binding.layer << ':'
            << binding.expert;''',
)

replace(
    "src/model/weight_plan.cpp",
    '''        if (attention.query_norm.has_value()) {
            append(TensorRole::AttentionQueryNorm, {attention.head_dim});
        }''',
    '''        if (attention.query_norm.has_value() && !attention.query_norm->weightless()) {
            append(TensorRole::AttentionQueryNorm, {attention.head_dim});
        }''',
)
replace(
    "src/model/weight_plan.cpp",
    '''            if (attention.key_norm.has_value()) {
                append(TensorRole::AttentionKeyNorm, {attention.head_dim});
            }''',
    '''            if (attention.key_norm.has_value() && !attention.key_norm->weightless()) {
                append(TensorRole::AttentionKeyNorm, {attention.head_dim});
            }''',
)

replace(
    "tests/automatic_inference_test.cpp",
    '''    CELEG_TEST_CHECK(identity_a.provenance.identity == identity_b.provenance.identity);
    CELEG_TEST_CHECK(identity_a.weight_plan.requests.size() == identity_b.weight_plan.requests.size());''',
    '''    CELEG_TEST_CHECK(identity_a.graph.fingerprint() == identity_b.graph.fingerprint());
    CELEG_TEST_CHECK(identity_a.provenance.identity == identity_b.provenance.identity);
    CELEG_TEST_CHECK(identity_a.weight_plan.requests.size() == identity_b.weight_plan.requests.size());''',
)

replace(
    "tests/automatic_inference_test.cpp",
    '''    CELEG_TEST_CHECK(qk_false_rejected);

    auto yarn_metadata = metadata();''',
    '''    CELEG_TEST_CHECK(qk_false_rejected);

    auto norm_eps_metadata = metadata();
    norm_eps_metadata.values["norm_eps"] = 1.0e-6;
    norm_eps_metadata.values["norm_type"] = std::string("rmsnorm");
    norm_eps_metadata.values["qk_norm_type"] = std::string("rmsnorm");
    const auto normalized_norm_eps = celeg::normalize_model_metadata(norm_eps_metadata);
    CELEG_TEST_CHECK(normalized_norm_eps.core.norm_epsilon == std::optional<float>{1.0e-6f});

    auto sandwich_metadata = metadata();
    sandwich_metadata.values["layer_layouts"] =
        std::vector<std::string>{"sandwich", "sandwich"};
    auto sandwich_repository = repository();
    for (int layer = 0; layer < 2; ++layer) {
        const std::string model_layer = "model.layers." + std::to_string(layer);
        sandwich_repository->add(model_layer + ".post_attn_norm.weight", {8});
        sandwich_repository->add(model_layer + ".post_mlp_norm.weight", {8});
    }
    celeg::CheckpointView sandwich_checkpoint;
    sandwich_checkpoint.metadata = sandwich_metadata;
    sandwich_checkpoint.repository = sandwich_repository;
    const celeg::ResolvedModel sandwich_model =
        catalog.select(sandwich_checkpoint.metadata).resolve(sandwich_checkpoint);
    for (const auto& layer : sandwich_model.graph.layers) {
        CELEG_TEST_CHECK(layer.mixer_norm.before.has_value());
        CELEG_TEST_CHECK(layer.mixer_norm.after.has_value());
        CELEG_TEST_CHECK(layer.feed_forward_norm.before.has_value());
        CELEG_TEST_CHECK(layer.feed_forward_norm.after.has_value());
    }

    auto yarn_metadata = metadata();''',
)

path = ROOT / "tests/automatic_inference_test.cpp"
text = path.read_text(encoding="utf-8")
text = text.replace(
    '    qk_metadata.values["use_qk_norm"] = true;\n',
    '    qk_metadata.values["use_qk_norm"] = true;\n    qk_metadata.values["qk_norm_type"] = std::string("rmsnorm");\n',
    1,
)
path.write_text(text, encoding="utf-8")

print("semantic identity and norm topology coverage hardened")
