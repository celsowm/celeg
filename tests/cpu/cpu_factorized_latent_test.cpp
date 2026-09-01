#include "celeg/backend/cpu/model.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/runtime/context.hpp"
#include "cpu/support/synthetic_checkpoint.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

class FactorizedLatentArchitecture final : public celeg::IArchitecture {
public:
    explicit FactorizedLatentArchitecture(bool gated) : gated_(gated) {}

    std::string_view id() const override {
        return gated_ ? "cpu_factorized_latent_gated_fixture"
                      : "cpu_factorized_latent_fixture";
    }

    celeg::ProbeResult probe(const celeg::CheckpointMetadata&) const override {
        return {true, 1000000, "CPU factorized-latent execution fixture"};
    }

    celeg::ResolvedModel resolve(const celeg::CheckpointView&) const override {
        celeg::ResolvedModel model;
        model.provenance.architecture_id = std::string(id());
        model.provenance.identity = std::string(id());
        model.provenance.source_format = "safetensors";
        model.graph.hidden = 4;
        model.graph.final_norm = celeg::NormSpec{
            1.0e-5f, celeg::NormWeightKind::None,
            celeg::NormGranularity::WholeVector};
        model.graph.tied_embeddings = true;
        model.graph.layers.resize(1);

        celeg::FactorizedLatentProjection projection;
        projection.query_rank = 2;
        projection.value_head_dim = 2;
        projection.query_latent_norm = celeg::NormSpec{
            1.0e-5f, celeg::NormWeightKind::Scale,
            celeg::NormGranularity::WholeVector};
        projection.key_latent_norm = celeg::NormSpec{
            1.0e-5f, celeg::NormWeightKind::Scale,
            celeg::NormGranularity::WholeVector};

        celeg::LatentAttentionStateSpec latent;
        latent.latent_rank = 2;
        latent.rope_head_dim = 0;
        latent.nope_head_dim = 2;
        latent.decoupled_rope = false;
        latent.projection = projection;

        celeg::AttentionSpec attention;
        attention.query_heads = 2;
        attention.key_value_heads = 1;
        attention.head_dim = 2;
        attention.query_scale = 1.0f / std::sqrt(2.0f);
        attention.position = celeg::NoPositionEncodingSpec{};
        attention.pattern = celeg::FullCausalPattern{};
        attention.state = latent;
        if (gated_) {
            attention.output_gate = celeg::SigmoidAttentionGateSpec{
                false, celeg::AttentionGateGranularity::HeadWise};
        }
        model.graph.layers[0].mixer = attention;
        model.graph.layers[0].feed_forward = std::monostate{};

        celeg::CheckpointDimensions dimensions;
        dimensions.vocab_size = 8;
        dimensions.max_position_embeddings = 32;
        dimensions.token_policy.bos_token_id = 0;
        dimensions.token_policy.eos_token_ids = {7};
        dimensions.token_policy.pad_token_id = 0;
        model.topology = celeg::compose_runtime_topology(
            std::move(dimensions), model.graph);

        using celeg::test_support::tensor_request;
        model.weight_plan.requests = {
            tensor_request(celeg::TensorRole::TokenEmbedding, -1,
                           "embed", {8, 4}),
            tensor_request(celeg::TensorRole::AttentionLatentQueryProjection, 0,
                           "latent.q_proj", {2, 4}),
            tensor_request(celeg::TensorRole::AttentionLatentQueryExpansion, 0,
                           "latent.q_expand", {4, 2}),
            tensor_request(celeg::TensorRole::AttentionLatentQueryNorm, 0,
                           "latent.q_norm", {2}),
            tensor_request(celeg::TensorRole::AttentionLatentKeyProjection, 0,
                           "latent.k_proj", {2, 4}),
            tensor_request(celeg::TensorRole::AttentionLatentKeyNorm, 0,
                           "latent.k_norm", {2}),
            tensor_request(celeg::TensorRole::AttentionLatentExpansion, 0,
                           "latent.expand", {8, 2}),
            tensor_request(celeg::TensorRole::AttentionLatentOutput, 0,
                           "latent.out", {4, 4}),
        };
        if (gated_) {
            model.weight_plan.requests.push_back(
                tensor_request(celeg::TensorRole::AttentionGate, 0,
                               "latent.gate", {2, 4}));
        }
        return model;
    }

private:
    bool gated_ = false;
};

std::shared_ptr<const celeg::RuntimeContext> fixture_runtime(bool gated) {
    celeg::RuntimeBuilder builder;
    builder.add_builtins();
    builder.add_architecture(
        std::make_unique<FactorizedLatentArchitecture>(gated));
    return builder.build_shared();
}

void compare_logits(const std::vector<float>& expected,
                    const std::vector<float>& actual) {
    CELEG_TEST_CHECK(expected.size() == actual.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        CELEG_TEST_CHECK(std::abs(expected[index] - actual[index]) < 1.0e-4f);
    }
}

void exercise_variant(const std::filesystem::path& directory, bool gated) {
    celeg::CpuModelOptions options;
    options.use_pack_cache = false;
    options.threads = 1;
    options.prefill_chunk_tokens = 2;
    options.prefill_chunk_threshold = 1;

    celeg::GenerationConfig generation;
    generation.seed = gated ? 29 : 23;
    generation.top_k = 1;

    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    const auto runtime = fixture_runtime(gated);

    celeg::CpuModel scalar(
        directory.string(), 32, options, generation, runtime);
    celeg::CpuModel chunked(
        directory.string(), 32, options, generation, runtime);

    scalar.session().prefill(prompt);
    const celeg::CpuBatchMetrics chunk_metrics = celeg::CpuModel::prefill_chunk(
        chunked, std::span<const int32_t>(prompt.data(), prompt.size()), true);
    CELEG_TEST_CHECK(chunk_metrics.batch_size == prompt.size());
    CELEG_TEST_CHECK(chunked.session().ready_for_decode());
    compare_logits(scalar.diagnostics().copy_logits(),
                   chunked.diagnostics().copy_logits());
    CELEG_TEST_CHECK(scalar.session().position() == chunked.session().position());
    const std::vector<float> prefill_logits = scalar.diagnostics().copy_logits();

    const int32_t scalar_token = scalar.session().decode();
    const int32_t chunked_token = chunked.session().decode();
    CELEG_TEST_CHECK(scalar_token == chunked_token);
    compare_logits(scalar.diagnostics().copy_logits(),
                   chunked.diagnostics().copy_logits());

    celeg::CpuModel packed_owner(
        directory.string(), 32, options, generation, runtime);
    std::unique_ptr<celeg::CpuModel> packed_a = packed_owner.clone_session();
    std::unique_ptr<celeg::CpuModel> packed_b = packed_owner.clone_session();

    for (std::size_t index = 0; index < prompt.size(); ++index) {
        const bool final = index + 1 == prompt.size();
        const celeg::CpuPrefillItem items[] = {
            {packed_a.get(), prompt[index], final},
            {packed_b.get(), prompt[index], final},
        };
        const celeg::CpuBatchMetrics metrics = celeg::CpuModel::prefill_batch(items);
        CELEG_TEST_CHECK(metrics.batch_size == 2);
    }

    CELEG_TEST_CHECK(packed_a->session().ready_for_decode());
    CELEG_TEST_CHECK(packed_b->session().ready_for_decode());
    compare_logits(prefill_logits, packed_a->diagnostics().copy_logits());
    compare_logits(prefill_logits, packed_b->diagnostics().copy_logits());

    celeg::CpuModel packed_reference(
        directory.string(), 32, options, generation, runtime);
    packed_reference.session().prefill(prompt);
    compare_logits(prefill_logits, packed_reference.diagnostics().copy_logits());
    const int32_t reference_token = packed_reference.session().decode();

    celeg::CpuModel* packed_models[] = {packed_a.get(), packed_b.get()};
    const auto [tokens, decode_metrics] = celeg::CpuModel::decode_batch(packed_models);
    CELEG_TEST_CHECK(decode_metrics.batch_size == 2);
    CELEG_TEST_CHECK(tokens.size() == 2);
    CELEG_TEST_CHECK(tokens[0] == reference_token);
    CELEG_TEST_CHECK(tokens[1] == reference_token);
    compare_logits(packed_reference.diagnostics().copy_logits(),
                   packed_a->diagnostics().copy_logits());
    compare_logits(packed_reference.diagnostics().copy_logits(),
                   packed_b->diagnostics().copy_logits());
}

}

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        "celeg-cpu-factorized-latent-test";

    celeg::test_support::write_safetensors_checkpoint(
        directory, "celeg_cpu_factorized_latent_fixture", {
            celeg::test_support::pattern_tensor("embed", {8, 4},
                                                0.02f, 0.003f),
            celeg::test_support::pattern_tensor("latent.q_proj", {2, 4},
                                                0.03f, 0.0017f),
            celeg::test_support::pattern_tensor("latent.q_expand", {4, 2},
                                                0.04f, 0.0013f),
            celeg::test_support::constant_tensor("latent.q_norm", {2}, 1.0f),
            celeg::test_support::pattern_tensor("latent.k_proj", {2, 4},
                                                0.05f, 0.0011f),
            celeg::test_support::constant_tensor("latent.k_norm", {2}, 1.0f),
            celeg::test_support::pattern_tensor("latent.expand", {8, 2},
                                                0.025f, 0.0019f),
            celeg::test_support::pattern_tensor("latent.out", {4, 4},
                                                0.02f, 0.0021f),
            celeg::test_support::pattern_tensor("latent.gate", {2, 4},
                                                -0.03f, 0.004f),
        });

    try {
        exercise_variant(directory, false);
        exercise_variant(directory, true);
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }

    std::filesystem::remove_all(directory);
    std::puts("cpu_factorized_latent_test: ok");
    return 0;
}
