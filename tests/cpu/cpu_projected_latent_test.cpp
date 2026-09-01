#include "celeg/backend/cpu/model.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/runtime/context.hpp"
#include "cpu/support/synthetic_checkpoint.hpp"
#include "support/assertions.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

class ProjectedLatentArchitecture final : public celeg::IArchitecture {
public:
    std::string_view id() const override {
        return "cpu_projected_latent_fixture";
    }

    celeg::ProbeResult probe(const celeg::CheckpointMetadata&) const override {
        return {true, 1000000, "CPU projected-latent execution fixture"};
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

        celeg::AttentionSpec attention;
        attention.query_heads = 1;
        attention.key_value_heads = 1;
        attention.head_dim = 4;
        attention.query_scale = 0.5f;
        attention.position = celeg::NoPositionEncodingSpec{};
        attention.pattern = celeg::FullCausalPattern{};
        celeg::LatentAttentionStateSpec latent;
        latent.latent_rank = 4;
        latent.rope_head_dim = 0;
        latent.nope_head_dim = 4;
        latent.decoupled_rope = false;
        attention.state = latent;
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
            tensor_request(celeg::TensorRole::AttentionLatentQuery, 0,
                           "latent.q", {4, 4}),
            tensor_request(celeg::TensorRole::AttentionLatentKey, 0,
                           "latent.k", {4, 4}),
            tensor_request(celeg::TensorRole::AttentionLatentValue, 0,
                           "latent.v", {4, 4}),
            tensor_request(celeg::TensorRole::AttentionLatentOutput, 0,
                           "latent.out", {4, 4}),
        };
        return model;
    }
};

std::shared_ptr<const celeg::RuntimeContext> fixture_runtime() {
    celeg::RuntimeBuilder builder;
    builder.add_builtins();
    builder.add_architecture(std::make_unique<ProjectedLatentArchitecture>());
    return builder.build_shared();
}

void compare_logits(const std::vector<float>& expected,
                    const std::vector<float>& actual) {
    CELEG_TEST_CHECK(expected.size() == actual.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        CELEG_TEST_CHECK(std::abs(expected[index] - actual[index]) < 1.0e-4f);
    }
}

}

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        "celeg-cpu-projected-latent-test";
    celeg::test_support::write_safetensors_checkpoint(
        directory, "celeg_cpu_projected_latent_fixture", {
            celeg::test_support::pattern_tensor("embed", {8, 4},
                                                0.02f, 0.003f),
            celeg::test_support::pattern_tensor("latent.q", {4, 4},
                                                0.03f, 0.0017f),
            celeg::test_support::pattern_tensor("latent.k", {4, 4},
                                                0.04f, 0.0013f),
            celeg::test_support::pattern_tensor("latent.v", {4, 4},
                                                0.05f, 0.0011f),
            celeg::test_support::pattern_tensor("latent.out", {4, 4},
                                                0.02f, 0.0021f),
        });

    celeg::CpuModelOptions scalar_options;
    scalar_options.use_pack_cache = false;
    scalar_options.threads = 1;
    scalar_options.prefill_chunk_tokens = 2;
    scalar_options.prefill_chunk_threshold = 64;

    celeg::CpuModelOptions chunk_options = scalar_options;
    chunk_options.prefill_chunk_threshold = 1;

    celeg::GenerationConfig generation;
    generation.seed = 19;
    generation.top_k = 1;
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    const std::vector<int32_t> peer_prompt = {2, 3, 1, 5, 4};

    try {
        celeg::CpuModel scalar(
            directory.string(), 32, scalar_options, generation,
            fixture_runtime());
        celeg::CpuModel chunked(
            directory.string(), 32, chunk_options, generation,
            fixture_runtime());

        scalar.session().prefill(prompt);
        chunked.session().prefill(prompt);
        compare_logits(scalar.diagnostics().copy_logits(),
                       chunked.diagnostics().copy_logits());
        CELEG_TEST_CHECK(scalar.session().position() ==
                         chunked.session().position());

        const int32_t scalar_token = scalar.session().decode();
        const int32_t chunked_token = chunked.session().decode();
        CELEG_TEST_CHECK(scalar_token == chunked_token);
        compare_logits(scalar.diagnostics().copy_logits(),
                       chunked.diagnostics().copy_logits());
        CELEG_TEST_CHECK(scalar.session().position() ==
                         chunked.session().position());

        celeg::CpuModel packed(
            directory.string(), 32, scalar_options, generation,
            fixture_runtime());
        std::unique_ptr<celeg::CpuModel> packed_peer = packed.clone_session();
        celeg::CpuModel packed_reference(
            directory.string(), 32, scalar_options, generation,
            fixture_runtime());
        celeg::CpuModel peer_reference(
            directory.string(), 32, scalar_options, generation,
            fixture_runtime());
        packed_reference.session().prefill(prompt);
        peer_reference.session().prefill(peer_prompt);

        for (std::size_t index = 0; index < prompt.size(); ++index) {
            const bool final_token = index + 1 == prompt.size();
            const std::array<celeg::CpuPrefillItem, 2> items{{
                {&packed, prompt[index], final_token},
                {packed_peer.get(), peer_prompt[index], final_token},
            }};
            const celeg::CpuBatchMetrics metrics =
                celeg::CpuModel::prefill_batch(items);
            CELEG_TEST_CHECK(metrics.batch_size == 2);
        }
        compare_logits(packed_reference.diagnostics().copy_logits(),
                       packed.diagnostics().copy_logits());
        compare_logits(peer_reference.diagnostics().copy_logits(),
                       packed_peer->diagnostics().copy_logits());

        const int32_t reference_token = packed_reference.session().decode();
        const int32_t peer_reference_token = peer_reference.session().decode();
        const std::array<celeg::CpuModel*, 2> batch_models{
            &packed, packed_peer.get()};
        auto [batch_tokens, batch_metrics] =
            celeg::CpuModel::decode_batch(batch_models);
        CELEG_TEST_CHECK(batch_metrics.batch_size == 2);
        CELEG_TEST_CHECK(batch_tokens.size() == 2);
        CELEG_TEST_CHECK(batch_tokens[0] == reference_token);
        CELEG_TEST_CHECK(batch_tokens[1] == peer_reference_token);
        compare_logits(packed_reference.diagnostics().copy_logits(),
                       packed.diagnostics().copy_logits());
        compare_logits(peer_reference.diagnostics().copy_logits(),
                       packed_peer->diagnostics().copy_logits());
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }

    std::filesystem::remove_all(directory);
    std::puts("cpu_projected_latent_test: ok");
    return 0;
}
