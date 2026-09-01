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

class FixtureArchitecture final : public celeg::IArchitecture {
public:
    explicit FixtureArchitecture(bool shared) : shared_(shared) {}

    std::string_view id() const override {
        return shared_ ? "cpu_shared_kv_fixture" : "cpu_private_kv_fixture";
    }

    celeg::ProbeResult probe(const celeg::CheckpointMetadata&) const override {
        return {true, 1000000, "CPU shared-KV execution fixture"};
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
        model.graph.layers.resize(2);

        for (celeg::LayerSpec& layer : model.graph.layers) {
            celeg::AttentionSpec attention;
            attention.query_heads = 1;
            attention.key_value_heads = 1;
            attention.head_dim = 4;
            attention.query_scale = 0.5f;
            attention.position = celeg::NoPositionEncodingSpec{};
            attention.pattern = celeg::FullCausalPattern{};
            layer.mixer = attention;
            layer.feed_forward = std::monostate{};
        }
        if (shared_) {
            std::get<celeg::AttentionSpec>(model.graph.layers[0].mixer).kv_sharing =
                celeg::SharedKvPublisher{0};
            std::get<celeg::AttentionSpec>(model.graph.layers[1].mixer).kv_sharing =
                celeg::SharedKvConsumer{0};
        }

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
            tensor_request(celeg::TensorRole::AttentionQuery, 0,
                           "l0.q", {4, 4}),
            tensor_request(celeg::TensorRole::AttentionKey, 0,
                           "l0.k", {4, 4}),
            tensor_request(celeg::TensorRole::AttentionValue, 0,
                           "l0.v", {4, 4}),
            tensor_request(celeg::TensorRole::AttentionOutput, 0,
                           "l0.out", {4, 4}),
            tensor_request(celeg::TensorRole::AttentionQuery, 1,
                           "l1.q", {4, 4}),
            tensor_request(celeg::TensorRole::AttentionOutput, 1,
                           "l1.out", {4, 4}),
        };
        if (!shared_) {
            model.weight_plan.requests.push_back(
                tensor_request(celeg::TensorRole::AttentionKey, 1,
                               "l1.k", {4, 4}));
            model.weight_plan.requests.push_back(
                tensor_request(celeg::TensorRole::AttentionValue, 1,
                               "l1.v", {4, 4}));
        }
        return model;
    }

private:
    bool shared_ = false;
};

std::shared_ptr<const celeg::RuntimeContext> fixture_runtime(bool shared) {
    celeg::RuntimeBuilder builder;
    builder.add_builtins();
    builder.add_architecture(std::make_unique<FixtureArchitecture>(shared));
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
        std::filesystem::temp_directory_path() / "celeg-cpu-shared-kv-test";
    celeg::test_support::write_safetensors_checkpoint(
        directory, "celeg_cpu_shared_kv_fixture", {
            celeg::test_support::pattern_tensor("embed", {8, 4}, 0.02f, 0.003f),
            celeg::test_support::pattern_tensor("l0.q", {4, 4}, 0.01f, 0.002f),
            celeg::test_support::pattern_tensor("l0.k", {4, 4}, 0.03f, 0.0015f),
            celeg::test_support::pattern_tensor("l0.v", {4, 4}, 0.04f, 0.0013f),
            celeg::test_support::constant_tensor("l0.out", {4, 4}, 0.0f),
            celeg::test_support::pattern_tensor("l1.q", {4, 4}, 0.05f, 0.0017f),
            celeg::test_support::pattern_tensor("l1.k", {4, 4}, 0.03f, 0.0015f),
            celeg::test_support::pattern_tensor("l1.v", {4, 4}, 0.04f, 0.0013f),
            celeg::test_support::pattern_tensor("l1.out", {4, 4}, 0.02f, 0.0021f),
        });

    celeg::CpuModelOptions scalar_options;
    scalar_options.use_pack_cache = false;
    scalar_options.threads = 1;
    scalar_options.prefill_chunk_tokens = 2;
    scalar_options.prefill_chunk_threshold = 64;

    celeg::CpuModelOptions chunk_options = scalar_options;
    chunk_options.prefill_chunk_threshold = 1;

    celeg::GenerationConfig generation;
    generation.seed = 11;
    generation.top_k = 1;

    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    const std::vector<int32_t> peer_prompt = {2, 3, 1, 5, 4};

    try {
        celeg::CpuModel shared_scalar(
            directory.string(), 32, scalar_options, generation,
            fixture_runtime(true));
        celeg::CpuModel private_scalar(
            directory.string(), 32, scalar_options, generation,
            fixture_runtime(false));
        celeg::CpuModel shared_chunk(
            directory.string(), 32, chunk_options, generation,
            fixture_runtime(true));
        celeg::CpuModel private_chunk(
            directory.string(), 32, chunk_options, generation,
            fixture_runtime(false));

        CELEG_TEST_CHECK(shared_scalar.shared_kv_pools().size() == 1);
        CELEG_TEST_CHECK(private_scalar.shared_kv_pools().size() == 2);

        shared_scalar.session().prefill(prompt);
        private_scalar.session().prefill(prompt);
        shared_chunk.session().prefill(prompt);
        private_chunk.session().prefill(prompt);

        compare_logits(private_scalar.diagnostics().copy_logits(),
                       shared_scalar.diagnostics().copy_logits());
        compare_logits(private_chunk.diagnostics().copy_logits(),
                       shared_chunk.diagnostics().copy_logits());
        compare_logits(shared_scalar.diagnostics().copy_logits(),
                       shared_chunk.diagnostics().copy_logits());
        compare_logits(private_scalar.diagnostics().copy_logits(),
                       private_chunk.diagnostics().copy_logits());

        const int32_t shared_scalar_token = shared_scalar.session().decode();
        const int32_t private_scalar_token = private_scalar.session().decode();
        const int32_t shared_chunk_token = shared_chunk.session().decode();
        const int32_t private_chunk_token = private_chunk.session().decode();
        CELEG_TEST_CHECK(shared_scalar_token == private_scalar_token);
        CELEG_TEST_CHECK(shared_scalar_token == shared_chunk_token);
        CELEG_TEST_CHECK(shared_scalar_token == private_chunk_token);

        compare_logits(private_scalar.diagnostics().copy_logits(),
                       shared_scalar.diagnostics().copy_logits());
        compare_logits(private_chunk.diagnostics().copy_logits(),
                       shared_chunk.diagnostics().copy_logits());
        compare_logits(shared_scalar.diagnostics().copy_logits(),
                       shared_chunk.diagnostics().copy_logits());

        celeg::CpuModel shared_packed(
            directory.string(), 32, scalar_options, generation,
            fixture_runtime(true));
        std::unique_ptr<celeg::CpuModel> shared_packed_peer =
            shared_packed.clone_session();
        celeg::CpuModel private_packed(
            directory.string(), 32, scalar_options, generation,
            fixture_runtime(false));
        std::unique_ptr<celeg::CpuModel> private_packed_peer =
            private_packed.clone_session();

        for (std::size_t index = 0; index < prompt.size(); ++index) {
            const bool final_token = index + 1 == prompt.size();
            const std::array<celeg::CpuPrefillItem, 2> shared_items{{
                {&shared_packed, prompt[index], final_token},
                {shared_packed_peer.get(), peer_prompt[index], final_token},
            }};
            const std::array<celeg::CpuPrefillItem, 2> private_items{{
                {&private_packed, prompt[index], final_token},
                {private_packed_peer.get(), peer_prompt[index], final_token},
            }};
            CELEG_TEST_CHECK(
                celeg::CpuModel::prefill_batch(shared_items).batch_size == 2);
            CELEG_TEST_CHECK(
                celeg::CpuModel::prefill_batch(private_items).batch_size == 2);
        }

        compare_logits(private_packed.diagnostics().copy_logits(),
                       shared_packed.diagnostics().copy_logits());
        compare_logits(private_packed_peer->diagnostics().copy_logits(),
                       shared_packed_peer->diagnostics().copy_logits());

        const std::array<celeg::CpuModel*, 2> shared_batch{
            &shared_packed, shared_packed_peer.get()};
        const std::array<celeg::CpuModel*, 2> private_batch{
            &private_packed, private_packed_peer.get()};
        auto [shared_tokens, shared_metrics] =
            celeg::CpuModel::decode_batch(shared_batch);
        auto [private_tokens, private_metrics] =
            celeg::CpuModel::decode_batch(private_batch);
        CELEG_TEST_CHECK(shared_metrics.batch_size == 2);
        CELEG_TEST_CHECK(private_metrics.batch_size == 2);
        CELEG_TEST_CHECK(shared_tokens == private_tokens);
        compare_logits(private_packed.diagnostics().copy_logits(),
                       shared_packed.diagnostics().copy_logits());
        compare_logits(private_packed_peer->diagnostics().copy_logits(),
                       shared_packed_peer->diagnostics().copy_logits());
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }

    std::filesystem::remove_all(directory);
    std::puts("cpu_shared_kv_test: ok");
    return 0;
}
