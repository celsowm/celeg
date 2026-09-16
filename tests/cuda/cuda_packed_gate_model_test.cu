/// Caller-level regression test for the packed Q+Gate decode path
/// (Qwen3.8-27B-NVFP4): a one-layer synthetic model whose attention carries
/// a checkpoint-packed output gate runs prefill plus decode on CUDA and on
/// the CPU backend from the same fixture, and logits plus sampled tokens
/// must agree. Skipping the decode-side gate extraction (the old bug made
/// every query head gate-contaminated and misrouted the gate itself) makes
/// the CUDA forward diverge, so this fails without the fix.
#include "backend/cuda/model/detail/compiled_model.hpp"
#include "backend/cuda/weight_cache.hpp"
#include "celeg/backend/cpu/model.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/runtime/context.hpp"
#include "cpu/support/synthetic_checkpoint.hpp"
#include "../support/assertions.hpp"
#include "../support/numerical_compare.hpp"

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
    std::string_view id() const override {
        return "cuda_packed_gate_fixture";
    }

    celeg::ProbeResult probe(const celeg::CheckpointMetadata&) const override {
        return {true, 1000000, "CUDA packed-gate decode fixture"};
    }

    celeg::ResolvedModel resolve(const celeg::CheckpointView&) const override {
        celeg::ResolvedModel model;
        model.provenance.architecture_id = std::string(id());
        model.provenance.identity = std::string(id());
        model.provenance.source_format = "safetensors";
        model.graph.hidden = 8;
        model.graph.final_norm = celeg::NormSpec{
            1.0e-5f, celeg::NormWeightKind::None,
            celeg::NormGranularity::WholeVector};
        model.graph.tied_embeddings = true;
        model.graph.layers.resize(1);

        /// Two query heads are essential: with a single head the packed
        /// per-head interleave and the coarse halves split coincide on head
        /// zero, and the misreading the decode path once had would go
        /// unnoticed.
        celeg::AttentionSpec attention;
        attention.query_heads = 2;
        attention.key_value_heads = 1;
        attention.head_dim = 4;
        attention.query_scale = 1.0f;
        attention.output_gate = celeg::SigmoidAttentionGateSpec{
            true, celeg::AttentionGateGranularity::ElementWise};
        attention.position = celeg::NoPositionEncodingSpec{};
        attention.pattern = celeg::FullCausalPattern{};
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
                           "embed", {8, 8}),
            /// Required by the CUDA loader even when embeddings are tied
            /// (it resolves the head name unconditionally).
            tensor_request(celeg::TensorRole::LanguageModelHead, -1,
                           "head", {8, 8}),
            /// Packed Q+Gate projection: two heads of query plus two heads
            /// of gate, interleaved per head.
            tensor_request(celeg::TensorRole::AttentionQuery, 0,
                           "q", {16, 8}),
            tensor_request(celeg::TensorRole::AttentionKey, 0,
                           "k", {4, 8}),
            tensor_request(celeg::TensorRole::AttentionValue, 0,
                           "v", {4, 8}),
            tensor_request(celeg::TensorRole::AttentionOutput, 0,
                           "out", {8, 8}),
        };
        return model;
    }
};

std::shared_ptr<const celeg::RuntimeContext> fixture_runtime() {
    celeg::RuntimeBuilder builder;
    builder.add_builtins();
    builder.add_architecture(std::make_unique<FixtureArchitecture>());
    return builder.build_shared();
}

void compare_logits(const std::vector<float>& expected,
                    const std::vector<float>& actual) {
    CELEG_TEST_CHECK(expected.size() == actual.size());
    CELEG_TEST_CHECK(
        celeg::test::numerical::max_absolute_error(expected, actual) < 0.05f);
    CELEG_TEST_CHECK(
        celeg::test::numerical::cosine_similarity(expected, actual) >= 0.999);
}

}

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "celeg-cuda-packed-gate-test";
    /// Value design carries the sensitivity, and it must push both
    /// nonlinearities into their responsive regimes: embeddings are O(1) so
    /// the packed projection lands gate values across sigmoid's slope
    /// (halves-vs-interleave then changes the multiplicative gate instead
    /// of two copies of sigmoid(0) ~= 0.5), and QK dots land O(1) so
    /// attention spreads mass (a flat uniform softmax would wash out any
    /// query contamination). Tiny-positive-everywhere values fail both and
    /// let a broken mixer hide.
    celeg::test_support::write_safetensors_checkpoint(
        directory, "celeg_cuda_packed_gate_fixture", {
            celeg::test_support::pattern_tensor("embed", {8, 8}, -0.50f, 0.120f),
            celeg::test_support::pattern_tensor("q", {16, 8}, -0.30f, 0.050f),
            celeg::test_support::pattern_tensor("k", {4, 8}, -0.25f, 0.040f),
            celeg::test_support::pattern_tensor("v", {4, 8}, -0.60f, 0.080f),
            celeg::test_support::pattern_tensor("out", {8, 8}, -0.20f, 0.050f),
            celeg::test_support::pattern_tensor("head", {8, 8}, 0.015f, 0.0025f),
        });

    celeg::CpuModelOptions cpu_options;
    cpu_options.use_pack_cache = false;
    cpu_options.threads = 1;
    cpu_options.prefill_chunk_tokens = 2;
    cpu_options.prefill_chunk_threshold = 64;
    /// Lossless reference: the default Q4 group packing would drown the
    /// CUDA-vs-CPU comparison in quantization noise (and it did -- the
    /// first passing runs only proved both backends read the same table).
    cpu_options.weight_format = celeg::CpuWeightFormat::Bf16;

    celeg::CudaModelOptions cuda_options;
    cuda_options.cuda_graph = false;

    celeg::GenerationConfig generation;
    generation.seed = 11;
    generation.top_k = 1;
    generation.temperature = 0.0f;
    generation.repetition_penalty = 1.0f;

    const std::vector<int32_t> prompt = {1, 2, 3};

    try {
        celeg::CpuModel cpu(
            directory.string(), 32, cpu_options, generation,
            fixture_runtime());
        celeg::CudaWeightCache weight_cache;
        celeg::CudaCompiledModel cuda(
            directory.string(), 32, cuda_options, generation, weight_cache,
            fixture_runtime());

        cpu.session().prefill(prompt);
        cuda.prefill(prompt);
        compare_logits(cpu.diagnostics().copy_logits(), cuda.copy_logits());

        const int32_t cpu_token = cpu.session().decode();
        const int32_t cuda_token = cuda.decode();
        CELEG_TEST_CHECK(cpu_token == cuda_token);
        compare_logits(cpu.diagnostics().copy_logits(), cuda.copy_logits());
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }

    std::filesystem::remove_all(directory);
    std::puts("cuda_packed_gate_model_test: ok");
    return 0;
}
