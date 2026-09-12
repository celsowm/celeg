#include "qwen35_tests.hpp"

#include "celeg/checkpoint/view.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "support.hpp"
#include "support/assertions.hpp"

#include <array>
#include <cmath>
#include <cstdio>

namespace celeg::automatic_inference_test {

void run_qwen35_tests(const celeg::ArchitectureCatalog& catalog) {
    /// Qwen3.5: one linear_attn (gated-DeltaNet) layer followed by one
    /// full-attention layer combining a q_proj-fused output gate, partial
    /// rotary (0.25 of head_dim -> here 0.5, scaled for the tiny synthetic
    /// head_dim), and interleaved M-RoPE sectioning -- the exact feature
    /// combination Phase 2 needed to prove out for the generic/automatic
    /// architecture path (no per-model descriptor).
    celeg::CheckpointView qwen35_checkpoint;
    qwen35_checkpoint.metadata = qwen35_metadata();
    qwen35_checkpoint.repository = qwen35_repository();
    celeg::ResolvedModel qwen35_model;
    try {
        qwen35_model = catalog.select(qwen35_checkpoint.metadata).resolve(qwen35_checkpoint);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "qwen3.5 failure: %s\n", error.what());
        throw;
    }
    CELEG_TEST_CHECK(std::holds_alternative<celeg::GatedDeltaNetSpec>(
        qwen35_model.graph.layers[0].mixer));
    const celeg::GatedDeltaNetSpec& qwen35_delta =
        std::get<celeg::GatedDeltaNetSpec>(qwen35_model.graph.layers[0].mixer);
    CELEG_TEST_CHECK(qwen35_delta.key_heads == 2);
    CELEG_TEST_CHECK(qwen35_delta.key_head_dim == 4);
    CELEG_TEST_CHECK(qwen35_delta.value_heads == 3);
    CELEG_TEST_CHECK(qwen35_delta.value_head_dim == 4);
    CELEG_TEST_CHECK(qwen35_delta.conv_kernel == 4);

    CELEG_TEST_CHECK(std::holds_alternative<celeg::AttentionSpec>(
        qwen35_model.graph.layers[1].mixer));
    const celeg::AttentionSpec& qwen35_attention =
        std::get<celeg::AttentionSpec>(qwen35_model.graph.layers[1].mixer);
    CELEG_TEST_CHECK(qwen35_attention.output_gate.has_value());
    const auto* qwen35_mrope =
        std::get_if<celeg::MultiAxisRopeSpec>(&qwen35_attention.position);
    CELEG_TEST_CHECK(qwen35_mrope != nullptr);
    CELEG_TEST_CHECK(qwen35_mrope->interleaved);
    const std::array<int, 3> expected_sections{2, 1, 1};
    CELEG_TEST_CHECK(qwen35_mrope->sections == expected_sections);
    CELEG_TEST_CHECK(std::abs(qwen35_mrope->base.rotary_fraction - 0.5f) < 1.0e-6f);
    CELEG_TEST_CHECK(celeg::explain_resolution(qwen35_checkpoint).failures.empty());

    /// Qwen3.5's `Qwen3_5RMSNorm` multiplies by `1 + weight`, not `weight`
    /// directly (the checkpoint stores a zero-centered offset) -- detected
    /// from the same `linear_attn.in_proj_qkv.weight` grammar that selects
    /// the gated-delta layer above, so every structurally-bound norm in this
    /// checkpoint (input/post-attention layernorm, q/k-norm, final norm)
    /// must resolve to `OnePlusScale`.
    CELEG_TEST_CHECK(qwen35_model.graph.final_norm.weight_kind ==
                     celeg::NormWeightKind::OnePlusScale);
    CELEG_TEST_CHECK(qwen35_model.graph.layers[0].mixer_norm.before.has_value());
    CELEG_TEST_CHECK(qwen35_model.graph.layers[0].mixer_norm.before->weight_kind ==
                     celeg::NormWeightKind::OnePlusScale);
}

}
