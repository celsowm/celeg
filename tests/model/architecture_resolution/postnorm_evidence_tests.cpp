#include "postnorm_evidence_tests.hpp"

#include "support.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <memory>

namespace celeg::architecture_resolution_test {

void run_postnorm_evidence_tests(const celeg::ArchitectureCatalog& catalog) {
    const auto evidence_a = resolve_postnorm_evidence(catalog, "arbitrary_evidence_model");
    const auto evidence_b = resolve_postnorm_evidence(catalog, "another_arbitrary_identity");
    CELEG_TEST_CHECK(evidence_a.graph.fingerprint() == evidence_b.graph.fingerprint());
    CELEG_TEST_CHECK(equivalent_weight_plan(evidence_a.weight_plan, evidence_b.weight_plan));
    CELEG_TEST_CHECK(evidence_a.graph.layers.size() == 4);
    for (int layer = 0; layer < 4; ++layer) {
        const auto& semantic_layer = evidence_a.graph.layers[static_cast<size_t>(layer)];
        CELEG_TEST_CHECK(!semantic_layer.mixer_norm.before.has_value());
        CELEG_TEST_CHECK(semantic_layer.mixer_norm.after.has_value());
        CELEG_TEST_CHECK(!semantic_layer.feed_forward_norm.before.has_value());
        CELEG_TEST_CHECK(semantic_layer.feed_forward_norm.after.has_value());
        CELEG_TEST_CHECK(has_weight_role(evidence_a.weight_plan,
                                         celeg::TensorRole::AttentionPostNorm, layer));
        CELEG_TEST_CHECK(has_weight_role(evidence_a.weight_plan,
                                         celeg::TensorRole::FfnOutputNorm, layer));
        CELEG_TEST_CHECK(!has_weight_role(evidence_a.weight_plan,
                                          celeg::TensorRole::AttentionInputNorm, layer));
        CELEG_TEST_CHECK(!has_weight_role(evidence_a.weight_plan,
                                          celeg::TensorRole::FfnInputNorm, layer));
        CELEG_TEST_CHECK(has_weight_role(evidence_a.weight_plan,
                                         celeg::TensorRole::AttentionQueryNorm, layer));
        CELEG_TEST_CHECK(has_weight_role(evidence_a.weight_plan,
                                         celeg::TensorRole::AttentionKeyNorm, layer));
        const auto& attention = std::get<celeg::AttentionSpec>(semantic_layer.mixer);
        CELEG_TEST_CHECK(attention.query_norm.has_value());
        CELEG_TEST_CHECK(attention.key_norm.has_value());
        if (layer < 3) {
            CELEG_TEST_CHECK(std::holds_alternative<celeg::SlidingWindowPattern>(
                attention.pattern));
            CELEG_TEST_CHECK(std::get<celeg::SlidingWindowPattern>(attention.pattern).window ==
                             4096);
        } else {
            CELEG_TEST_CHECK(std::holds_alternative<celeg::FullCausalPattern>(
                attention.pattern));
        }
        const auto& rope = std::get<celeg::RopePositionSpec>(attention.position);
        const auto& scaling = std::get<celeg::YarnRopeScaling>(rope.scaling);
        CELEG_TEST_CHECK(scaling.factor == 8.0);
        CELEG_TEST_CHECK(scaling.original_context == 8192);
    }

    /// Per-pattern RoPE: nested `rope_parameters.<layer_type>.*` fans out
    /// over the `layer_types` schedule so sliding and full layers carry
    /// distinct thetas (and rotary fractions) instead of a first-wins
    /// global. No test fed these nested keys before; without the fan-out the
    /// sliding layers would silently inherit the full theta.
    {
        celeg::CheckpointMetadata nested = postnorm_evidence_metadata("nested_rope_model");
        nested.values["rope_parameters.sliding_attention.rope_theta"] = int64_t(10000);
        nested.values["rope_parameters.full_attention.rope_theta"] = int64_t(1000000);
        nested.values["rope_parameters.full_attention.partial_rotary_factor"] = 0.5;
        celeg::CheckpointView nested_checkpoint;
        nested_checkpoint.metadata = std::move(nested);
        nested_checkpoint.repository = std::make_shared<PostnormEvidenceRepository>();
        const auto nested_model = catalog.select(nested_checkpoint.metadata).resolve(nested_checkpoint);
        CELEG_TEST_CHECK(nested_model.graph.layers.size() == 4);
        for (int layer = 0; layer < 4; ++layer) {
            const auto& semantic_layer = nested_model.graph.layers[static_cast<size_t>(layer)];
            const auto& attention = std::get<celeg::AttentionSpec>(semantic_layer.mixer);
            const auto& rope = std::get<celeg::RopePositionSpec>(attention.position);
            if (layer < 3) {
                CELEG_TEST_CHECK(rope.theta == 10000.0);
                CELEG_TEST_CHECK(rope.rotary_fraction == 1.0);
            } else {
                CELEG_TEST_CHECK(rope.theta == 1000000.0);
                CELEG_TEST_CHECK(std::abs(rope.rotary_fraction - 0.5) < 1.0e-6);
            }
            /// Global YaRN still applies (no nested `rope_type` overrides it).
            CELEG_TEST_CHECK(std::holds_alternative<celeg::YarnRopeScaling>(rope.scaling));
        }
    }
}

}
