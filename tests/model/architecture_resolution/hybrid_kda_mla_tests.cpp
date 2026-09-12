#include "hybrid_kda_mla_tests.hpp"

#include "support.hpp"
#include "support/assertions.hpp"

#include <cstdlib>
#include <string>

namespace celeg::architecture_resolution_test {

void run_hybrid_kda_mla_tests(const celeg::ArchitectureCatalog& catalog) {
    /// Ling-style hybrid: KDA layers resolve to the factorized gated-delta
    /// spec, the group-schedule layer resolves to latent attention with a
    /// head-wise gate, and the whole hybrid key set is ledger-clean under
    /// strict semantics. Strict is scoped to this block so the remaining
    /// tests keep their default warn-mode standing.
    const char* previous_strict = std::getenv("CELEG_STRICT_SEMANTICS");
    const bool had_strict = previous_strict != nullptr;
    const std::string saved_strict = had_strict ? std::string(previous_strict) : std::string();
    setenv("CELEG_STRICT_SEMANTICS", "1", 1);
    const auto hybrid = resolve_hybrid_kda_mla(catalog, hybrid_kda_mla_metadata("ling_hybrid"));
    CELEG_TEST_CHECK(hybrid.graph.layers.size() == 4);
    for (int layer = 0; layer < 3; ++layer) {
        const auto& mixer = std::get<celeg::GatedDeltaNetSpec>(
            hybrid.graph.layers[static_cast<size_t>(layer)].mixer);
        CELEG_TEST_CHECK(mixer.vector_decay);
        CELEG_TEST_CHECK(mixer.safe_decay);
        CELEG_TEST_CHECK(mixer.factorized_projections);
        CELEG_TEST_CHECK(mixer.sigmoid_output_gate);
    }
    const auto& mla = std::get<celeg::AttentionSpec>(hybrid.graph.layers[3].mixer);
    CELEG_TEST_CHECK(mla.output_gate.has_value());
    CELEG_TEST_CHECK(mla.output_gate->granularity ==
                     celeg::AttentionGateGranularity::HeadWise);
    CELEG_TEST_CHECK(mla.output_gate_width() == 2);

    /// A stated gate granularity that disagrees with the `g_proj` shape
    /// is a configuration/weight mismatch, not a resolvable model.
    {
        auto mismatch = hybrid_kda_mla_metadata("ling_gate_mismatch");
        mismatch.values["gated_attention_proj_granularity_type"] =
            std::string("element_wise");
        CELEG_TEST_CHECK(hybrid_resolve_fails_with(
            std::move(mismatch), celeg::ResolutionFailureKind::ConflictingMetadata));
    }

    /// `no_kda_lora: false` selects the LoRA-factorized KDA dialect no
    /// rule binds; it must fail loudly rather than misresolve.
    {
        auto lora = hybrid_kda_mla_metadata("ling_kda_lora");
        lora.values["no_kda_lora"] = false;
        CELEG_TEST_CHECK(hybrid_resolve_fails_with(
            std::move(lora), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));
    }

    /// A schedule claim the grammar-resolved mixers contradict (group 2
    /// would make layers 1 and 3 attention) fails loudly.
    {
        auto schedule = hybrid_kda_mla_metadata("ling_schedule_mismatch");
        schedule.values["layer_group_size"] = int64_t(2);
        CELEG_TEST_CHECK(hybrid_resolve_fails_with(
            std::move(schedule), celeg::ResolutionFailureKind::ConflictingMetadata));
    }

    /// Degenerate proven-inert values are ledger-clean; non-degenerate
    /// ones stay loud.
    {
        auto grouped = hybrid_kda_mla_metadata("ling_grouped_norm");
        grouped.values["group_norm_size"] = int64_t(2);
        CELEG_TEST_CHECK(hybrid_resolve_fails_with(
            std::move(grouped), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));
        auto kv_heads = hybrid_kda_mla_metadata("ling_linear_kv_heads");
        kv_heads.values["num_kv_heads_for_linear_attn"] = int64_t(2);
        CELEG_TEST_CHECK(hybrid_resolve_fails_with(
            std::move(kv_heads), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));
    }
    if (had_strict) {
        setenv("CELEG_STRICT_SEMANTICS", saved_strict.c_str(), 1);
    } else {
        unsetenv("CELEG_STRICT_SEMANTICS");
    }
}

}
