#include "stated_policy_tests.hpp"

#include "support.hpp"
#include "support/assertions.hpp"

#include <memory>

namespace celeg::architecture_resolution_test {

void run_stated_policy_tests(const celeg::ArchitectureCatalog& catalog) {
    /// Stated-and-verified keys: `position_embedding_type`, `mlp_type` and
    /// `norm_type` confirm the resolved default and fail loudly on anything
    /// else, so a checkpoint asking for a different policy never resolves
    /// as RoPE/SwiGLU/RMS silently.
    {
        auto policy = structural_metadata("unknown_policy_rope");
        policy.values["position_embedding_type"] = std::string("rope");
        celeg::CheckpointView policy_checkpoint;
        policy_checkpoint.metadata = std::move(policy);
        policy_checkpoint.repository = std::make_shared<GptxRepository>();
        (void)catalog.select(policy_checkpoint.metadata).resolve(policy_checkpoint);

        auto absolute = structural_metadata("unknown_policy_absolute");
        absolute.values["position_embedding_type"] = std::string("absolute");
        CELEG_TEST_CHECK(normalize_fails_with(
            std::move(absolute), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));

        auto gated = structural_metadata("unknown_mlp_gated");
        gated.values["mlp_type"] = std::string("gated");
        celeg::CheckpointView gated_checkpoint;
        gated_checkpoint.metadata = std::move(gated);
        gated_checkpoint.repository = std::make_shared<GptxRepository>();
        (void)catalog.select(gated_checkpoint.metadata).resolve(gated_checkpoint);

        auto linear_mlp = structural_metadata("unknown_mlp_linear");
        linear_mlp.values["mlp_type"] = std::string("linear");
        CELEG_TEST_CHECK(normalize_fails_with(
            std::move(linear_mlp), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));

        auto rms = structural_metadata("unknown_norm_rms");
        rms.values["norm_type"] = std::string("rmsnorm");
        celeg::CheckpointView rms_checkpoint;
        rms_checkpoint.metadata = std::move(rms);
        rms_checkpoint.repository = std::make_shared<GptxRepository>();
        (void)catalog.select(rms_checkpoint.metadata).resolve(rms_checkpoint);

        auto layer_norm = structural_metadata("unknown_norm_layer");
        layer_norm.values["norm_type"] = std::string("layernorm");
        CELEG_TEST_CHECK(normalize_fails_with(
            std::move(layer_norm), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));
    }

    /// A nested `rope_scaling.rope_theta` restates the global theta and must
    /// agree with it instead of silently winning.
    {
        auto restated = structural_metadata("unknown_theta_restated");
        restated.values["rope_scaling.rope_theta"] = 100000.0;
        celeg::CheckpointView restated_checkpoint;
        restated_checkpoint.metadata = std::move(restated);
        restated_checkpoint.repository = std::make_shared<GptxRepository>();
        (void)catalog.select(restated_checkpoint.metadata).resolve(restated_checkpoint);

        auto conflicted = structural_metadata("unknown_theta_conflict");
        conflicted.values["rope_scaling.rope_theta"] = 999.0;
        CELEG_TEST_CHECK(normalize_fails_with(
            std::move(conflicted), celeg::ResolutionFailureKind::ConflictingMetadata));
    }
}

}
