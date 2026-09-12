#include "agnes_tests.hpp"

#include "celeg/checkpoint/view.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "support.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <cstdio>
#include <optional>
#include <utility>

namespace celeg::automatic_inference_test {

void run_agnes_tests(const celeg::ArchitectureCatalog& catalog) {
    /// Agnes: `delta_attn` spelling, `agnes_*` layer-type tokens,
    /// `global_attn` full attention with a stated output gate, and a parallel
    /// FFN branch -- resolved under both layer roots the bindings accept.
    const auto agnes_facts = celeg::normalize_model_metadata(agnes_metadata());
    CELEG_TEST_CHECK(agnes_facts.core.parallel_intermediate == std::optional<int>{8});
    CELEG_TEST_CHECK(agnes_facts.attention.output_gate == std::optional<bool>{true});
    CELEG_TEST_CHECK(agnes_facts.gated_delta.hybrid_group_size == std::optional<int>{2});
    CELEG_TEST_CHECK(agnes_facts.attention.position_encoding.global.has_value());
    CELEG_TEST_CHECK(std::abs(std::get<celeg::InferredRopePosition>(
                                  *agnes_facts.attention.position_encoding.global)
                                  .rotary_fraction -
                              0.5f) < 1.0e-6f);
    for (const auto& [layer_root, model_root] : {
             std::pair<std::string, std::string>{"model.layers.", "model."},
             std::pair<std::string, std::string>{"model.language_model.layers.",
                                                 "model.language_model."}}) {
        celeg::CheckpointView agnes_checkpoint;
        agnes_checkpoint.metadata = agnes_metadata();
        agnes_checkpoint.repository = agnes_repository(layer_root, model_root);
        celeg::ResolvedModel agnes_model;
        try {
            agnes_model =
                catalog.select(agnes_checkpoint.metadata).resolve(agnes_checkpoint);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "agnes failure (%s): %s\n", layer_root.c_str(),
                         error.what());
            throw;
        }
        check_agnes_model(agnes_model);
        CELEG_TEST_CHECK(celeg::explain_resolution(agnes_checkpoint).failures.empty());
    }
}

}
