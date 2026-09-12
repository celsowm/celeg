#include "layer_facts_tests.hpp"

#include "celeg/model/inference.hpp"
#include "support.hpp"
#include "support/assertions.hpp"

#include <optional>
#include <vector>

namespace celeg::automatic_inference_test {

void run_layer_facts_tests(const celeg::ArchitectureCatalog&) {
    auto scoped = gguf_metadata();
    scoped.values["conventional.attention.head_count_kv"] =
        std::vector<int64_t>{2, 1};
    const auto scoped_facts = celeg::normalize_model_metadata(scoped);
    CELEG_TEST_CHECK(scoped_facts.attention.key_value_heads.global == std::nullopt);
    CELEG_TEST_CHECK(scoped_facts.attention.key_value_heads.value_for(0) == std::optional<int>{2});
    CELEG_TEST_CHECK(scoped_facts.attention.key_value_heads.value_for(1) == std::optional<int>{1});

    auto invalid_length = scoped;
    invalid_length.values["conventional.attention.head_count_kv"] =
        std::vector<int64_t>{2};
    bool invalid_length_rejected = false;
    try { (void)celeg::normalize_model_metadata(invalid_length); }
    catch (const celeg::ResolutionError& error) {
        invalid_length_rejected =
            error.kind() == celeg::ResolutionFailureKind::IncompleteLayerSchedule;
    }
    CELEG_TEST_CHECK(invalid_length_rejected);

    auto conflicting_scope = scoped;
    conflicting_scope.values["num_key_value_heads"] = int64_t(2);
    bool conflicting_scope_rejected = false;
    try { (void)celeg::normalize_model_metadata(conflicting_scope); }
    catch (const celeg::ResolutionError& error) {
        conflicting_scope_rejected =
            error.kind() == celeg::ResolutionFailureKind::ConflictingMetadata;
    }
    CELEG_TEST_CHECK(conflicting_scope_rejected);

    /// NormalizedModelMetadata::position_encoding is the single canonical
    /// representation of inferred positional semantics: a checkpoint carrying
    /// "active-looking" rope hparams under a no-RoPE GGUF architecture must
    /// resolve to NoPositionEncodingSpec (not a RoPE payload the runtime
    /// happens to ignore), and a checkpoint under an ordinary GGUF
    /// architecture with the same hparams must resolve to a real
    /// InferredRopePosition carrying those values through unmodified.
    const auto no_rope_facts = celeg::normalize_model_metadata(no_rope_gguf_metadata());
    CELEG_TEST_CHECK(no_rope_facts.attention.position_encoding.global.has_value());
    CELEG_TEST_CHECK(std::holds_alternative<celeg::NoPositionEncodingSpec>(
        *no_rope_facts.attention.position_encoding.global));
    const auto rope_facts = celeg::normalize_model_metadata(gguf_metadata());
    CELEG_TEST_CHECK(rope_facts.attention.position_encoding.global.has_value());
    CELEG_TEST_CHECK(std::holds_alternative<celeg::InferredRopePosition>(
        *rope_facts.attention.position_encoding.global));
    CELEG_TEST_CHECK(std::get<celeg::InferredRopePosition>(*rope_facts.attention.position_encoding.global).theta ==
        10000.0);

    auto ling_alias = metadata();
    ling_alias.values.erase("qk_norm");
    ling_alias.values["use_qk_norm"] = true;
    const auto ling_facts = celeg::normalize_model_metadata(ling_alias);
    CELEG_TEST_CHECK(ling_facts.attention.query_key_norm == std::optional<bool>{true});
}

}
