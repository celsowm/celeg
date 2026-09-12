#include "alias_conflict_tests.hpp"

#include "celeg/model/inference.hpp"
#include "support.hpp"
#include "support/assertions.hpp"

namespace celeg::automatic_inference_test {

void run_alias_conflict_tests() {
    auto conflicting = metadata();
    conflicting.values["n_embd"] = int64_t(9);
    bool rejected = false;
    try { (void)celeg::normalize_model_metadata(conflicting); }
    catch (const celeg::ResolutionError& error) {
        rejected = error.kind() == celeg::ResolutionFailureKind::ConflictingMetadata;
    }
    CELEG_TEST_CHECK(rejected);
}

}
