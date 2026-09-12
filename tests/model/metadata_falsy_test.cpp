#include "celeg/checkpoint/metadata.hpp"
#include "model/inference/support.hpp"
#include "support/assertions.hpp"

#include <iostream>

/// Falsy opt-in flags are provably inert: only a set flag may fail the
/// gate, so `use_qkv_bias: false` never blocks resolution while
/// `use_qkv_bias: true` without resolver support still fails loudly.
int main() {
    CELEG_TEST_CHECK(celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{false}));
    CELEG_TEST_CHECK(celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{int64_t(0)}));
    CELEG_TEST_CHECK(celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{0.0}));
    CELEG_TEST_CHECK(!celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{true}));
    CELEG_TEST_CHECK(!celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{int64_t(1)}));
    CELEG_TEST_CHECK(!celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{std::string("false")}));

    std::cout << "metadata_falsy_test: ok\n";
}
