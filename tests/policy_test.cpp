#include "celeg/model/resolved.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    celeg::TokenPolicy tokens;
    tokens.bos_token_id = 1;
    tokens.eos_token_ids = {2};
    tokens.pad_token_id = 0;
    tokens.validate();

    celeg::NumericalPolicy numerical;
    numerical.norm_eps = 1.0e-5f;
    numerical.logits_divisor = 1.0f;
    numerical.validate();

    bool rejected_tokens = false;
    try {
        celeg::TokenPolicy invalid;
        invalid.validate();
    } catch (const std::runtime_error&) {
        rejected_tokens = true;
    }
    CELEG_TEST_CHECK(rejected_tokens);

    bool rejected_numerical = false;
    try {
        celeg::NumericalPolicy invalid;
        invalid.norm_eps = 0.0f;
        invalid.validate();
    } catch (const std::runtime_error&) {
        rejected_numerical = true;
    }
    CELEG_TEST_CHECK(rejected_numerical);
    std::cout << "policy_test: ok\n";
    return 0;
}
