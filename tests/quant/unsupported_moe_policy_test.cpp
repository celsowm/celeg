
#include "weight_policy.hpp"

#include "support/assertions.hpp"

#include <stdexcept>
#include <string>

namespace {

void expect_rejected(celeg::WeightMode mode, bool is_moe,
                     const char* why_must_mention) {
    bool threw = false;
    std::string message;
    try {
        celeg::check_moe_quantization_policy(mode, is_moe);
    } catch (const std::invalid_argument& e) {
        threw = true;
        message = e.what();
    }
    CELEG_TEST_CHECK(threw);
    CELEG_TEST_CHECK(message.find("MoE") != std::string::npos);
    CELEG_TEST_CHECK(message.find(why_must_mention) != std::string::npos);
}

void expect_accepted(celeg::WeightMode mode, bool is_moe) {
    celeg::check_moe_quantization_policy(mode, is_moe);
}

void expect_not_moe_short_circuits() {
    for (auto mode : {celeg::WeightMode::Bf16, celeg::WeightMode::Int8,
                      celeg::WeightMode::Int4, celeg::WeightMode::NativeGguf}) {
        celeg::check_moe_quantization_policy(mode, /*is_moe=*/false);
    }
}

}

int main() {
    expect_not_moe_short_circuits();

    expect_accepted(celeg::WeightMode::Bf16, /*is_moe=*/true);

    expect_rejected(celeg::WeightMode::Int8, /*is_moe=*/true, "int8");
    expect_rejected(celeg::WeightMode::Int4, /*is_moe=*/true, "int4");

    expect_accepted(celeg::WeightMode::NativeGguf, /*is_moe=*/true);

    return 0;
}
