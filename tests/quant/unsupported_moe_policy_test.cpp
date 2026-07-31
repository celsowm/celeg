// Phase 1.3: regression test for the explicit MoE quantization policy.
//
// Before Phase 1.1 / 1.2, the runtime silently accepted any global WeightMode
// against a MoE checkpoint and built a session that crashed mid-construction
// when model setup tried to cast the (null) BF16 router pointer. The Phase 1
// contract (plan section 1.2) is:
//   * safetensors + MoE + --weight-mode bf16  -> supported (dense + MoE BF16)
//   * safetensors + MoE + --weight-mode int8   -> REJECTED at model construction
//                                                with a clear diagnostic, since
//                                                experts have no INT8/INT4 kernel
//                                                and a "silently dense-quantized
//                                                with BF16 experts" report would
//                                                be misleading.
//   * safetensors + MoE + --weight-mode int4   -> rejected for the same reason.
//   * GGUF      + MoE + --weight-mode native   -> supported (Q4_K/Q6_K experts).
//
// This test simulates the construction-time decision in isolation by exercising
// the helper introduced in PR 2 (`lfm::check_moe_quantization_policy`) against
// the four combinations above and asserting that the supported pairs pass while
// the unsupported pairs throw `std::invalid_argument` whose message mentions
// the offending weight mode and the word "MoE".

#include "lfm/model/weights/policy.hpp"

#include "support/assertions.hpp"

#include <stdexcept>
#include <string>

namespace {

void expect_rejected(lfm::WeightMode mode, bool is_moe,
                     const char* why_must_mention) {
    bool threw = false;
    std::string message;
    try {
        lfm::check_moe_quantization_policy(mode, is_moe);
    } catch (const std::invalid_argument& e) {
        threw = true;
        message = e.what();
    }
    LFM_TEST_CHECK(threw);
    // The diagnostic must name MoE and the weight mode so the user can act.
    LFM_TEST_CHECK(message.find("MoE") != std::string::npos);
    LFM_TEST_CHECK(message.find(why_must_mention) != std::string::npos);
}

void expect_accepted(lfm::WeightMode mode, bool is_moe) {
    lfm::check_moe_quantization_policy(mode, is_moe);  // must not throw.
}

void expect_not_moe_short_circuits() {
    // Non-MoE models never reach the MoE policy check; any mode is fine.
    for (auto mode : {lfm::WeightMode::Bf16, lfm::WeightMode::Int8,
                      lfm::WeightMode::Int4, lfm::WeightMode::NativeGguf}) {
        lfm::check_moe_quantization_policy(mode, /*is_moe=*/false);
    }
}

} // namespace

int main() {
    // Dense (non-MoE): every mode is acceptable.
    expect_not_moe_short_circuits();

    // MoE + safetensors BF16: supported.
    expect_accepted(lfm::WeightMode::Bf16, /*is_moe=*/true);

    // MoE + safetensors INT8 / INT4: rejected.
    expect_rejected(lfm::WeightMode::Int8, /*is_moe=*/true, "int8");
    expect_rejected(lfm::WeightMode::Int4, /*is_moe=*/true, "int4");

    // MoE + NativeGguf: supported (Q4_K / Q6_K experts).
    expect_accepted(lfm::WeightMode::NativeGguf, /*is_moe=*/true);

    return 0;
}
