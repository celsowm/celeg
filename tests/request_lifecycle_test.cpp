#include "celeg/serve/request_lifecycle.hpp"
#include "support/assertions.hpp"

#include <cstdint>
#include <vector>

int main() {
    using celeg::serve::FinishReason;
    using celeg::serve::RequestLifecycle;
    using celeg::serve::RequestStatus;

    RequestLifecycle lifecycle;
    lifecycle.submitted(7, 99);

    const std::vector<std::int32_t> first{1, 2};
    CELEG_TEST_CHECK(lifecycle.finish_reason(7, RequestStatus::Decoding, first) ==
                     FinishReason::None);

    const std::vector<std::int32_t> eos{4, 99};
    CELEG_TEST_CHECK(lifecycle.finish_reason(7, RequestStatus::Decoding, eos) ==
                     FinishReason::None);
    CELEG_TEST_CHECK(lifecycle.finish_reason(7, RequestStatus::Finished, {}) ==
                     FinishReason::Stop);

    lifecycle.submitted(8, 99);
    CELEG_TEST_CHECK(lifecycle.finish_reason(8, RequestStatus::Finished, {}) ==
                     FinishReason::Length);
    CELEG_TEST_CHECK(lifecycle.finish_reason(8, RequestStatus::Cancelled, {}) ==
                     FinishReason::Cancelled);

    lifecycle.submitted(9, 99);
    CELEG_TEST_CHECK(lifecycle.finish_reason(9, RequestStatus::Failed, {}) ==
                     FinishReason::Error);
    lifecycle.released(9);
    CELEG_TEST_CHECK(lifecycle.finish_reason(9, RequestStatus::Finished, {}) ==
                     FinishReason::Length);
}
