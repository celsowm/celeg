#include "checkpoint/hf_http.hpp"
#include "support/assertions.hpp"
#include <iostream>

int main() {
    /// The resume offset must always travel as an explicit `Range` header
    /// value, on both transports.
    CELEG_TEST_CHECK(celeg::hf_internal::range_header_value(0) == "bytes=0-");
    CELEG_TEST_CHECK(celeg::hf_internal::range_header_value(1) == "bytes=1-");
    CELEG_TEST_CHECK(celeg::hf_internal::range_header_value(16777216) == "bytes=16777216-");

    /// A partial prefix resumes; a prefix larger than the expected total is
    /// discarded (offset 0 re-opens the file truncated).
    CELEG_TEST_CHECK(celeg::hf_internal::resume_offset(0, 100) == 0);
    CELEG_TEST_CHECK(celeg::hf_internal::resume_offset(40, 100) == 40);
    CELEG_TEST_CHECK(celeg::hf_internal::resume_offset(100, 100) == 100);
    CELEG_TEST_CHECK(celeg::hf_internal::resume_offset(101, 100) == 0);
    CELEG_TEST_CHECK(celeg::hf_internal::resume_offset(101, 0) == 101);
    std::cout << "http_resume_test: ok\n";
}
