#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace celeg::hf_internal {

struct HttpResponse {
    unsigned long status = 0;
    std::string body;
};

/// Percent-encode a path/query component for the HuggingFace REST API.
std::string url_encode(const std::string& value);

/// Issue an HTTP request against the HuggingFace hub host and return the
/// status code and response body. Transport is WinHTTP on Windows and libcurl
/// elsewhere; callers do not depend on the implementation.
HttpResponse http_request(const std::string& method,
                          const std::string& path,
                          bool follow_redirects = true);

/// Build the value for an HTTP `Range` header that resumes a transfer at
/// `offset` bytes (`bytes=<offset>-`). Both transports use this so the header
/// is always explicit on the wire when resuming a partial download.
std::string range_header_value(size_t offset);

/// Decide the resume offset for a partial download: returns 0 when the
/// partial content cannot be trusted (larger than `expected_size`), otherwise
/// the current size to resume from. A zero `expected_size` means the total is
/// unknown, so any prefix is resumed.
size_t resume_offset(size_t current_size, size_t expected_size);

/// Stream a file from the HuggingFace resolve endpoint to `output`, resuming
/// from any existing partial content. Honors `expected_size` for progress and
/// integrity, and retries transient failures.
void http_download_file(const std::string& path,
                        const std::filesystem::path& output,
                        size_t expected_size,
                        bool quiet);

}
