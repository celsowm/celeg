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

/// Stream a file from the HuggingFace resolve endpoint to `output`, resuming
/// from any existing partial content. Honors `expected_size` for progress and
/// integrity, and retries transient failures.
void http_download_file(const std::string& path,
                        const std::filesystem::path& output,
                        size_t expected_size,
                        bool quiet);

}
