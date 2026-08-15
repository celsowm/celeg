#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace celeg::hf_internal {

#if defined(_WIN32)
struct HttpResponse {
    unsigned long status = 0;
    std::string body;
};

std::string url_encode(const std::string& value);
HttpResponse http_request(const std::string& method,
                          const std::string& path,
                          bool follow_redirects = true);
void http_download_file(const std::string& path,
                        const std::filesystem::path& output,
                        size_t expected_size,
                        bool quiet);
#endif

}
