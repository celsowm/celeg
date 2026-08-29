#include "hf_http.hpp"

#if defined(_WIN32)

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

namespace celeg::hf_internal {
namespace {

std::wstring to_wide(const std::string& value) {
    if (value.empty()) return L"";
    const int length = MultiByteToWideChar(
        CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0);
    std::wstring result(static_cast<size_t>(length), 0);
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(),
                        static_cast<int>(value.size()), result.data(), length);
    return result;
}

constexpr const wchar_t* kHost = L"huggingface.co";
constexpr INTERNET_PORT kPort = INTERNET_DEFAULT_HTTPS_PORT;

struct WinHttpHandle {
    HINTERNET handle = nullptr;
    WinHttpHandle() = default;
    explicit WinHttpHandle(HINTERNET value) : handle(value) {}
    ~WinHttpHandle() { if (handle) WinHttpCloseHandle(handle); }
    WinHttpHandle(const WinHttpHandle&) = delete;
    WinHttpHandle& operator=(const WinHttpHandle&) = delete;
    operator HINTERNET() const { return handle; }
    explicit operator bool() const { return handle != nullptr; }
};

}

std::string url_encode(const std::string& value) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string result;
    for (unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' ||
            c == '/' || c == ':') {
            result += static_cast<char>(c);
        } else {
            result += '%';
            result += hex[c >> 4];
            result += hex[c & 0xF];
        }
    }
    return result;
}

HttpResponse http_request(const std::string& method,
                          const std::string& path,
                          bool follow_redirects) {
    WinHttpHandle session(WinHttpOpen(
        L"celeg-native-cpp/0.0.20", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session) throw std::runtime_error("WinHttpOpen failed");
    DWORD policy = follow_redirects
        ? WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS
        : WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
    WinHttpSetOption(session, WINHTTP_OPTION_REDIRECT_POLICY,
                     &policy, sizeof(policy));
    WinHttpHandle connection(WinHttpConnect(session, kHost, kPort, 0));
    if (!connection) throw std::runtime_error("WinHttpConnect failed");
    const std::wstring wide_path = to_wide(path);
    const std::wstring wide_method = to_wide(method);
    WinHttpHandle request(WinHttpOpenRequest(
        connection, wide_method.c_str(), wide_path.c_str(), nullptr,
        WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE));
    if (!request) throw std::runtime_error("WinHttpOpenRequest failed");
    if (!WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        throw std::runtime_error("WinHttpSendRequest failed: " +
                                 std::to_string(GetLastError()));
    }
    if (!WinHttpReceiveResponse(request, nullptr)) {
        throw std::runtime_error("WinHttpReceiveResponse failed: " +
                                 std::to_string(GetLastError()));
    }
    DWORD status = 0;
    DWORD status_size = sizeof(status);
    WinHttpQueryHeaders(request,
        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
        WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size,
        WINHTTP_NO_HEADER_INDEX);

    HttpResponse response;
    response.status = status;
    DWORD available = 0;
    do {
        available = 0;
        if (!WinHttpQueryDataAvailable(request, &available)) break;
        if (available == 0) continue;
        std::vector<char> buffer(available);
        DWORD read = 0;
        if (WinHttpReadData(request, buffer.data(), available, &read) && read > 0) {
            response.body.append(buffer.data(), read);
        }
    } while (available > 0);
    return response;
}

void http_download_file(const std::string& path,
                        const std::filesystem::path& output,
                        size_t expected_size,
                        bool quiet) {
    std::error_code error;
    size_t total = std::filesystem::exists(output, error)
        ? static_cast<size_t>(std::filesystem::file_size(output, error)) : 0;
    if (error) throw std::runtime_error("cannot inspect: " + output.string());
    if (expected_size != 0 && total > expected_size) {
        std::filesystem::resize_file(output, 0, error);
        if (error) throw std::runtime_error("cannot reset: " + output.string());
        total = 0;
    }

    constexpr int kAttempts = 5;
    constexpr DWORD kChunk = 1024 * 1024;
    constexpr size_t kProgressInterval = 16 * 1024 * 1024;
    std::vector<char> buffer(kChunk);
    std::string last_error;
    for (int attempt = 0; attempt < kAttempts; ++attempt) {
        WinHttpHandle session(WinHttpOpen(
            L"celeg-native-cpp/0.0.20", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
            WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
        if (!session) throw std::runtime_error("WinHttpOpen failed");
        WinHttpSetTimeouts(session, 30000, 30000, 30000, 60000);
        DWORD read_buffer_size = kChunk;
        WinHttpSetOption(session, WINHTTP_OPTION_READ_BUFFER_SIZE,
                         &read_buffer_size, sizeof(read_buffer_size));
        DWORD redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
        WinHttpSetOption(session, WINHTTP_OPTION_REDIRECT_POLICY,
                         &redirect_policy, sizeof(redirect_policy));
        WinHttpHandle connection(WinHttpConnect(session, kHost, kPort, 0));
        if (!connection) throw std::runtime_error("WinHttpConnect failed");
        const std::wstring wide_path = to_wide(path);
        WinHttpHandle request(WinHttpOpenRequest(
            connection, L"GET", wide_path.c_str(), nullptr, WINHTTP_NO_REFERER,
            WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE));
        if (!request) throw std::runtime_error("WinHttpOpenRequest failed");
        if (total != 0) {
            const std::wstring range = L"Range: bytes=" +
                std::to_wstring(total) + L"-\r\n";
            if (!WinHttpAddRequestHeaders(request, range.c_str(),
                                          static_cast<DWORD>(range.size()),
                                          WINHTTP_ADDREQ_FLAG_ADD)) {
                throw std::runtime_error("cannot add Range header: " +
                                         std::to_string(GetLastError()));
            }
        }
        if (!WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
            !WinHttpReceiveResponse(request, nullptr)) {
            last_error = "HTTP request failed: " + std::to_string(GetLastError());
            continue;
        }
        DWORD status = 0;
        DWORD status_size = sizeof(status);
        WinHttpQueryHeaders(request,
            WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size,
            WINHTTP_NO_HEADER_INDEX);
        if (status == 200 && total != 0) {
            std::ofstream reset(output, std::ios::binary | std::ios::trunc);
            if (!reset) throw std::runtime_error("cannot reset: " + output.string());
            total = 0;
            continue;
        }
        if (status != (total == 0 ? 200 : 206)) {
            last_error = "HTTP " + std::to_string(status) + " for " + path;
            continue;
        }
        std::ofstream out(output, std::ios::binary | std::ios::app);
        if (!out) throw std::runtime_error("cannot open: " + output.string());
        size_t last_report = total;
        bool complete = false;
        for (;;) {
            DWORD available = 0;
            if (!WinHttpQueryDataAvailable(request, &available)) {
                last_error = "download interrupted: " +
                    std::to_string(GetLastError());
                break;
            }
            if (available == 0) {
                complete = true;
                break;
            }
            const DWORD to_read = std::min(available, kChunk);
            DWORD read = 0;
            if (!WinHttpReadData(request, buffer.data(), to_read, &read) || read == 0) {
                last_error = "download interrupted: " +
                    std::to_string(GetLastError());
                break;
            }
            out.write(buffer.data(), static_cast<std::streamsize>(read));
            if (!out) throw std::runtime_error("write failed: " + output.string());
            total += read;
            if (!quiet && expected_size != 0 &&
                (total - last_report >= kProgressInterval || total == expected_size)) {
                const double percent = 100.0 * static_cast<double>(total) /
                                       static_cast<double>(expected_size);
                std::fprintf(stderr, "\r  %zu / %zu bytes (%.1f%%)",
                             total, expected_size, percent);
                last_report = total;
            }
        }
        out.close();
        if (complete && (expected_size == 0 || total == expected_size)) {
            if (!quiet && expected_size != 0) std::fprintf(stderr, "\n");
            return;
        }
        if (complete) {
            last_error = "download size mismatch: got " + std::to_string(total) +
                ", expected " + std::to_string(expected_size);
        }
    }
    throw std::runtime_error("download failed after retries: " + last_error);
}

}

#else  // !_WIN32: POSIX libcurl transport

#include <curl/curl.h>

#include <cstdio>
#include <fstream>
#include <system_error>

namespace celeg::hf_internal {
namespace {

constexpr const char* kHost = "huggingface.co";

size_t body_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* body = static_cast<std::string*>(userdata);
    body->append(ptr, size * nmemb);
    return size * nmemb;
}

constexpr size_t kProgressInterval = 16 * 1024 * 1024;

struct DownloadWriteState {
    std::ofstream stream;
    std::filesystem::path output;
    size_t total = 0;
    size_t expected = 0;
    bool quiet = false;
    size_t last_report = 0;
    std::string* error = nullptr;
};

size_t download_write_callback(char* ptr, size_t size, size_t nmemb,
                              void* userdata) {
    auto* st = static_cast<DownloadWriteState*>(userdata);
    const size_t got = size * nmemb;
    st->stream.write(ptr, static_cast<std::streamsize>(got));
    if (!st->stream) {
        *st->error = "write failed: " + st->output.string();
        return static_cast<size_t>(0);
    }
    st->total += got;
    if (!st->quiet && st->expected != 0 &&
        (st->total - st->last_report >= kProgressInterval ||
         st->total == st->expected)) {
        const double percent = 100.0 * static_cast<double>(st->total) /
                               static_cast<double>(st->expected);
        std::fprintf(stderr, "\r  %zu / %zu bytes (%.1f%%)", st->total,
                     st->expected, percent);
        st->last_report = st->total;
    }
    return got;
}

}  // namespace

std::string url_encode(const std::string& value) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string result;
    for (unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' ||
            c == '/' || c == ':') {
            result += static_cast<char>(c);
        } else {
            result += '%';
            result += hex[c >> 4];
            result += hex[c & 0xF];
        }
    }
    return result;
}

HttpResponse http_request(const std::string& method,
                          const std::string& path,
                          bool follow_redirects) {
    HttpResponse response;
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    const std::string url = std::string("https://") + kHost + path;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION,
                     follow_redirects ? 1L : 0L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, body_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.body);

    CURLcode code = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    if (code != CURLE_OK) {
        const std::string message = "HTTP " + method + " failed: " +
                                    curl_easy_strerror(code);
        curl_easy_cleanup(curl);
        throw std::runtime_error(message);
    }
    response.status = static_cast<unsigned long>(status);
    curl_easy_cleanup(curl);
    return response;
}

void http_download_file(const std::string& path,
                        const std::filesystem::path& output,
                        size_t expected_size,
                        bool quiet) {
    std::error_code error;
    size_t total = std::filesystem::exists(output, error)
                       ? static_cast<size_t>(std::filesystem::file_size(output, error))
                       : 0;
    if (error) throw std::runtime_error("cannot inspect: " + output.string());
    if (expected_size != 0 && total > expected_size) {
        std::filesystem::resize_file(output, 0, error);
        if (error) throw std::runtime_error("cannot reset: " + output.string());
        total = 0;
    }

    constexpr int kAttempts = 5;
    std::string last_error;

    for (int attempt = 0; attempt < kAttempts; ++attempt) {
        DownloadWriteState state;
        state.output = output;
        state.total = total;
        state.expected = expected_size;
        state.quiet = quiet;
        state.last_report = total;
        state.error = &last_error;
        state.stream.open(output, std::ios::binary |
                                       (total == 0 ? std::ios::trunc : std::ios::app));
        if (!state.stream) {
            throw std::runtime_error("cannot open: " + output.string());
        }

        CURL* curl = curl_easy_init();
        if (!curl) throw std::runtime_error("curl_easy_init failed");

        const std::string url = std::string("https://") + kHost + path;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &state);
        if (total != 0) {
            curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE,
                             static_cast<curl_off_t>(total));
        }

        CURLcode code = curl_easy_perform(curl);
        long status = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
        curl_easy_cleanup(curl);
        state.stream.close();

        if (code == CURLE_OK &&
            (status == 200 || (total != 0 && status == 206)) &&
            (expected_size == 0 || state.total == expected_size)) {
            if (!quiet && expected_size != 0) std::fprintf(stderr, "\n");
            return;
        }
        if (code != CURLE_OK) {
            last_error = "HTTP GET failed: " + std::string(curl_easy_strerror(code));
            // Retry from scratch if the partial write is suspect.
            std::filesystem::resize_file(output, 0, error);
            total = 0;
        } else if (status == 200 && total != 0) {
            // Server ignored the range request; restart from scratch.
            std::filesystem::resize_file(output, 0, error);
            total = 0;
            last_error = "server ignored resume; restarting";
        } else {
            last_error = "HTTP " + std::to_string(status) + " for " + path;
        }
    }
    throw std::runtime_error("download failed after retries: " + last_error);
}

#endif
