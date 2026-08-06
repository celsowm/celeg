#include "celeg/checkpoint/downloader.hpp"
#include "celeg/checkpoint/formats/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>
#include <shlwapi.h>
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "shlwapi.lib")
#endif

namespace celeg {

namespace {

std::string repo_folder_name(const std::string& repo_id) {
    std::string folder = "models";
    size_t pos = 0;
    while (pos < repo_id.size()) {
        size_t slash = repo_id.find('/', pos);
        std::string part = (slash == std::string::npos)
            ? repo_id.substr(pos)
            : repo_id.substr(pos, slash - pos);
        folder += "--" + part;
        if (slash == std::string::npos) break;
        pos = slash + 1;
    }
    return folder;
}

bool is_commit_hash(const std::string& s) {
    if (s.size() != 40) return false;
    return std::all_of(s.begin(), s.end(),
                       [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); });
}

bool contains_case_insensitive(std::string_view value, std::string_view needle) {
    if (needle.empty() || value.size() < needle.size()) return false;
    for (size_t start = 0; start + needle.size() <= value.size(); ++start) {
        bool match = true;
        for (size_t index = 0; index < needle.size(); ++index) {
            const auto lower = [](char c) {
                return static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
            };
            if (lower(value[start + index]) != lower(needle[index])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

std::string format_bytes(size_t bytes) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB"};
    double v = static_cast<double>(bytes);
    int u = 0;
    while (v >= 1024.0 && u < 3) { v /= 1024.0; ++u; }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(u ? 2 : 0) << v << ' ' << units[u];
    return ss.str();
}

} // namespace

#if defined(_WIN32)

namespace {

std::wstring to_wide(const std::string& s) {
    if (s.empty()) return L"";
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(),
                                  static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(),
                        static_cast<int>(s.size()), w.data(), len);
    return w;
}

std::string to_narrow(const wchar_t* w, int len) {
    if (len == 0) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w, len,
                                nullptr, 0, nullptr, nullptr);
    std::string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w, len,
                        s.data(), n, nullptr, nullptr);
    return s;
}

std::string url_encode(const std::string& s) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~'
            || c == '/' || c == ':')
            out += static_cast<char>(c);
        else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 0xF];
        }
    }
    return out;
}

constexpr const char* HF_ENDPOINT = "https://huggingface.co";
constexpr const wchar_t* HF_HOST_W = L"huggingface.co";
constexpr INTERNET_PORT HF_PORT = INTERNET_DEFAULT_HTTPS_PORT;

struct WinHttpHandle {
    HINTERNET h = nullptr;
    WinHttpHandle() = default;
    explicit WinHttpHandle(HINTERNET v) : h(v) {}
    ~WinHttpHandle() { if (h) WinHttpCloseHandle(h); }
    WinHttpHandle(const WinHttpHandle&) = delete;
    WinHttpHandle& operator=(const WinHttpHandle&) = delete;
    operator HINTERNET() const { return h; }
    explicit operator bool() const { return h != nullptr; }
};

struct HttpResponse {
    DWORD status = 0;
    std::string body;
};

HttpResponse http_request(const std::string& method,
                          const std::string& path,
                          bool follow_redirects = true) {
    WinHttpHandle session(WinHttpOpen(
        L"celeg-native-cpp/0.0.20",
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session) throw std::runtime_error("WinHttpOpen failed");

    DWORD policy = follow_redirects
        ? WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS
        : WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
    WinHttpSetOption(session, WINHTTP_OPTION_REDIRECT_POLICY,
                     &policy, sizeof(policy));

    WinHttpHandle conn(WinHttpConnect(session, HF_HOST_W, HF_PORT, 0));
    if (!conn) throw std::runtime_error("WinHttpConnect failed");

    std::wstring wpath = to_wide(path);
    WinHttpHandle req(WinHttpOpenRequest(
        conn, to_wide(method).c_str(), wpath.c_str(),
        nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
        WINHTTP_FLAG_SECURE));
    if (!req) throw std::runtime_error("WinHttpOpenRequest failed");

    if (!WinHttpSendRequest(req, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0))
        throw std::runtime_error("WinHttpSendRequest failed: " +
            std::to_string(GetLastError()));

    if (!WinHttpReceiveResponse(req, nullptr))
        throw std::runtime_error("WinHttpReceiveResponse failed: " +
            std::to_string(GetLastError()));

    DWORD statusCode = 0;
    DWORD size = sizeof(statusCode);
    WinHttpQueryHeaders(req,
        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
        WINHTTP_HEADER_NAME_BY_INDEX, &statusCode, &size,
        WINHTTP_NO_HEADER_INDEX);

    HttpResponse resp;
    resp.status = statusCode;

    DWORD avail = 0;
    do {
        avail = 0;
        if (!WinHttpQueryDataAvailable(req, &avail)) break;
        if (avail > 0) {
            std::vector<char> buf(avail);
            DWORD read = 0;
            if (WinHttpReadData(req, buf.data(), avail, &read) && read > 0)
                resp.body.append(buf.data(), read);
        }
    } while (avail > 0);

    return resp;
}

void http_download_file(const std::string& path,
                        const std::filesystem::path& output,
                        size_t expected_size,
                        bool quiet) {
    std::error_code ec;
    size_t total = std::filesystem::exists(output, ec)
        ? static_cast<size_t>(std::filesystem::file_size(output, ec)) : 0;
    if (ec) throw std::runtime_error("cannot inspect: " + output.string());
    if (expected_size != 0 && total > expected_size) {
        std::filesystem::resize_file(output, 0, ec);
        if (ec) throw std::runtime_error("cannot reset: " + output.string());
        total = 0;
    }

    constexpr int kAttempts = 5;
    constexpr DWORD kChunk = 1024 * 1024;
    constexpr size_t kProgressInterval = 16 * 1024 * 1024;
    std::vector<char> buffer(kChunk);
    std::string last_error;
    for (int attempt = 0; attempt < kAttempts; ++attempt) {
        WinHttpHandle session(WinHttpOpen(
            L"celeg-native-cpp/0.0.20",
            WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
            WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
        if (!session) throw std::runtime_error("WinHttpOpen failed");
        // A dropped large-file connection must fail and resume, rather than
        // leaving a silent, truncated blob in the Hugging Face cache.
        WinHttpSetTimeouts(session, 30000, 30000, 30000, 60000);
        DWORD read_buffer_size = kChunk;
        WinHttpSetOption(session, WINHTTP_OPTION_READ_BUFFER_SIZE,
                         &read_buffer_size, sizeof(read_buffer_size));
        DWORD policy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
        WinHttpSetOption(session, WINHTTP_OPTION_REDIRECT_POLICY,
                         &policy, sizeof(policy));

        WinHttpHandle conn(WinHttpConnect(session, HF_HOST_W, HF_PORT, 0));
        if (!conn) throw std::runtime_error("WinHttpConnect failed");
        WinHttpHandle req(WinHttpOpenRequest(
            conn, L"GET", to_wide(path).c_str(),
            nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
            WINHTTP_FLAG_SECURE));
        if (!req) throw std::runtime_error("WinHttpOpenRequest failed");

        std::wstring range;
        if (total != 0) {
            range = L"Range: bytes=" + std::to_wstring(total) + L"-\r\n";
            if (!WinHttpAddRequestHeaders(req, range.c_str(),
                                          static_cast<DWORD>(range.size()),
                                          WINHTTP_ADDREQ_FLAG_ADD)) {
                throw std::runtime_error("cannot add Range header: " +
                                         std::to_string(GetLastError()));
            }
        }
        if (!WinHttpSendRequest(req, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
            !WinHttpReceiveResponse(req, nullptr)) {
            last_error = "HTTP request failed: " + std::to_string(GetLastError());
            continue;
        }

        DWORD status = 0;
        DWORD status_size = sizeof(status);
        WinHttpQueryHeaders(req,
            WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size,
            WINHTTP_NO_HEADER_INDEX);
        if (status == 200 && total != 0) {
            // Some mirrors ignore Range. Restart instead of appending a full
            // file after the partial prefix.
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
            if (!WinHttpQueryDataAvailable(req, &available)) {
                last_error = "download interrupted: " + std::to_string(GetLastError());
                break;
            }
            if (available == 0) {
                complete = true;
                break;
            }
            const DWORD to_read = std::min(available, kChunk);
            DWORD read = 0;
            if (!WinHttpReadData(req, buffer.data(), to_read, &read) || read == 0) {
                last_error = "download interrupted: " + std::to_string(GetLastError());
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

std::string normalize_etag(const std::string& etag) {
    std::string s = etag;
    if (s.rfind("W/", 0) == 0) s = s.substr(2);
    if (!s.empty() && s.front() == '"') s = s.substr(1);
    if (!s.empty() && s.back() == '"') s.pop_back();
    return s;
}

bool create_link_or_copy(const std::filesystem::path& src,
                         const std::filesystem::path& dst) {
    std::error_code ec;
    if (std::filesystem::exists(dst, ec))
        std::filesystem::remove(dst, ec);
    std::filesystem::create_hard_link(src, dst, ec);
    if (!ec) return true;
    std::filesystem::copy_file(src, dst,
        std::filesystem::copy_options::overwrite_existing, ec);
    return !ec;
}

struct TreeFile {
    std::string path;
    size_t size = 0;
    std::string oid;
    std::string lfs_oid;
};

std::filesystem::path resume_path_for(const std::filesystem::path& blob_path) {
    std::filesystem::path canonical = blob_path;
    canonical += ".incomplete";
    std::error_code ec;
    if (std::filesystem::exists(canonical, ec)) return canonical;

    // huggingface_hub uses a random suffix for interrupted blobs. Reuse its
    // largest matching partial instead of making a second multi-gigabyte
    // transfer when switching to the native downloader.
    const std::string prefix = blob_path.filename().string() + ".";
    constexpr std::string_view suffix = ".incomplete";
    std::filesystem::path best = canonical;
    uintmax_t best_size = 0;
    for (const auto& entry : std::filesystem::directory_iterator(
             blob_path.parent_path(), ec)) {
        if (ec || !entry.is_regular_file(ec)) continue;
        const std::string name = entry.path().filename().string();
        if (!name.starts_with(prefix) || !name.ends_with(suffix)) continue;
        const uintmax_t size = entry.file_size(ec);
        if (!ec && size > best_size) {
            best = entry.path();
            best_size = size;
        }
    }
    return best;
}

std::string resolve_revision(const std::string& repo_id,
                             const std::string& revision) {
    std::string path = "/api/models/" + url_encode(repo_id)
                     + "/revision/" + url_encode(revision);
    HttpResponse resp = http_request("GET", path);
    if (resp.status != 200)
        throw std::runtime_error("cannot resolve revision " + revision
            + " for " + repo_id + ": HTTP " + std::to_string(resp.status));
    Json root = Json::parse(resp.body);
    if (!root.contains("sha"))
        throw std::runtime_error("API response missing 'sha' field");
    return root["sha"].as_string();
}

std::vector<TreeFile> list_repo_files(const std::string& repo_id,
                                      const std::string& commit) {
    std::string path = "/api/models/" + url_encode(repo_id)
                     + "/tree/" + url_encode(commit)
                     + "?recursive=true&expand=true";
    HttpResponse resp = http_request("GET", path);
    if (resp.status != 200)
        throw std::runtime_error("cannot list files: HTTP "
            + std::to_string(resp.status));
    Json root = Json::parse(resp.body);
    if (!root.is_array())
        throw std::runtime_error("tree API returned non-array");

    std::vector<TreeFile> files;
    for (const Json& entry : root.as_array()) {
        if (entry["type"].as_string() != "file") continue;
        TreeFile f;
        f.path = entry["path"].as_string();
        f.size = static_cast<size_t>(entry["size"].as_i64());
        f.oid = entry["oid"].as_string();
        if (entry.contains("lfs") && entry["lfs"].is_object()) {
            const Json& lfs = entry["lfs"];
            if (lfs.contains("oid")) f.lfs_oid = lfs["oid"].as_string();
        }
        files.push_back(std::move(f));
    }
    return files;
}

} // namespace

std::filesystem::path default_hf_cache_dir() {
    const char* env = std::getenv("HF_HUB_CACHE");
    if (env && *env) return std::filesystem::path(env);
    env = std::getenv("HUGGINGFACE_HUB_CACHE");
    if (env && *env) return std::filesystem::path(env);
    env = std::getenv("HF_HOME");
    if (env && *env) return std::filesystem::path(env) / "hub";
    env = std::getenv("USERPROFILE");
    if (!env || !*env) env = std::getenv("HOME");
    if (!env || !*env) env = ".";
    return std::filesystem::path(env) / ".cache" / "huggingface" / "hub";
}

DownloadResult download_model(const DownloadOptions& options) {
    if (options.repo_id.empty())
        throw std::runtime_error("repo_id is required");

    std::filesystem::path cache_dir = options.cache_dir;
    if (cache_dir.empty()) cache_dir = default_hf_cache_dir();

    const std::string folder = repo_folder_name(options.repo_id);
    const std::filesystem::path storage = cache_dir / folder;
    const std::filesystem::path blobs = storage / "blobs";
    const std::filesystem::path snapshots = storage / "snapshots";
    const std::filesystem::path refs = storage / "refs";

    std::filesystem::create_directories(blobs);
    std::filesystem::create_directories(refs);

    std::string commit = options.revision;
    if (!is_commit_hash(commit)) {
        commit = resolve_revision(options.repo_id, options.revision);
    }

    if (!options.quiet)
        std::cerr << "downloading " << options.repo_id
                  << " @ " << commit.substr(0, 12) << "\n";

    std::vector<TreeFile> all_files = list_repo_files(options.repo_id, commit);

    std::vector<TreeFile> wanted;
    if (options.files.empty()) {
        for (const auto& f : all_files) wanted.push_back(f);
    } else {
        for (const auto& name : options.files) {
            auto it = std::find_if(all_files.begin(), all_files.end(),
                [&](const TreeFile& f) { return f.path == name; });
            if (it == all_files.end())
                throw std::runtime_error("file not found in repo: " + name);
            wanted.push_back(*it);
        }
    }

    DownloadResult result;
    result.commit_hash = commit;
    result.snapshot_path = snapshots / commit;

    for (const TreeFile& tf : wanted) {
        const std::string etag = tf.lfs_oid.empty() ? tf.oid : tf.lfs_oid;
        const std::filesystem::path blob_path = blobs / etag;
        const std::filesystem::path pointer_dir =
            result.snapshot_path / std::filesystem::path(tf.path).parent_path();
        const std::filesystem::path pointer_path =
            result.snapshot_path / tf.path;

        std::filesystem::create_directories(pointer_dir);

        bool blob_exists = std::filesystem::exists(blob_path);
        if (blob_exists && !options.force) {
            if (!options.quiet)
                std::cerr << "  cached: " << tf.path
                          << " (" << format_bytes(tf.size) << ")\n";
        } else {
            if (!options.quiet)
                std::cerr << "  downloading: " << tf.path
                          << " (" << format_bytes(tf.size) << ")\n";

            std::string url_path = "/" + url_encode(options.repo_id)
                                 + "/resolve/" + url_encode(commit)
                                 + "/" + url_encode(tf.path);

            const std::filesystem::path incomplete = resume_path_for(blob_path);

            http_download_file(url_path, incomplete, tf.size, options.quiet);

            std::filesystem::rename(incomplete, blob_path);
            result.downloaded_files.push_back(tf.path);
        }

        if (!create_link_or_copy(blob_path, pointer_path)) {
            throw std::runtime_error(
                "cannot link " + blob_path.string()
                + " -> " + pointer_path.string());
        }
    }

    if (!is_commit_hash(options.revision)) {
        std::filesystem::create_directories(refs);
        std::filesystem::path ref_path = refs / options.revision;
        std::ofstream rf(ref_path, std::ios::binary | std::ios::trunc);
        if (rf) {
            rf << commit;
            rf.close();
        }
    }

    if (!options.quiet)
        std::cerr << "snapshot: " << result.snapshot_path.string() << "\n";

    return result;
}

std::filesystem::path resolve_hf_model(
    const std::string& repo_id,
    const std::string& revision,
    bool auto_download) {
    std::filesystem::path cache = default_hf_cache_dir();
    std::string folder = repo_folder_name(repo_id);
    std::filesystem::path storage = cache / folder;

    std::string commit = revision;
    if (!is_commit_hash(commit)) {
        std::filesystem::path ref_path = storage / "refs" / revision;
        std::ifstream rf(ref_path);
        if (rf) {
            std::getline(rf, commit);
            rf.close();
        }
    }

    if (is_commit_hash(commit)) {
        std::filesystem::path snap = storage / "snapshots" / commit;
        const bool has_safetensors =
            std::filesystem::exists(snap / "model.safetensors") ||
            std::filesystem::exists(snap / "model.safetensors.index.json");
        if (has_safetensors
            && std::filesystem::exists(snap / "config.json")
            && std::filesystem::exists(snap / "tokenizer.json"))
            return snap;
    }

    if (!auto_download) {
        throw std::runtime_error(
            "model not found in HF cache: " + repo_id
            + " @ " + revision);
    }

    DownloadOptions opts;
    opts.repo_id = repo_id;
    opts.revision = revision;
    DownloadResult result = download_model(opts);
    return result.snapshot_path;
}

std::filesystem::path resolve_hf_gguf(
    const std::string& repo_id,
    const std::string& quant_tag) {
    std::filesystem::path cache = default_hf_cache_dir();
    std::filesystem::path storage = cache / repo_folder_name(repo_id);
    std::error_code ec;
    for (const auto& snap_entry :
         std::filesystem::directory_iterator(storage / "snapshots", ec)) {
        if (!snap_entry.is_directory()) continue;
        const std::filesystem::path dir = snap_entry.path();
        std::vector<std::filesystem::path> candidates;
        for (const auto& f : std::filesystem::directory_iterator(dir, ec)) {
            if (f.path().extension() == ".gguf") candidates.push_back(f.path());
        }
        if (candidates.empty()) continue;
        if (!quant_tag.empty()) {
            for (const auto& c : candidates) {
                if (contains_case_insensitive(c.filename().string(), quant_tag))
                    return c;
            }
        }
        // Prefer a Q4_K_M shard, else the first .gguf found.
        for (const auto& c : candidates) {
            if (contains_case_insensitive(c.filename().string(), "Q4_K_M"))
                return c;
        }
        return candidates.front();
    }
    throw std::runtime_error(
        "GGUF checkpoint not found in HF cache: " + repo_id
        + ". Run: celeg-download " + repo_id);
}

#else // !_WIN32

std::filesystem::path default_hf_cache_dir() {
    const char* env = std::getenv("HF_HUB_CACHE");
    if (env && *env) return std::filesystem::path(env);
    env = std::getenv("HUGGINGFACE_HUB_CACHE");
    if (env && *env) return std::filesystem::path(env);
    env = std::getenv("HF_HOME");
    if (env && *env) return std::filesystem::path(env) / "hub";
    env = std::getenv("HOME");
    if (!env || !*env) env = ".";
    return std::filesystem::path(env) / ".cache" / "huggingface" / "hub";
}

DownloadResult download_model(const DownloadOptions&) {
    throw std::runtime_error(
        "native downloader is only available on Windows (WinHTTP). "
        "Use hf download or huggingface-cli instead.");
}

std::filesystem::path resolve_hf_model(
    const std::string& repo_id,
    const std::string& revision,
    bool /*auto_download*/) {
    std::filesystem::path cache = default_hf_cache_dir();
    std::filesystem::path storage = cache / repo_folder_name(repo_id);
    std::string commit = revision;

    if (!is_commit_hash(commit)) {
        std::filesystem::path ref_path = storage / "refs" / revision;
        std::ifstream rf(ref_path);
        if (rf) {
            std::getline(rf, commit);
            rf.close();
        }
    }

    if (is_commit_hash(commit)) {
        std::filesystem::path snap = storage / "snapshots" / commit;
        if (std::filesystem::exists(snap / "model.safetensors") ||
            std::filesystem::exists(snap / "model.safetensors.index.json"))
            return snap;
    }
    throw std::runtime_error(
        "model not found in HF cache: " + repo_id + " @ " + revision
        + ". Run: hf download " + repo_id);
}

std::filesystem::path resolve_hf_gguf(
    const std::string& repo_id,
    const std::string& quant_tag) {
    std::filesystem::path cache = default_hf_cache_dir();
    std::filesystem::path storage = cache / repo_folder_name(repo_id);
    std::error_code ec;
    std::filesystem::path snapshots = storage / "snapshots";
    if (!std::filesystem::exists(snapshots)) {
        throw std::runtime_error(
            "GGUF checkpoint not found (no snapshots): " + repo_id
            + ". Run: hf download " + repo_id);
    }
    for (const auto& snap_entry :
         std::filesystem::directory_iterator(snapshots, ec)) {
        if (!snap_entry.is_directory()) continue;
        const std::filesystem::path dir = snap_entry.path();
        std::vector<std::filesystem::path> candidates;
        for (const auto& f : std::filesystem::directory_iterator(dir, ec)) {
            if (f.path().extension() == ".gguf") candidates.push_back(f.path());
        }
        if (candidates.empty()) continue;
        if (!quant_tag.empty()) {
            for (const auto& c : candidates) {
                if (contains_case_insensitive(c.filename().string(), quant_tag))
                    return c;
            }
        }
        for (const auto& c : candidates) {
            if (contains_case_insensitive(c.filename().string(), "Q4_K_M"))
                return c;
        }
        return candidates.front();
    }
    throw std::runtime_error(
        "GGUF checkpoint not found in HF cache: " + repo_id
        + ". Run: celeg-download " + repo_id);
}

#endif

} // namespace celeg
