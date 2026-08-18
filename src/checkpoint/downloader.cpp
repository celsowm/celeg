#include "celeg/checkpoint/downloader.hpp"
#include "celeg/checkpoint/formats/json.hpp"
#include "hf_http.hpp"

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

}

#if defined(_WIN32)

namespace {

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
    std::string path = "/api/models/" + hf_internal::url_encode(repo_id)
                     + "/revision/" + hf_internal::url_encode(revision);
    hf_internal::HttpResponse resp = hf_internal::http_request("GET", path);
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
    std::string path = "/api/models/" + hf_internal::url_encode(repo_id)
                     + "/tree/" + hf_internal::url_encode(commit)
                     + "?recursive=true&expand=true";
    hf_internal::HttpResponse resp = hf_internal::http_request("GET", path);
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

}

#else  // !_WIN32

namespace {

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
    std::string path = "/api/models/" + hf_internal::url_encode(repo_id)
                     + "/revision/" + hf_internal::url_encode(revision);
    hf_internal::HttpResponse resp = hf_internal::http_request("GET", path);
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
    std::string path = "/api/models/" + hf_internal::url_encode(repo_id)
                     + "/tree/" + hf_internal::url_encode(commit)
                     + "?recursive=true&expand=true";
    hf_internal::HttpResponse resp = hf_internal::http_request("GET", path);
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

}

#endif

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

            std::string url_path = "/" + hf_internal::url_encode(options.repo_id)
                                 + "/resolve/" + hf_internal::url_encode(commit)
                                 + "/" + hf_internal::url_encode(tf.path);

            const std::filesystem::path incomplete = resume_path_for(blob_path);

            hf_internal::http_download_file(url_path, incomplete, tf.size,
                                            options.quiet);

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


}
