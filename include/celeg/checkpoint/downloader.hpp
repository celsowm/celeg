#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace celeg {

struct DownloadOptions {
    std::string repo_id;
    std::string revision = "main";
    std::filesystem::path cache_dir;
    std::vector<std::string> files;
    bool force = false;
    bool quiet = false;
};

struct DownloadResult {
    std::filesystem::path snapshot_path;
    std::string commit_hash;
    std::vector<std::string> downloaded_files;
};

std::filesystem::path default_hf_cache_dir();

DownloadResult download_model(const DownloadOptions& options);

std::filesystem::path resolve_hf_model(
    const std::string& repo_id,
    const std::string& revision = "main",
    bool auto_download = true);

// Resolves a GGUF checkpoint from the HuggingFace cache. `repo_id` is the
// dataset/repo id (typically ending in `-GGUF`); `quant_tag` selects which
// shard to load (e.g. "Q4_K_M"), defaulting to "Q4_K_M" when empty. The
// returned path is the concrete `*.gguf` file, ready to hand to Model.
std::filesystem::path resolve_hf_gguf(
    const std::string& repo_id,
    const std::string& quant_tag = "Q4_K_M");

} // namespace celeg
