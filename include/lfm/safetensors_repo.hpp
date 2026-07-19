#pragma once

#include "lfm/safetensors.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfm {

// Resolves tensors to their source shard for checkpoints that ship split
// across multiple safetensors files (a `model.safetensors.index.json`
// weight map). Also supports a single `model.safetensors` file unchanged.
//
// Shards are memory-mapped lazily and kept alive for the lifetime of the
// repository so HostTensorView values returned by tensor() remain valid.
class SafeTensorRepository {
public:
    // `model_dir` may be:
    //   * a directory containing `model.safetensors.index.json` (sharded),
    //   * a directory containing `model.safetensors` (single file), or
    //   * a single `.safetensors` file directly.
    explicit SafeTensorRepository(const std::filesystem::path& model_dir);

    bool contains(std::string_view name) const;
    HostTensorView tensor(std::string_view name) const;
    std::vector<std::string> names() const;

    // True when the checkpoint is split across multiple shard files.
    bool sharded() const { return sharded_; }

private:
    const SafeTensorFile& shard_for(const std::string& shard_filename) const;

    std::filesystem::path dir_;
    bool sharded_ = false;
    // tensor name -> shard filename (relative to dir_). Empty in single-file mode.
    std::unordered_map<std::string, std::string> name_to_shard_;
    // Single-file mode handle.
    std::unique_ptr<SafeTensorFile> single_file_;
    // Lazily opened, kept-alive shard files keyed by shard filename.
    mutable std::unordered_map<std::string, std::unique_ptr<SafeTensorFile>> shards_;
};

} // namespace lfm
