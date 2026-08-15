#pragma once

#include "celeg/checkpoint/formats/safetensors.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace celeg {

class SafeTensorRepository final
    : public IWeightRepository,
      public ILocatableTensorRepository,
      public IRandomAccessTensorReader {
public:
    explicit SafeTensorRepository(const std::filesystem::path& model_dir);

    bool contains(std::string_view name) const override;
    HostTensorView tensor(std::string_view name) const override;
    std::vector<std::string> names() const override;

    TensorLocator locate(std::string_view name) const override;
    void read(const TensorLocator& locator,
              std::span<std::byte> destination) const override;

    bool sharded() const { return sharded_; }

    std::filesystem::path shard_path(std::uint32_t shard_id) const;

private:
    const SafeTensorFile& shard_for(const std::string& shard_filename) const;

    std::filesystem::path dir_;
    bool sharded_ = false;
    std::unordered_map<std::string, std::string> name_to_shard_;
    std::unique_ptr<SafeTensorFile> single_file_;
    mutable std::unordered_map<std::string, std::unique_ptr<SafeTensorFile>> shards_;

    std::vector<std::string> shard_filenames_;
    std::unordered_map<std::string, std::uint32_t> shard_filename_to_id_;
};

}
