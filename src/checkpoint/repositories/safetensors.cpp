#include "lfm/checkpoint/repositories/safetensors.hpp"
#include "lfm/checkpoint/formats/json.hpp"

#include <algorithm>
#include <stdexcept>

namespace lfm {
namespace {

std::filesystem::path resolve_model_dir(const std::filesystem::path& model_dir) {
    if (std::filesystem::is_regular_file(model_dir)) {
        return model_dir;
    }
    if (!std::filesystem::is_directory(model_dir)) {
        throw std::runtime_error(
            "safetensors repository path is neither a file nor a directory: " +
            model_dir.string());
    }
    return model_dir;
}

} // namespace

SafeTensorRepository::SafeTensorRepository(const std::filesystem::path& model_dir) {
    const std::filesystem::path root = resolve_model_dir(model_dir);
    dir_ = root;

    const std::filesystem::path index_path = root / "model.safetensors.index.json";
    const std::filesystem::path single_path = root / "model.safetensors";

    if (std::filesystem::is_regular_file(index_path)) {
        sharded_ = true;
        const Json index = Json::parse_file(index_path.string());
        if (!index.contains("weight_map")) {
            throw std::runtime_error(
                "safetensors index is missing weight_map: " + index_path.string());
        }
        const Json& weight_map = index["weight_map"];
        if (!weight_map.is_object()) {
            throw std::runtime_error("safetensors weight_map is not an object");
        }
        for (const auto& [tensor_name, shard_value] : weight_map.as_object()) {
            if (!shard_value.is_string()) {
                throw std::runtime_error(
                    "safetensors weight_map entry is not a string: " + tensor_name);
            }
            const std::string shard_filename = shard_value.as_string();
            auto [_, inserted] = name_to_shard_.emplace(tensor_name, shard_filename);
            if (!inserted) {
                throw std::runtime_error(
                    "duplicate tensor mapping in safetensors index: " + tensor_name);
            }
            const std::filesystem::path shard_path = dir_ / shard_filename;
            if (!std::filesystem::is_regular_file(shard_path)) {
                throw std::runtime_error(
                    "missing safetensors shard: " + shard_path.string());
            }

            // Populate the unique shard list
            if (shard_filename_to_id_.find(shard_filename) == shard_filename_to_id_.end()) {
                std::uint32_t new_id = static_cast<std::uint32_t>(shard_filenames_.size());
                shard_filenames_.push_back(shard_filename);
                shard_filename_to_id_.emplace(shard_filename, new_id);
            }
        }
        if (name_to_shard_.empty()) {
            throw std::runtime_error("safetensors index references no tensors");
        }
    } else if (std::filesystem::is_regular_file(single_path)) {
        sharded_ = false;
        single_file_ = std::make_unique<SafeTensorFile>(single_path.string());
        shard_filenames_.push_back(single_path.filename().string());
        shard_filename_to_id_.emplace(single_path.filename().string(), 0);
    } else if (std::filesystem::is_regular_file(root)) {
        // A single .safetensors file passed directly.
        sharded_ = false;
        single_file_ = std::make_unique<SafeTensorFile>(root.string());
        shard_filenames_.push_back(root.filename().string());
        shard_filename_to_id_.emplace(root.filename().string(), 0);
    } else {
        throw std::runtime_error(
            "no model.safetensors or model.safetensors.index.json found in " +
            root.string());
    }
}

bool SafeTensorRepository::contains(std::string_view name) const {
    if (!sharded_) {
        return single_file_ && single_file_->contains(name);
    }
    return name_to_shard_.find(std::string(name)) != name_to_shard_.end();
}

const SafeTensorFile& SafeTensorRepository::shard_for(
    const std::string& shard_filename) const {
    auto it = shards_.find(shard_filename);
    if (it != shards_.end()) {
        return *it->second;
    }
    auto file = std::make_unique<SafeTensorFile>((dir_ / shard_filename).string());
    const SafeTensorFile& ref = *file;
    shards_.emplace(shard_filename, std::move(file));
    return ref;
}

HostTensorView SafeTensorRepository::tensor(std::string_view name) const {
    if (!sharded_) {
        if (!single_file_) {
            throw std::out_of_range("safetensors repository has no tensors loaded");
        }
        return single_file_->tensor(name);
    }
    const auto it = name_to_shard_.find(std::string(name));
    if (it == name_to_shard_.end()) {
        throw std::out_of_range("missing tensor in safetensors index: " +
                                std::string(name));
    }
    return shard_for(it->second).tensor(name);
}

std::vector<std::string> SafeTensorRepository::names() const {
    if (!sharded_) {
        return single_file_ ? single_file_->names() : std::vector<std::string>{};
    }
    std::vector<std::string> result;
    result.reserve(name_to_shard_.size());
    for (const auto& [name, _] : name_to_shard_) result.push_back(name);
    std::sort(result.begin(), result.end());
    return result;
}

TensorLocator SafeTensorRepository::locate(std::string_view name) const {
    if (!sharded_) {
        if (!single_file_) {
            throw std::out_of_range("safetensors repository has no tensors loaded");
        }
        return single_file_->locate(name, 0);
    }
    const auto it = name_to_shard_.find(std::string(name));
    if (it == name_to_shard_.end()) {
        throw std::out_of_range("missing tensor in safetensors index: " +
                                std::string(name));
    }
    const std::string& shard_filename = it->second;
    std::uint32_t shard_id = shard_filename_to_id_.at(shard_filename);
    return shard_for(shard_filename).locate(name, shard_id);
}

void SafeTensorRepository::read(const TensorLocator& locator, std::span<std::byte> destination) const {
    if (locator.shard_id >= shard_filenames_.size()) {
        throw std::out_of_range("read: invalid shard_id " + std::to_string(locator.shard_id));
    }
    if (!sharded_) {
        if (!single_file_) {
            throw std::out_of_range("safetensors repository has no tensors loaded");
        }
        single_file_->read(locator, destination);
        return;
    }
    const std::string& shard_filename = shard_filenames_[locator.shard_id];
    shard_for(shard_filename).read(locator, destination);
}

std::filesystem::path SafeTensorRepository::shard_path(std::uint32_t shard_id) const {
    if (shard_id >= shard_filenames_.size()) {
        throw std::out_of_range("shard_path: invalid shard_id: " + std::to_string(shard_id));
    }
    if (!sharded_) {
        if (std::filesystem::is_regular_file(dir_)) {
            return dir_;
        }
        return dir_ / "model.safetensors";
    }
    return dir_ / shard_filenames_[shard_id];
}

} // namespace lfm
