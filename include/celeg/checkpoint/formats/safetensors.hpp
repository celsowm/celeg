#pragma once

#include "celeg/checkpoint/weight_repository.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <span>
#include <unordered_map>
#include <vector>

namespace celeg {

class SafeTensorFile {
public:
    explicit SafeTensorFile(const std::string& path);
    ~SafeTensorFile();

    SafeTensorFile(const SafeTensorFile&) = delete;
    SafeTensorFile& operator=(const SafeTensorFile&) = delete;

    bool contains(std::string_view name) const;
    HostTensorView tensor(std::string_view name) const;
    std::vector<std::string> names() const;

    TensorLocator locate(std::string_view name, std::uint32_t shard_id = 0) const;
    void read(const TensorLocator& locator, std::span<std::byte> destination) const;

private:
    struct Entry {
        TensorDType dtype;
        std::vector<int64_t> shape;
        size_t begin;
        size_t end;
    };

    int fd_ = -1;
    void* mapping_ = nullptr;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#endif
    size_t file_size_ = 0;
    size_t data_offset_ = 0;
    std::unordered_map<std::string, Entry> entries_;
};

} // namespace celeg
