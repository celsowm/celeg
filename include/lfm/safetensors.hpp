#pragma once

#include "lfm/json.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfm {

enum class TensorDType { BF16, F16, F32, I8, Unknown };

struct HostTensorView {
    TensorDType dtype = TensorDType::Unknown;
    std::vector<int64_t> shape;
    const std::byte* data = nullptr;
    size_t bytes = 0;
};

class SafeTensorFile {
public:
    explicit SafeTensorFile(const std::string& path);
    ~SafeTensorFile();

    SafeTensorFile(const SafeTensorFile&) = delete;
    SafeTensorFile& operator=(const SafeTensorFile&) = delete;

    bool contains(std::string_view name) const;
    HostTensorView tensor(std::string_view name) const;
    std::vector<std::string> names() const;

private:
    struct Entry {
        TensorDType dtype;
        std::vector<int64_t> shape;
        size_t begin;
        size_t end;
    };

    int fd_ = -1;
    void* mapping_ = nullptr;
    size_t file_size_ = 0;
    size_t data_offset_ = 0;
    std::unordered_map<std::string, Entry> entries_;
};

} // namespace lfm
