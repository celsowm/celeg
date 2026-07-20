#pragma once

#include "lfm/gguf.hpp"
#include "lfm/json.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfm {

// Non-quantized source element types. GGUF pre-quantized tensors additionally
// carry a `ggml_type` (see below) and set `dtype == Quantized`.
enum class TensorDType { BF16, F16, F32, I8, Quantized, Unknown };

struct HostTensorView {
    TensorDType dtype = TensorDType::Unknown;
    std::vector<int64_t> shape;
    const std::byte* data = nullptr;
    size_t bytes = 0;
    // Set only when dtype == Quantized (GGUF source). Identifies the on-disk
    // block quantization so the loader can upload the packed blocks verbatim.
    GgmlType ggml_type = GgmlType::Unknown;
};

// Abstract weight source. Both the safetensors repository and the GGUF
// repository implement this so the weight loader and model builder stay
// format-agnostic. Implementations translate their native tensor naming to the
// canonical HuggingFace `model.*` scheme used by the model builder.
class IWeightRepository {
public:
    virtual ~IWeightRepository() = default;
    virtual bool contains(std::string_view name) const = 0;
    virtual HostTensorView tensor(std::string_view name) const = 0;
    virtual std::vector<std::string> names() const = 0;
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
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#endif
    size_t file_size_ = 0;
    size_t data_offset_ = 0;
    std::unordered_map<std::string, Entry> entries_;
};

} // namespace lfm
