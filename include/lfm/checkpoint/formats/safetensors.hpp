#pragma once

#include "lfm/checkpoint/formats/gguf.hpp"
#include "lfm/checkpoint/formats/json.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lfm {

// Non-quantized source element types. GGUF pre-quantized tensors additionally
// carry a `ggml_type` (see below) and set `dtype == Quantized`.
enum class TensorDType { BF16, F16, F32, I8, Quantized, Unknown };

struct TensorLocator {
    std::uint32_t shard_id = 0;
    std::uint64_t absolute_offset = 0;
    std::uint64_t bytes = 0;
    TensorDType dtype = TensorDType::Unknown;
    std::vector<std::int64_t> shape;
};

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

// Optional checkpoint capability for repositories that can resolve a tensor
// to its source location. Repositories that only expose mapped tensor views
// implement IWeightRepository without pretending to support this operation.
class ILocatableTensorRepository {
public:
    virtual ~ILocatableTensorRepository() = default;
    virtual TensorLocator locate(std::string_view name) const = 0;
};

// Optional checkpoint capability for repositories that can copy source bytes
// by location. This is intentionally separate from IWeightRepository: GGUF
// and in-memory repositories do not need to inherit an operation that is
// invalid for their storage model.
class IRandomAccessTensorReader {
public:
    virtual ~IRandomAccessTensorReader() = default;
    virtual void read(const TensorLocator& locator,
                      std::span<std::byte> destination) const = 0;
};

inline const ILocatableTensorRepository&
require_locatable_tensor_repository(const IWeightRepository& repository) {
    const auto* locator = dynamic_cast<const ILocatableTensorRepository*>(&repository);
    if (locator == nullptr) {
        throw std::runtime_error(
            "checkpoint repository does not provide tensor locations");
    }
    return *locator;
}

// Capability discovery helper for cold/offloaded reads. The exception is
// raised at the boundary where the capability is required, rather than by a
// misleading default implementation on every repository.
inline const IRandomAccessTensorReader&
require_random_access_tensor_reader(const IWeightRepository& repository) {
    const auto* reader = dynamic_cast<const IRandomAccessTensorReader*>(&repository);
    if (reader == nullptr) {
        throw std::runtime_error(
            "checkpoint repository does not provide random-access tensor reads");
    }
    return *reader;
}

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

} // namespace lfm
