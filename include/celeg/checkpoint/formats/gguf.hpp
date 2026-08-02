#pragma once

#include "celeg/checkpoint/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace celeg {

// GGML tensor element types (subset of llama.cpp's ggml_type). Only the types
// that appear in LFM2.5 GGUF checkpoints are enumerated; everything else is
// rejected at parse time. Values match the upstream ggml enum ordinals so the
// on-disk type ids map directly.
enum class GgmlType : int32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    Q4_K = 12,
    Q6_K = 14,
    BF16 = 30,
    Unknown = -1,
};

// GGUF metadata value discriminator (mirrors gguf_metadata_value_type).
enum class GgufValueKind : uint32_t {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
};

// One metadata value. Scalars are stored in the union-like fields; strings in
// `str`; arrays retain their element kind plus the raw element payloads. Array
// scalar elements are stored little-endian packed in `array_data`; array string
// elements are stored decoded in `array_strings`.
struct GgufValue {
    GgufValueKind kind = GgufValueKind::U32;
    // Scalar storage (only the field matching `kind` is meaningful).
    double number = 0.0;   // holds F32/F64
    int64_t integer = 0;   // holds any integer / bool
    std::string str;       // holds String
    // Array storage.
    GgufValueKind array_kind = GgufValueKind::U32;
    std::vector<int64_t> array_integers;   // integer/bool arrays
    std::vector<double> array_numbers;     // f32/f64 arrays
    std::vector<std::string> array_strings; // string arrays
};

// Tensor info from the GGUF tensor directory. `dims` is stored in GGUF order
// (fastest-varying first == [cols, rows, ...]); `hf_shape()` returns the
// reversed HuggingFace convention [rows, cols, ...].
struct GgufTensorInfo {
    std::string name;
    GgmlType type = GgmlType::Unknown;
    std::vector<uint64_t> dims;   // GGUF order
    uint64_t offset = 0;          // relative to the tensor data section

    std::vector<int64_t> hf_shape() const;
    uint64_t element_count() const;
};

// Number of bytes a ggml block occupies and how many logical elements it packs.
struct GgmlTypeTrait {
    int block_size = 0;   // logical elements per block
    int type_size = 0;    // bytes per block
};

GgmlTypeTrait ggml_type_trait(GgmlType type);
const char* ggml_type_name(GgmlType type);

// The GGUF module owns the mapping between the neutral TensorBlockEncoding
// descriptor (celeg/checkpoint/tensor.hpp) and this module's concrete
// GgmlType enum. Values round-trip exactly: GgmlType's ordinals already match
// the on-disk ggml type ids, so the neutral descriptor simply carries that id.
inline TensorBlockEncoding block_encoding_from_ggml_type(GgmlType type) {
    return TensorBlockEncoding{static_cast<std::int32_t>(type)};
}
inline GgmlType ggml_type_from_block_encoding(const TensorBlockEncoding& encoding) {
    return static_cast<GgmlType>(encoding.id);
}

// Raw view into a GGUF tensor's on-disk (memory-mapped) bytes. `data` points
// into the mapping and stays valid for the lifetime of the owning GgufFile.
struct GgufTensorView {
    GgmlType type = GgmlType::Unknown;
    std::vector<int64_t> shape;   // HuggingFace order [rows, cols, ...]
    const std::byte* data = nullptr;
    size_t bytes = 0;
    uint64_t element_count = 0;
};

// Memory-maps and parses a GGUF v2/v3 container: the header, the full key/value
// metadata map, and the tensor directory. Tensor payloads are not copied; use
// tensor()/raw() to obtain zero-copy views into the mapping.
class GgufFile {
public:
    explicit GgufFile(const std::string& path);
    ~GgufFile();

    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    // Metadata access.
    bool has(std::string_view key) const;
    const GgufValue& value(std::string_view key) const;
    const std::unordered_map<std::string, GgufValue>& metadata() const { return kv_; }

    uint32_t u32(std::string_view key) const;
    uint64_t u64(std::string_view key) const;
    int64_t i64(std::string_view key) const;
    float f32(std::string_view key) const;
    bool boolean(std::string_view key) const;
    const std::string& str(std::string_view key) const;

    // Optional scalar getters returning a default when the key is absent.
    uint32_t u32_or(std::string_view key, uint32_t fallback) const;
    float f32_or(std::string_view key, float fallback) const;
    bool boolean_or(std::string_view key, bool fallback) const;
    std::string str_or(std::string_view key, const std::string& fallback) const;

    // Tensor access.
    bool contains_tensor(std::string_view name) const;
    GgufTensorView tensor(std::string_view name) const;
    const GgufTensorInfo& tensor_info(std::string_view name) const;
    std::vector<std::string> tensor_names() const;
    size_t tensor_count() const { return tensors_.size(); }

    uint32_t version() const { return version_; }

private:
    void parse();

    int fd_ = -1;
    void* mapping_ = nullptr;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#endif
    size_t file_size_ = 0;
    uint32_t version_ = 0;
    size_t tensor_data_offset_ = 0;  // absolute file offset of tensor payloads

    std::unordered_map<std::string, GgufValue> kv_;
    std::unordered_map<std::string, GgufTensorInfo> tensors_;
};

} // namespace celeg
