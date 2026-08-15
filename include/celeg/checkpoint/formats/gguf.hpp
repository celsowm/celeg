#pragma once

#include "celeg/checkpoint/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace celeg {

enum class GgmlType : int32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    BF16 = 30,
    Unknown = -1,
};

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

struct GgufValue {
    GgufValueKind kind = GgufValueKind::U32;
    double number = 0.0;
    int64_t integer = 0;
    std::string str;
    GgufValueKind array_kind = GgufValueKind::U32;
    std::vector<int64_t> array_integers;
    std::vector<double> array_numbers;
    std::vector<std::string> array_strings;
};

struct GgufTensorInfo {
    std::string name;
    GgmlType type = GgmlType::Unknown;
    std::vector<uint64_t> dims;
    uint64_t offset = 0;

    std::vector<int64_t> hf_shape() const;
    uint64_t element_count() const;
};

struct GgmlTypeTrait {
    int block_size = 0;
    int type_size = 0;
};

GgmlTypeTrait ggml_type_trait(GgmlType type);
const char* ggml_type_name(GgmlType type);

inline TensorBlockEncoding block_encoding_from_ggml_type(GgmlType type) {
    return TensorBlockEncoding{static_cast<std::int32_t>(type)};
}
inline GgmlType ggml_type_from_block_encoding(const TensorBlockEncoding& encoding) {
    return static_cast<GgmlType>(encoding.id);
}

struct GgufTensorView {
    GgmlType type = GgmlType::Unknown;
    std::vector<int64_t> shape;
    const std::byte* data = nullptr;
    size_t bytes = 0;
    uint64_t element_count = 0;
};

class GgufFile {
public:
    explicit GgufFile(const std::string& path);
    ~GgufFile();

    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    bool has(std::string_view key) const;
    const GgufValue& value(std::string_view key) const;
    const std::unordered_map<std::string, GgufValue>& metadata() const { return kv_; }

    uint32_t u32(std::string_view key) const;
    uint64_t u64(std::string_view key) const;
    int64_t i64(std::string_view key) const;
    float f32(std::string_view key) const;
    bool boolean(std::string_view key) const;
    const std::string& str(std::string_view key) const;

    uint32_t u32_or(std::string_view key, uint32_t fallback) const;
    float f32_or(std::string_view key, float fallback) const;
    bool boolean_or(std::string_view key, bool fallback) const;
    std::string str_or(std::string_view key, const std::string& fallback) const;

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
    size_t tensor_data_offset_ = 0;

    std::unordered_map<std::string, GgufValue> kv_;
    std::unordered_map<std::string, GgufTensorInfo> tensors_;
};

}
