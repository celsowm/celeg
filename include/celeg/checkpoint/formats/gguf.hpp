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
    IQ3_XXS = 18,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
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
    /// The type ordinal exactly as the file carried it. Retained so an
    /// unrecognised quantization can be reported by number instead of as an
    /// anonymous "Unknown", which is the difference between a one-step and a
    /// several-step diagnosis.
    int32_t raw_type = -1;
    std::vector<uint64_t> dims;
    uint64_t offset = 0;

    std::vector<int64_t> hf_shape() const;
    uint64_t element_count() const;
};

struct GgmlTypeTrait {
    int block_size = 0;
    int type_size = 0;
};

/// Which backend paths can actually consume a GGUF block type.
///
/// This is the single source of truth for every "is this quantization
/// supported here?" question in the engine. Backends must query it rather
/// than inlining their own type lists: the CUDA loader previously carried
/// three independent hardcoded chains that drifted apart from the CPU's
/// (see docs/inference_report.md), which is how Q4_1 ended up loading on CPU
/// and failing on GPU for the same file.
///
/// A flag is set only when the corresponding decoder or kernel exists and is
/// covered by a test; docs/QUANTIZATION_SUPPORT_MATRIX.md records the
/// evidence per cell.
struct GgufTypeSupport {
    /// A dequantizer exists in the CPU kernels (cpu_gguf_dequantize_row).
    bool cpu_dequantize = false;
    /// cpu_gguf_dot_scalar can consume the packed blocks directly, so the
    /// weight codec keeps them packed instead of repacking to groupwise Q4.
    bool cpu_native_dot = false;
    /// A host dequantizer exists for the CUDA loader (dequantize_gguf_to_bf16).
    bool cuda_dequantize = false;
    /// A native CUDA MMQ kernel exists, so WeightMode::NativeGguf can keep
    /// the packed blocks resident on the device.
    bool cuda_native_mmq = false;
};

GgmlTypeTrait ggml_type_trait(GgmlType type);
const char* ggml_type_name(GgmlType type);
GgufTypeSupport ggml_type_support(GgmlType type);

/// Maps a raw GGUF type ordinal onto the enum, or Unknown when celeg has no
/// entry for it. Kept public so diagnostics can report the ordinal a file
/// actually carried.
GgmlType ggml_type_from_ordinal(int32_t raw);

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
