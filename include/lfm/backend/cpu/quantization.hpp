#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace lfm {

struct Q4GroupMatrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t group_size = 32;
    uint32_t groups_per_row = 0;
    std::vector<uint8_t> values;
    std::vector<uint16_t> scales_bf16;

    size_t packed_values_per_row() const { return (static_cast<size_t>(cols) + 1) / 2; }
    size_t memory_bytes() const {
        return values.size() * sizeof(uint8_t) +
               scales_bf16.size() * sizeof(uint16_t);
    }
    void validate() const;
};

Q4GroupMatrix quantize_bf16_groupwise_q4(const std::byte* data,
                                          size_t rows,
                                          size_t cols,
                                          size_t group_size);
Q4GroupMatrix quantize_float_groupwise_q4(const float* data,
                                           size_t rows,
                                           size_t cols,
                                           size_t group_size);
void dequantize_q4_row(const Q4GroupMatrix& matrix, size_t row, float* output);

// Dynamically quantized activation used by VNNI/I8MM-style kernels. Each
// group has an independent FP32 scale and signed INT8 values.
struct Q8GroupVector {
    uint32_t elements = 0;
    uint32_t group_size = 32;
    uint32_t groups = 0;
    std::vector<int8_t> values;
    std::vector<float> scales;
    std::vector<int32_t> sums;

    void validate() const;
};

Q8GroupVector quantize_float_groupwise_q8(const float* data,
                                           size_t elements,
                                           size_t group_size);

// Quantize into caller-owned storage. Lets a batch of activation rows live in
// one contiguous allocation instead of one Q8GroupVector (three heap vectors)
// per row, which is what the GEMM path needs. `values` holds `elements` bytes;
// `scales` and `sums` hold ceil(elements / group_size) entries each.
void quantize_float_groupwise_q8_into(const float* data,
                                       size_t elements,
                                       size_t group_size,
                                       int8_t* values,
                                       float* scales,
                                       int32_t* sums);

struct CpuPackMetadata {
    uint32_t version = 2;
    std::string source_id;
    std::string isa;
    uint32_t group_size = 32;
};

class CpuPackWriter {
public:
    CpuPackWriter(const std::filesystem::path& path, CpuPackMetadata metadata);
    ~CpuPackWriter();
    CpuPackWriter(const CpuPackWriter&) = delete;
    CpuPackWriter& operator=(const CpuPackWriter&) = delete;

    void add_q4_matrix(const std::string& name, const Q4GroupMatrix& matrix);
    void add_bf16_vector(const std::string& name,
                         const std::byte* data, size_t elements);
    void commit();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class CpuPackReader {
public:
    explicit CpuPackReader(const std::filesystem::path& path);
    ~CpuPackReader();
    CpuPackReader(const CpuPackReader&) = delete;
    CpuPackReader& operator=(const CpuPackReader&) = delete;

    const CpuPackMetadata& metadata() const;
    bool contains(const std::string& name) const;
    Q4GroupMatrix read_q4_matrix(const std::string& name) const;
    std::vector<float> read_bf16_vector(const std::string& name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lfm
