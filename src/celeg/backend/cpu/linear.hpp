#pragma once

#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/backend/cpu/quantization.hpp"
#include "celeg/backend/cpu/quantized_dot.hpp"
#include "celeg/backend/cpu/thread_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace celeg {

struct CpuKernelBackend;

struct CpuGroupedGemmJob {
    const CpuLinearWeight* weight = nullptr;
    size_t row_offset = 0;
    size_t rows = 0;
};

class CpuLinearEngine {
public:
    CpuLinearEngine(const CpuKernelBackend& backend, CpuThreadPool& pool);

    CpuIsa isa() const { return isa_; }
    void gemv(const Q4GroupMatrix& weight, const float* input, float* output,
              float beta = 0.0f) const;
    void gemm(const Q4GroupMatrix& weight, const float* input, float* output,
              size_t rows, float beta = 0.0f) const;
    void embedding(const Q4GroupMatrix& table, int32_t token, float* output) const;
    void gemv(const CpuLinearWeight& weight, const float* input, float* output,
              float beta = 0.0f) const;
    void gemv_transpose(const CpuLinearWeight& weight, const float* input,
                        float* output, size_t row_offset = 0,
                        size_t row_count = 0) const;
    /// Row-sliced GEMV: `output[j] = W[row_offset + j][:] · input` for
    /// `j < row_count`. The transposed variant above projects column-space
    /// sums; attention value decompression needs the plain row slice.
    void gemv_rows(const CpuLinearWeight& weight, const float* input,
                   float* output, size_t row_offset, size_t row_count) const;
    void gemm(const CpuLinearWeight& weight, const float* input, float* output,
              size_t rows, float beta = 0.0f) const;
    void prepare_gguf_activation(const float* input, size_t rows, size_t cols,
                                 std::vector<CpuQ8KBlock>& activation) const;
    void gemm_gguf(std::span<const CpuQ8KBlock> activation,
                   const CpuLinearWeight& weight, float* output,
                   size_t rows, float beta = 0.0f) const;
    void gemm_grouped(std::span<const CpuGroupedGemmJob> jobs,
                      const float* input, float* output) const;
    void embedding(const CpuLinearWeight& table, int32_t token,
                   float* output) const;
    void gemv_raw(const float* weight, const float* input, float* output,
                  int n, int k) const;
    void gemm_raw(const float* weight, const float* input, float* output,
                  size_t rows, int n, int k) const;

private:
    void gemv_int8(const CpuInt8Matrix& matrix, const float* input, float* output,
                   float beta) const;
    void gemv_gguf(const GgmlMatrixView& matrix,
                   std::span<const CpuQ8KBlock> activation,
                   float* output, float beta) const;
    void gemm_int8(const CpuInt8Matrix& matrix, const float* input, float* output,
                   size_t rows, float beta, size_t output_stride,
                   size_t output_base) const;
    void gemv_bf16(const CpuBf16Matrix& matrix, const float* input, float* output,
                   float beta) const;
    void gemm_bf16(const CpuBf16Matrix& matrix, const float* input, float* output,
                   size_t rows, float beta, size_t output_stride,
                   size_t output_base) const;

    CpuIsa isa_;
    CpuThreadPool* pool_;
    Q4DotFunction dot_;
    Q4Q8DotFunction q8_dot_ = nullptr;
    CpuGgufDotFunction gguf_dot_ = nullptr;
    CpuGgufDot4Function gguf_dot4_ = nullptr;
    bool dynamic_q8_ = false;
};

}
