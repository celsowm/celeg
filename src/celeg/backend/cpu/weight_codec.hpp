#pragma once

#include "celeg/backend/cpu/linear.hpp"
#include "celeg/backend/cpu/quantization.hpp"
#include "celeg/backend/cpu/runtime_types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace celeg {

class IWeightRepository;

class CpuWeightCodec {
public:
    CpuWeightCodec(IWeightRepository* source, CpuPackReader* reader,
                   CpuPackWriter* writer, CpuWeightFormat format);

    CpuLinearWeight matrix(const std::string& name,
                          const std::vector<int64_t>& expected) const;
    CpuLinearWeight concat(
        const std::string& synthetic,
        const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts) const;
    std::vector<CpuLinearWeight> packed_matrices(
        const std::string& name, const std::vector<int64_t>& expected) const;
    std::vector<float> vector(const std::string& name,
                              const std::vector<int64_t>& expected) const;

private:
    IWeightRepository* source_ = nullptr;
    CpuPackReader* reader_ = nullptr;
    CpuPackWriter* writer_ = nullptr;
    size_t group_size_ = 32;
    bool bf16_ = false;

    CpuLinearWeight dense_result(std::vector<float> values, uint32_t rows,
                                 uint32_t cols, const std::string& name) const;
};

}
