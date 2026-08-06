#pragma once

#include "celeg/backend/cpu/kernels.hpp"
#include "celeg/backend/cpu/quantization.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace celeg {

class IWeightRepository;

// Converts repository tensors into the CPU backend's resident weight formats.
// Pack-file I/O is deliberately part of this boundary: model assembly should
// not know whether a tensor came from a checkpoint or the local packed cache.
class CpuWeightCodec {
public:
    CpuWeightCodec(IWeightRepository* source, CpuPackReader* reader,
                   CpuPackWriter* writer, size_t group_size);

    CpuLinearWeight matrix(const std::string& name,
                          const std::vector<int64_t>& expected) const;
    CpuLinearWeight concat(
        const std::string& synthetic,
        const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts) const;
    std::vector<float> vector(const std::string& name,
                              const std::vector<int64_t>& expected) const;

private:
    IWeightRepository* source_ = nullptr;
    CpuPackReader* reader_ = nullptr;
    CpuPackWriter* writer_ = nullptr;
    size_t group_size_ = 32;
};

} // namespace celeg
