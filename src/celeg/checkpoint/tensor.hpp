#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace celeg {

enum class TensorDType { BF16, F16, F32, I8, U8, I32, I64, F8_E4M3, Quantized, Unknown };

struct TensorBlockEncoding {
    std::int32_t id = 0;

    friend bool operator==(const TensorBlockEncoding&, const TensorBlockEncoding&) = default;
};

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
    TensorBlockEncoding block_encoding;
};

}
