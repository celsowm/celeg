#pragma once

#include "celeg/checkpoint/packed/fp8.hpp"
#include "celeg/checkpoint/packed/int4.hpp"
#include "celeg/checkpoint/packed/int8.hpp"
#include "celeg/checkpoint/packed/nvfp4.hpp"
#include "celeg/checkpoint/tensor.hpp"
#include "celeg/checkpoint/tensor_codec.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
#include "backend/cuda/runtime_types.hpp"
#include "detail/device_weights.hpp"

#include <optional>
#include <span>
#include <string_view>
#include <string>
#include <variant>

namespace celeg {

struct PackedInt8Source {
    PackedInt8Matrix matrix;
};

struct PackedInt4Source {
    PackedInt4Matrix matrix;
};

struct PackedFp8Source {
    PackedFp8Matrix matrix;
};

struct PackedNvfp4Source {
    PackedNvfp4Matrix matrix;
};

struct GgufSource {
    HostTensorView tensor;
};

struct DenseSource {
    HostTensorView tensor;
};

using LinearSource = std::variant<
    PackedInt8Source,
    PackedInt4Source,
    PackedFp8Source,
    PackedNvfp4Source,
    GgufSource,
    DenseSource>;

std::optional<LinearSource> classify_linear_source(
    const IWeightRepository& repository,
    std::string_view name,
    std::span<const std::int64_t> expected);

struct MaterializationPlan {
    LinearSource source;
    WeightMode mode = WeightMode::Bf16;
    std::string name;
    int rows = 0;
    int cols = 0;
    bool native_gguf = false;
    bool host_dequantization = false;
};

MaterializationPlan plan_linear_materialization(
    const LinearSource& source,
    WeightMode mode,
    std::string_view name,
    int rows,
    int cols);

DeviceWeight materialize_linear(const MaterializationPlan& plan,
                                CudaMemoryKind memory_kind);

}
