#include "linear_materialization.hpp"

#include "celeg/checkpoint/tensor_codec.hpp"

namespace celeg {

std::optional<LinearSource> classify_linear_source(
    const IWeightRepository& repository,
    std::string_view name,
    std::span<const std::int64_t> expected) {
    const std::vector<std::int64_t> expected_vector(expected.begin(), expected.end());
    if (has_packed_int8_matrix(repository, name)) {
        return PackedInt8Source{
            load_packed_int8_matrix(repository, name, expected_vector)};
    }
    if (has_packed_int4_matrix(repository, name)) {
        return PackedInt4Source{
            load_packed_int4_matrix(repository, name, expected_vector)};
    }
    if (has_packed_fp8_matrix(repository, name)) {
        return PackedFp8Source{
            load_packed_fp8_matrix(repository, name, expected_vector)};
    }
    if (has_packed_nvfp4_matrix(repository, name)) {
        return PackedNvfp4Source{
            load_packed_nvfp4_matrix(repository, name, expected_vector)};
    }
    const HostTensorView tensor = repository.tensor(name);
    if (!expected.empty() && !tensor_shape_matches(tensor.shape, expected)) {
        return std::nullopt;
    }
    if (tensor.dtype == TensorDType::Quantized) return GgufSource{tensor};
    if (tensor.dtype == TensorDType::BF16 || tensor.dtype == TensorDType::F16 ||
        tensor.dtype == TensorDType::F32) {
        return DenseSource{tensor};
    }
    return std::nullopt;
}

}
