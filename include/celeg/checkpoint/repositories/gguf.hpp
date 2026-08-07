#pragma once

#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/weight_repository.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace celeg {

class GgufTensorResolver;

// Weight repository backed by a single GGUF checkpoint file. Tensor-name
// resolution, packed-expert slicing, and native-view adaptation are delegated
// to internal GGUF components so this facade only exposes repository
// capabilities to the format-agnostic runtime.
//
// Returned HostTensorView values point directly into the memory-mapped GGUF
// file and stay valid for the lifetime of this repository. Quantized tensors
// (Q4_K/Q6_K/...) are reported with dtype == Quantized and the matching
// GgmlType; F32/F16 tensors map to the plain TensorDType values.
class GgufRepository final : public IWeightRepository,
                             public INativeBlockStorageRepository {
public:
    explicit GgufRepository(std::shared_ptr<GgufFile> gguf);
    ~GgufRepository();

    GgufRepository(const GgufRepository&) = delete;
    GgufRepository& operator=(const GgufRepository&) = delete;

    bool contains(std::string_view name) const override;
    HostTensorView tensor(std::string_view name) const override;
    std::vector<std::string> names() const override;
    bool has_native_block_storage() const override { return true; }

private:
    std::unique_ptr<GgufTensorResolver> tensors_;
};

} // namespace celeg
