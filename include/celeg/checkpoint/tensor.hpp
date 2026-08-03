#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace celeg {

// Non-quantized source element types. Pre-quantized tensors additionally
// carry a `block_encoding` (see below) and set `dtype == Quantized`.
enum class TensorDType { BF16, F16, F32, I8, Quantized, Unknown };

// Format-neutral identifier for an on-disk quantized block encoding. This
// header must not depend on any concrete checkpoint format: the owning
// format module (e.g. the GGUF module) defines the mapping between this id
// and its own concrete block-type enum, and translates in both directions at
// its boundary. `id == 0` means "no block encoding" (dtype != Quantized).
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
    // Set only when dtype == Quantized (e.g. a GGUF source). Identifies the
    // on-disk block quantization so the loader can upload the packed blocks
    // verbatim. The owning checkpoint format module maps this to/from its
    // concrete block-type enum; this header stays neutral.
    TensorBlockEncoding block_encoding;
    // GGUF attention Q/K rows can use the source format's RoPE layout.
    // Format repositories set this source-layout fact; backends consume it
    // without branching on an architecture or checkpoint implementation.
    bool rows_rope_permuted = false;
};

} // namespace celeg
