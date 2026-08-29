#pragma once

#include <cstdint>

namespace celeg {

enum class LinearOperationKind : std::uint8_t {
    MatVec,
    MatVecPair,
    MatMul,
    MatMulPair,
    MatMulTensor,
    Embedding,
    EmbeddingBatch,
    SwiGluMatVec,
};

struct LinearOperation {
    LinearOperationKind kind = LinearOperationKind::MatVec;
    std::uint32_t input_features = 0;
    std::uint32_t output_features = 0;
};

}
