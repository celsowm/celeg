#pragma once

// cuBLASLt plan state.
//
// This is the narrowest layer of the CUDA model implementation headers: it
// depends only on cuBLASLt itself and knows nothing about weights, layers or
// session state. `GemmDispatcher` owns the plan cache keyed by `MatmulKey`.

#include <cublasLt.h>

#include <cstddef>

namespace celeg {

// ---------------------------------------------------------------------------
// GEMM plan keys (cuBLASLt).
// ---------------------------------------------------------------------------

struct MatmulKey {
    int m = 0;
    int n = 0;
    int k = 0;

    bool operator==(const MatmulKey& other) const {
        return m == other.m && n == other.n && k == other.k;
    }
};

struct MatmulKeyHash {
    size_t operator()(const MatmulKey& key) const {
        size_t value = static_cast<size_t>(key.m);
        value = value * 1315423911u + static_cast<size_t>(key.n);
        value = value * 2654435761u + static_cast<size_t>(key.k);
        return value;
    }
};

// Cached cuBLASLt matmul plan for one (m, n, k) shape. Holds the operation
// descriptor, matrix layouts, selected algorithm, and workspace size.
// Destroyed via RAII in the dtor.
struct LtPlan {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatrixLayout_t d = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_size = 0;
    bool available = false;

    ~LtPlan() {
        if (d) cublasLtMatrixLayoutDestroy(d);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
};

} // namespace celeg
