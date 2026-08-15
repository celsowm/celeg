#pragma once


#include <cublasLt.h>

#include <cstddef>

namespace celeg {


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

}
