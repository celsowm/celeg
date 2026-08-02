#pragma once

#include "celeg/backend/cuda/utils.cuh"

namespace celeg {

// Owns the two specialized decode graph instances. Selection remains a
// resolved boolean in the caller, so graph choice never performs architecture
// or format dispatch in the hot path.
class CudaDecodeGraphs final {
public:
    CudaGraphExec& select(bool segmented) {
        return segmented ? segmented_decode : regular_decode;
    }

    const CudaGraphExec& select(bool segmented) const {
        return segmented ? segmented_decode : regular_decode;
    }

    bool ready() const {
        return regular_decode.ready() || segmented_decode.ready();
    }

    void reset() {
        regular_decode.reset();
        segmented_decode.reset();
    }

    CudaGraphExec regular_decode;
    CudaGraphExec segmented_decode;
};

} // namespace celeg
