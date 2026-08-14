#pragma once

#include "celeg/backend/cuda/utils.cuh"

#include <utility>

namespace celeg {

// Owns the specialized decode graph instances and their capture lifecycle.
// The compiled model supplies the operation to capture; this collaborator
// guarantees begin/end/abort symmetry and keeps graph-state transitions out of
// the model coordinator.
class CudaDecodeGraphs final {
public:
    CudaGraphExec& select(bool segmented) {
        return segmented ? segmented_decode : regular_decode;
    }

    const CudaGraphExec& select(bool segmented) const {
        return segmented ? segmented_decode : regular_decode;
    }

    template <typename Capture>
    void capture_if_needed(bool segmented, cudaStream_t stream, Capture&& capture) {
        CudaGraphExec& graph = select(segmented);
        if (graph.ready()) return;
        graph.capture_begin(stream);
        try {
            std::forward<Capture>(capture)();
            graph.capture_end(stream);
        } catch (...) {
            graph.abort_capture(stream);
            throw;
        }
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
