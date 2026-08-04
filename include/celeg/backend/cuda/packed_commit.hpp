#pragma once

#include "celeg/backend/cuda/packed.hpp"
#include "celeg/backend/cuda/packed_session.hpp"

#include <cstdint>
#include <span>

namespace celeg {

// Applies host-visible state only after the caller has observed successful GPU
// completion. These functions intentionally contain no CUDA launches, so the
// commit boundary can be reviewed and tested independently of execution.
void commit_packed_decode(std::span<const PackedSessionContext> sessions,
                          std::span<const int32_t> sampled,
                          double gpu_elapsed_ms,
                          std::span<int32_t> output);

void commit_packed_prefill(std::span<const PackedSessionContext> sessions,
                           std::span<const int32_t> explicit_tokens,
                           std::span<const int32_t> final_rows,
                           std::span<const PackedPrefillRow> rows);

} // namespace celeg
