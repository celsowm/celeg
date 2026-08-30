#pragma once

#include "packed/executor.hpp"
#include "packed/session.hpp"

#include <cstdint>
#include <span>

namespace celeg {

void commit_packed_decode(std::span<const PackedSessionContext> sessions,
                          std::span<const int32_t> sampled,
                          double gpu_elapsed_ms,
                          std::span<int32_t> output);

void commit_packed_prefill(std::span<const PackedSessionContext> sessions,
                           std::span<const int32_t> explicit_tokens,
                           std::span<const int32_t> final_rows,
                           std::span<const PackedPrefillRow> rows);

}
