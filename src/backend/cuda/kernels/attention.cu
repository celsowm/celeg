#include "kernel_common.cuh"

// Pulled in outside the namespace block below so every launcher definition in
// this translation unit is checked against its public declaration and the
// typed argument aggregates it takes.
#include "celeg/backend/cuda/kernels/attention.hpp"

namespace celeg {
#include "kv_cache.cuh"
#include "attention_common.cuh"
#include "attention_dense.cuh"
#include "attention_segmented.cuh"
#include "attention_batch_ptrs.cuh"
#include "attention_paged.cuh"
#include "attention_alibi.cuh"
#include "attention_latent.cuh"
#include "attention_gemm.cuh"
}
