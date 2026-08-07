#pragma once

// Umbrella header that pulls in every per-concern kernel declaration. New
// callers should depend on the narrow per-concern header they need (e.g.
// `celeg/backend/cuda/kernels/attention.hpp`) rather than this aggregate, so a translation
// unit that only wants `launch_embedding` does not also redeclare the
// sampling and paged-attention launchers (Interface Segregation Principle).

#include "celeg/backend/cuda/kernels/embedding.hpp"
#include "celeg/backend/cuda/kernels/norm_conv.hpp"
#include "celeg/backend/cuda/kernels/mamba2.hpp"
#include "celeg/backend/cuda/kernels/gated_delta.hpp"
#include "celeg/backend/cuda/kernels/rope.hpp"
#include "celeg/backend/cuda/kernels/kv_store.hpp"
#include "celeg/backend/cuda/kernels/attention.hpp"
#include "celeg/backend/cuda/kernels/sampling.hpp"
#include "celeg/backend/cuda/kernels/scatter.hpp"
