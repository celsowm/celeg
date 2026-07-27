#pragma once

// Umbrella header that pulls in every per-concern kernel declaration. New
// callers should depend on the narrow per-concern header they need (e.g.
// `lfm/backend/cuda/kernels/attention.hpp`) rather than this aggregate, so a translation
// unit that only wants `launch_embedding` does not also redeclare the
// sampling and paged-attention launchers (Interface Segregation Principle).

#include "lfm/backend/cuda/kernels/embedding.hpp"
#include "lfm/backend/cuda/kernels/norm_conv.hpp"
#include "lfm/backend/cuda/kernels/rope.hpp"
#include "lfm/backend/cuda/kernels/kv_store.hpp"
#include "lfm/backend/cuda/kernels/attention.hpp"
#include "lfm/backend/cuda/kernels/sampling.hpp"
#include "lfm/backend/cuda/kernels/scatter.hpp"
