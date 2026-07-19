#pragma once

// Umbrella header that pulls in every per-concern kernel declaration. New
// callers should depend on the narrow per-concern header they need (e.g.
// `lfm/kernels/attention.hpp`) rather than this aggregate, so a translation
// unit that only wants `launch_embedding` does not also redeclare the
// sampling and paged-attention launchers (Interface Segregation Principle).

#include "lfm/kernels/embedding.hpp"
#include "lfm/kernels/norm_conv.hpp"
#include "lfm/kernels/rope.hpp"
#include "lfm/kernels/kv_store.hpp"
#include "lfm/kernels/attention.hpp"
#include "lfm/kernels/sampling.hpp"
#include "lfm/kernels/scatter.hpp"
