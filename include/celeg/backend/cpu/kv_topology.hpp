#pragma once

#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/model/resolved.hpp"

#include <memory>
#include <vector>

namespace celeg {

struct CpuKvTopology {
    std::vector<std::shared_ptr<CpuKvPagePool>> pools;
    std::vector<int> layer_to_pool;
    std::vector<int> layer_to_owner;
};

CpuKvTopology build_cpu_kv_topology(const RuntimeTopology& shape,
                                    const CpuModelOptions& options);

} // namespace celeg
