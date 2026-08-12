#pragma once

#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/model/program.hpp"
#include "celeg/model/resolved.hpp"

#include <memory>
#include <vector>

namespace celeg {

struct CpuKvTopology {
    std::vector<std::shared_ptr<CpuKvPagePool>> pools;
    std::vector<int> layer_to_pool;
    std::vector<int> layer_to_owner;
};

CpuStatePageLayout lower_cpu_state_page_layout(const AttentionSpec& attention);

CpuKvTopology build_cpu_kv_topology(const ExecutionTopology& shape,
                                    const CompiledModelProgram& program,
                                    const CpuModelOptions& options);

} // namespace celeg
