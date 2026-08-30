#include "celeg/api.h"

#include <cstdint>

int main() {
    celeg_cpu_model_options cpu{};
    celeg_cpu_model_options_init(&cpu);
    if (cpu.struct_size != sizeof(cpu) || cpu.generation.temperature != 0.1f ||
        cpu.generation.top_k != 50 || cpu.cpu.q4_group_size != 32 ||
        cpu.cpu.kv_cache_mode != CELEG_CPU_KV_CACHE_BF16) return 1;

    celeg_cpu_backend_options cpu_backend{};
    celeg_cpu_backend_options_init(&cpu_backend);
    if (cpu_backend.struct_size != sizeof(cpu_backend) ||
        cpu_backend.engine.max_active_requests != 16 ||
        cpu_backend.engine.prefix_cache != 1) return 2;

    celeg_cuda_backend_options cuda{};
    celeg_cuda_backend_options_init(&cuda);
    if (cuda.struct_size != sizeof(cuda) ||
        cuda.model.weight_mode != CELEG_WEIGHT_MODE_BF16 ||
        cuda.model.attention_mode != CELEG_CUDA_ATTENTION_AUTO ||
        cuda.engine.scheduler_policy != 0) return 3;

    celeg_metal_backend_options metal{};
    celeg_metal_backend_options_init(&metal);
    if (metal.struct_size != sizeof(metal) || metal.model.kv_page_tokens != 16 ||
        metal.engine.max_active_requests != 1) return 4;

    celeg_request_options request{};
    celeg_request_options_init(&request);
    if (request.struct_size != sizeof(request) || request.max_new_tokens != 128 ||
        request.generation.seed != 1) return 5;

    if (CELEG_CPU_ISA_AUTO != 0 || CELEG_CPU_AFFINITY_SCATTER != 2 ||
        CELEG_WEIGHT_MODE_INT4 != 2 || CELEG_WEIGHT_MODE_NATIVE_GGUF != 3 ||
        CELEG_CUDA_KV_CACHE_INT8 != 1) return 6;
    return 0;
}
