#include "backend/cuda/runtime_types.hpp"
#include "support/assertions.hpp"

#include <cstdlib>

namespace {

void set_environment(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

}

int main() {
    set_environment("CELEG_FLASH_ATTN", "1");
    set_environment("CELEG_MMQ_TENSOR_CORES", "1");
    set_environment("CELEG_CUDA_MANAGED_WEIGHTS", "1");

    const celeg::CudaModelOptions defaults;
    CELEG_TEST_CHECK(!defaults.flash_attn);
    CELEG_TEST_CHECK(!defaults.mmq_tensor_cores.has_value());
    CELEG_TEST_CHECK(!defaults.managed_weights);

    const celeg::CudaModelOptions overridden =
        celeg::cuda_model_options_from_environment(defaults);
    CELEG_TEST_CHECK(overridden.flash_attn);
    CELEG_TEST_CHECK(overridden.mmq_tensor_cores.has_value());
    CELEG_TEST_CHECK(*overridden.mmq_tensor_cores);
    CELEG_TEST_CHECK(overridden.managed_weights);

    set_environment("CELEG_FLASH_ATTN", "0");
    set_environment("CELEG_MMQ_TENSOR_CORES", "0");
    set_environment("CELEG_CUDA_MANAGED_WEIGHTS", "0");
    const celeg::CudaModelOptions disabled =
        celeg::cuda_model_options_from_environment();
    CELEG_TEST_CHECK(!disabled.flash_attn);
    CELEG_TEST_CHECK(disabled.mmq_tensor_cores.has_value());
    CELEG_TEST_CHECK(!*disabled.mmq_tensor_cores);
    CELEG_TEST_CHECK(!disabled.managed_weights);

    return 0;
}
