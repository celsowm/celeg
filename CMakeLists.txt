cmake_minimum_required(VERSION 3.24)
project(lfm25_cuda_cpp VERSION 0.0.21 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LFM_ENABLE_CUDA "Build the NVIDIA CUDA backend when a CUDA compiler is available" ON)
option(LFM_BUILD_TESTS "Build host, CPU and CUDA tests" ON)
option(LFM_FAST_MATH "Enable CUDA --use_fast_math" OFF)
option(LFM_DEBUG_SYNC "Synchronize after every CUDA kernel" OFF)
option(LFM_RUN_CUDA_TESTS "Register CUDA runtime tests with CTest" ON)
set(LFM_VARIANTS "230m;1.2b-instruct;1.2b-thinking" CACHE STRING
    "Semicolon-separated list of model variants to compile in. Available: 230m, 1.2b-instruct, 1.2b-thinking")

list(REMOVE_DUPLICATES LFM_VARIANTS)
foreach(variant IN LISTS LFM_VARIANTS)
    if(variant STREQUAL "230m")
        message(STATUS "LFM variant enabled: LiquidAI/LFM2.5-230M")
    elseif(variant STREQUAL "1.2b-instruct")
        message(STATUS "LFM variant enabled: LiquidAI/LFM2.5-1.2B-Instruct")
    elseif(variant STREQUAL "1.2b-thinking")
        message(STATUS "LFM variant enabled: LiquidAI/LFM2.5-1.2B-Thinking")
    else()
        message(FATAL_ERROR "Unknown LFM variant: ${variant}. Valid: 230m, 1.2b-instruct, 1.2b-thinking")
    endif()
endforeach()
# Expose the variant list as a space-separated string for any tooling that
# wants to read it at compile time. Semicolons are avoided so MSVC does not
# split the value on the command line.
string(REPLACE ";" " " LFM_VARIANTS_STR "${LFM_VARIANTS}")
add_compile_definitions(LFM_VARIANTS="${LFM_VARIANTS_STR}")

find_package(Threads REQUIRED)

if(MSVC)
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/utf-8>)
endif()

add_library(lfm25_host
    src/json.cpp
    src/config.cpp
    src/reference.cpp
    src/quantization.cpp
    src/safetensors.cpp
    src/tokenizer.cpp
    src/runtime_types.cpp
    src/execution_plan.cpp
    src/concurrent_policy.cpp
    src/concurrent_metrics.cpp
    src/prefix_radix.cpp
    src/prefix_cache.cpp
    src/request_registry.cpp
    src/batch_planner.cpp
    src/engine_worker.cpp
    src/cpu_isa.cpp
    src/cpu_numa.cpp
    src/cpu_topology.cpp
    src/cpu_thread_pool.cpp
    src/cpu_quantization.cpp
    src/cpu_kernels.cpp
    src/cpu_paged_kv.cpp
    src/cpu_prefix_cache.cpp
    src/downloader.cpp
    src/model_shape.cpp
    src/model_variant.cpp
    src/chat_template.cpp
)
target_include_directories(lfm25_host PUBLIC include)
if(NOT MSVC)
    target_compile_options(lfm25_host PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(lfm25_host PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(lfm25_host PUBLIC winhttp)
endif()

add_library(lfm25_cpu_core
    src/cpu_model.cpp
    src/cpu_packed.cpp
    src/cpu_concurrent.cpp
)
target_include_directories(lfm25_cpu_core PUBLIC include)
if(NOT MSVC)
    target_compile_options(lfm25_cpu_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(lfm25_cpu_core PUBLIC lfm25_host Threads::Threads)

add_library(lfm25_cpu SHARED src/cpu_c_api.cpp)
target_link_libraries(lfm25_cpu PRIVATE lfm25_cpu_core)
target_include_directories(lfm25_cpu PUBLIC include)
target_compile_definitions(lfm25_cpu PRIVATE LFM25_CPU_C_API_BUILD=1)
set_target_properties(lfm25_cpu PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    OUTPUT_NAME lfm25_cpu)

add_executable(lfm25-cpu-run src/cpu_main.cpp)
target_link_libraries(lfm25-cpu-run PRIVATE lfm25_cpu_core)
add_executable(lfm25-cpu-pack src/cpu_pack.cpp)
target_link_libraries(lfm25-cpu-pack PRIVATE lfm25_cpu_core)
add_executable(lfm25-cpu-kernel-benchmark src/cpu_kernel_benchmark.cpp)
target_link_libraries(lfm25-cpu-kernel-benchmark PRIVATE lfm25_host)
add_executable(lfm25-cpu-c-api-example examples/cpu_c_api_example.c)
target_link_libraries(lfm25-cpu-c-api-example PRIVATE lfm25_cpu)
add_executable(lfm25-cpu-concurrent-c-api-example examples/cpu_concurrent_c_api_example.c)
target_link_libraries(lfm25-cpu-concurrent-c-api-example PRIVATE lfm25_cpu)
add_executable(lfm25-cpu-concurrent-benchmark src/cpu_concurrent_benchmark.cpp)
target_link_libraries(lfm25-cpu-concurrent-benchmark PRIVATE lfm25_cpu_core)
add_executable(lfm25-cpu-prefill-benchmark src/cpu_prefill_benchmark.cpp)
target_link_libraries(lfm25-cpu-prefill-benchmark PRIVATE lfm25_cpu_core)
add_executable(lfm25-cpu-prefix-cache-benchmark src/cpu_prefix_cache_benchmark.cpp)
target_link_libraries(lfm25-cpu-prefix-cache-benchmark PRIVATE lfm25_cpu_core)
add_executable(lfm25-cpu-compare-reference src/cpu_compare_reference.cpp)
target_link_libraries(lfm25-cpu-compare-reference PRIVATE lfm25_cpu_core)

add_executable(lfm25-download src/download_main.cpp)
target_link_libraries(lfm25-download PRIVATE lfm25_host)

include(CheckLanguage)
set(LFM_CUDA_AVAILABLE OFF)
if(LFM_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        set(LFM_CUDA_AVAILABLE ON)
    else()
        message(WARNING "CUDA compiler not found: building the CPU backend only")
    endif()
endif()

if(LFM_CUDA_AVAILABLE)
    add_library(lfm25_core
        src/kernels.cu
        src/model.cu
        src/packed.cu
        src/paged_kv.cu
        src/session_store.cu
        src/weight_layout.cu
        src/weight_loader.cu
        src/gemm_dispatcher.cu
        src/concurrent.cpp
    )
    target_include_directories(lfm25_core PUBLIC include)
    target_link_libraries(lfm25_core PUBLIC lfm25_host CUDA::cudart CUDA::cublas CUDA::cublasLt Threads::Threads)
    target_compile_options(lfm25_core PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    if(LFM_FAST_MATH)
        target_compile_options(lfm25_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    endif()
    if(LFM_DEBUG_SYNC)
        target_compile_definitions(lfm25_host PUBLIC LFM_DEBUG_SYNC=1)
        target_compile_definitions(lfm25_core PUBLIC LFM_DEBUG_SYNC=1)
    endif()

    add_executable(lfm25-run src/main.cpp)
    target_link_libraries(lfm25-run PRIVATE lfm25_core)

    add_library(lfm25 SHARED src/c_api.cpp src/concurrent_c_api.cpp)
    target_link_libraries(lfm25 PRIVATE lfm25_core)
    target_include_directories(lfm25 PUBLIC include)
    target_compile_definitions(lfm25 PRIVATE LFM25_C_API_BUILD=1)
    set_target_properties(lfm25 PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0 OUTPUT_NAME lfm25)

    add_executable(lfm25-concurrent-benchmark src/concurrent_benchmark.cpp)
    target_link_libraries(lfm25-concurrent-benchmark PRIVATE lfm25_core)
    add_executable(lfm25-prefix-cache-benchmark src/prefix_cache_benchmark.cpp)
    target_link_libraries(lfm25-prefix-cache-benchmark PRIVATE lfm25_core)
    add_executable(lfm25-c-api-example examples/c_api_example.c)
    target_link_libraries(lfm25-c-api-example PRIVATE lfm25)
    add_executable(lfm25-concurrent-c-api-example examples/concurrent_c_api_example.c)
    target_link_libraries(lfm25-concurrent-c-api-example PRIVATE lfm25)
endif()

add_executable(lfm25-compare-logits src/compare_logits.cpp)

include(GNUInstallDirs)
set(LFM_INSTALL_TARGETS lfm25_host lfm25_cpu_core lfm25_cpu lfm25-cpu-run lfm25-cpu-pack lfm25-cpu-kernel-benchmark lfm25-cpu-c-api-example
    lfm25-cpu-concurrent-c-api-example lfm25-cpu-concurrent-benchmark
    lfm25-cpu-prefill-benchmark lfm25-cpu-prefix-cache-benchmark
    lfm25-cpu-compare-reference lfm25-compare-logits lfm25-download)
if(LFM_CUDA_AVAILABLE)
    list(APPEND LFM_INSTALL_TARGETS lfm25_core lfm25 lfm25-run
        lfm25-concurrent-benchmark lfm25-prefix-cache-benchmark
        lfm25-c-api-example lfm25-concurrent-c-api-example)
endif()
install(TARGETS ${LFM_INSTALL_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(LFM_BUILD_TESTS)
    enable_testing()

    add_executable(json_test tests/json_test.cpp src/json.cpp)
    target_include_directories(json_test PRIVATE include)
    add_test(NAME json_test COMMAND json_test)
    add_executable(config_test tests/config_test.cpp src/config.cpp src/model_shape.cpp src/model_variant.cpp src/json.cpp)
    target_include_directories(config_test PRIVATE include)
    add_test(NAME config_test COMMAND config_test)
    add_executable(model_variant_test tests/model_variant_test.cpp src/model_shape.cpp src/model_variant.cpp)
    target_include_directories(model_variant_test PRIVATE include)
    add_test(NAME model_variant_test COMMAND model_variant_test)
    add_executable(chat_template_test tests/chat_template_test.cpp src/chat_template.cpp)
    target_include_directories(chat_template_test PRIVATE include)
    add_test(NAME chat_template_test COMMAND chat_template_test)
    add_executable(reference_test tests/reference_test.cpp src/reference.cpp)
    target_include_directories(reference_test PRIVATE include)
    add_test(NAME reference_test COMMAND reference_test)
    add_executable(policy_test tests/policy_test.cpp)
    target_include_directories(policy_test PRIVATE include)
    add_test(NAME policy_test COMMAND policy_test)
    add_executable(quantization_test tests/quantization_test.cpp src/quantization.cpp)
    target_include_directories(quantization_test PRIVATE include)
    add_test(NAME quantization_test COMMAND quantization_test)
    add_executable(tokenizer_test tests/tokenizer_test.cpp src/tokenizer.cpp src/chat_template.cpp src/json.cpp)
    target_include_directories(tokenizer_test PRIVATE include)
    add_test(NAME tokenizer_test COMMAND tokenizer_test)
    add_executable(safetensors_test tests/safetensors_test.cpp src/safetensors.cpp src/json.cpp)
    target_include_directories(safetensors_test PRIVATE include)
    add_test(NAME safetensors_test COMMAND safetensors_test)

    add_executable(concurrent_policy_test tests/concurrent_policy_test.cpp src/concurrent_policy.cpp)
    target_include_directories(concurrent_policy_test PRIVATE include)
    add_test(NAME concurrent_policy_test COMMAND concurrent_policy_test)
    add_executable(prefix_radix_test tests/prefix_radix_test.cpp src/prefix_radix.cpp)
    target_include_directories(prefix_radix_test PRIVATE include)
    add_test(NAME prefix_radix_test COMMAND prefix_radix_test)
    add_executable(concurrent_c_api_header_test tests/concurrent_c_api_header_test.c)
    target_include_directories(concurrent_c_api_header_test PRIVATE include)
    add_test(NAME concurrent_c_api_header_test COMMAND concurrent_c_api_header_test)
    add_executable(concurrent_metrics_test tests/concurrent_metrics_test.cpp)
    target_link_libraries(concurrent_metrics_test PRIVATE lfm25_host)
    add_test(NAME concurrent_metrics_test COMMAND concurrent_metrics_test)
    add_executable(execution_plan_test tests/execution_plan_test.cpp)
    target_link_libraries(execution_plan_test PRIVATE lfm25_host)
    add_test(NAME execution_plan_test COMMAND execution_plan_test)
    add_executable(prefix_cache_test tests/prefix_cache_test.cpp)
    target_link_libraries(prefix_cache_test PRIVATE lfm25_host)
    add_test(NAME prefix_cache_test COMMAND prefix_cache_test)
    add_executable(public_facade_test tests/public_facade_test.cpp)
    target_link_libraries(public_facade_test PRIVATE lfm25_host)
    add_test(NAME public_facade_test COMMAND public_facade_test)
    add_executable(request_registry_test tests/request_registry_test.cpp)
    target_link_libraries(request_registry_test PRIVATE lfm25_host)
    add_test(NAME request_registry_test COMMAND request_registry_test)
    add_executable(batch_planner_test tests/batch_planner_test.cpp)
    target_link_libraries(batch_planner_test PRIVATE lfm25_host)
    add_test(NAME batch_planner_test COMMAND batch_planner_test)
    add_executable(engine_worker_test tests/engine_worker_test.cpp)
    target_link_libraries(engine_worker_test PRIVATE lfm25_host)
    add_test(NAME engine_worker_test COMMAND engine_worker_test)

    add_executable(cpu_isa_test tests/cpu_isa_test.cpp)
    target_link_libraries(cpu_isa_test PRIVATE lfm25_host)
    add_test(NAME cpu_isa_test COMMAND cpu_isa_test)
    add_executable(cpu_thread_pool_test tests/cpu_thread_pool_test.cpp)
    target_link_libraries(cpu_thread_pool_test PRIVATE lfm25_host)
    add_test(NAME cpu_thread_pool_test COMMAND cpu_thread_pool_test)
    add_executable(cpu_topology_test tests/cpu_topology_test.cpp)
    target_link_libraries(cpu_topology_test PRIVATE lfm25_host)
    add_test(NAME cpu_topology_test COMMAND cpu_topology_test)
    add_executable(cpu_quantization_test tests/cpu_quantization_test.cpp)
    target_link_libraries(cpu_quantization_test PRIVATE lfm25_host)
    add_test(NAME cpu_quantization_test COMMAND cpu_quantization_test)
    add_executable(cpu_kernels_test tests/cpu_kernels_test.cpp)
    target_link_libraries(cpu_kernels_test PRIVATE lfm25_host)
    add_test(NAME cpu_kernels_test COMMAND cpu_kernels_test)
    add_executable(cpu_kv_cache_test tests/cpu_kv_cache_test.cpp)
    target_link_libraries(cpu_kv_cache_test PRIVATE lfm25_host)
    add_test(NAME cpu_kv_cache_test COMMAND cpu_kv_cache_test)
    add_executable(cpu_paged_kv_test tests/cpu_paged_kv_test.cpp)
    target_link_libraries(cpu_paged_kv_test PRIVATE lfm25_host)
    add_test(NAME cpu_paged_kv_test COMMAND cpu_paged_kv_test)
    add_executable(cpu_parallel_attention_test tests/cpu_parallel_attention_test.cpp)
    target_link_libraries(cpu_parallel_attention_test PRIVATE lfm25_host)
    add_test(NAME cpu_parallel_attention_test COMMAND cpu_parallel_attention_test)
    add_executable(cpu_prefix_cache_test tests/cpu_prefix_cache_test.cpp)
    target_link_libraries(cpu_prefix_cache_test PRIVATE lfm25_host)
    add_test(NAME cpu_prefix_cache_test COMMAND cpu_prefix_cache_test)
    add_executable(cpu_numa_test tests/cpu_numa_test.cpp)
    target_link_libraries(cpu_numa_test PRIVATE lfm25_host)
    add_test(NAME cpu_numa_test COMMAND cpu_numa_test)
    add_executable(cpu_concurrent_metrics_test tests/cpu_concurrent_metrics_test.cpp)
    target_link_libraries(cpu_concurrent_metrics_test PRIVATE lfm25_cpu_core)
    add_test(NAME cpu_concurrent_metrics_test COMMAND cpu_concurrent_metrics_test)
    add_executable(cpu_model_options_test tests/cpu_model_options_test.cpp)
    target_link_libraries(cpu_model_options_test PRIVATE lfm25_cpu_core)
    add_test(NAME cpu_model_options_test COMMAND cpu_model_options_test)
    add_executable(cpu_c_api_header_test tests/cpu_c_api_header_test.c)
    target_include_directories(cpu_c_api_header_test PRIVATE include)
    add_test(NAME cpu_c_api_header_test COMMAND cpu_c_api_header_test)
    add_executable(cpu_c_api_smoke_test tests/cpu_c_api_smoke_test.c)
    target_link_libraries(cpu_c_api_smoke_test PRIVATE lfm25_cpu)
    target_include_directories(cpu_c_api_smoke_test PRIVATE include)
    add_test(NAME cpu_c_api_smoke_test COMMAND cpu_c_api_smoke_test)

    if(LFM_CUDA_AVAILABLE)
        add_executable(c_api_header_test tests/c_api_header_test.c)
        target_include_directories(c_api_header_test PRIVATE include)
        add_test(NAME c_api_header_test COMMAND c_api_header_test)
        add_executable(c_api_smoke_test tests/c_api_smoke_test.c)
        target_link_libraries(c_api_smoke_test PRIVATE lfm25)
        target_include_directories(c_api_smoke_test PRIVATE include)
        add_test(NAME c_api_smoke_test COMMAND c_api_smoke_test)
        add_executable(cuda_kernels_test tests/cuda_kernels_test.cu)
        target_link_libraries(cuda_kernels_test PRIVATE lfm25_core)
        target_include_directories(cuda_kernels_test PRIVATE include)
        if(LFM_RUN_CUDA_TESTS)
            add_test(NAME cuda_kernels_test COMMAND cuda_kernels_test)
        endif()
    endif()
endif()
