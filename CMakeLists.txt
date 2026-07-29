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
option(LFM_ENABLE_SERVE "Build lfm25-serve and its OpenAI-compatible protocol layer (fetches Glaze)" ON)
set(LFM_VARIANTS "230m;1.2b-instruct;1.2b-thinking;8b-a1b" CACHE STRING
    "Semicolon-separated list of model variants to compile in. Available: 230m, 1.2b-instruct, 1.2b-thinking, 8b-a1b")

list(REMOVE_DUPLICATES LFM_VARIANTS)
foreach(variant IN LISTS LFM_VARIANTS)
    if(variant STREQUAL "230m")
        message(STATUS "LFM variant enabled: LiquidAI/LFM2.5-230M")
    elseif(variant STREQUAL "1.2b-instruct")
        message(STATUS "LFM variant enabled: LiquidAI/LFM2.5-1.2B-Instruct")
    elseif(variant STREQUAL "1.2b-thinking")
        message(STATUS "LFM variant enabled: LiquidAI/LFM2.5-1.2B-Thinking")
    elseif(variant STREQUAL "8b-a1b")
        message(STATUS "LFM variant enabled: LiquidAI/LFM2.5-8B-A1B (MoE)")
    else()
        message(FATAL_ERROR "Unknown LFM variant: ${variant}. Valid: 230m, 1.2b-instruct, 1.2b-thinking, 8b-a1b")
    endif()
endforeach()
# Expose the variant list as a space-separated string for any tooling that
# wants to read it at compile time. Semicolons are avoided so MSVC does not
# split the value on the command line.
string(REPLACE ";" " " LFM_VARIANTS_STR "${LFM_VARIANTS}")
add_compile_definitions(LFM_VARIANTS="${LFM_VARIANTS_STR}")

find_package(Threads REQUIRED)

include(cmake/sources/base_runtime.cmake)
include(cmake/sources/cpu_backend.cmake)
include(cmake/sources/cuda_backend.cmake)
include(cmake/sources/api.cmake)
include(cmake/sources/apps.cmake)
include(cmake/sources/tests.cmake)

# Fail during a clean configure when a source manifest drifts from the
# checkout. This is intentionally performed before any target is created so
# the diagnostic names the broken manifest entry directly.
foreach(_lfm_variable IN ITEMS
        LFM_BASE_RUNTIME_SOURCES LFM_CPU_BACKEND_SOURCES
        LFM_CUDA_BACKEND_SOURCES LFM_API_SOURCES)
    foreach(_lfm_source IN LISTS ${_lfm_variable})
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_lfm_source}")
            message(FATAL_ERROR
                "${_lfm_variable} declares missing source: ${_lfm_source}")
        endif()
    endforeach()
endforeach()

file(GLOB_RECURSE _lfm_public_headers CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "include/lfm/*.h" "include/lfm/*.hpp" "include/lfm/*.cuh" "include/lfm/*.inl")
foreach(_lfm_header IN LISTS _lfm_public_headers)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_lfm_header}")
        message(FATAL_ERROR "Public header disappeared: ${_lfm_header}")
    endif()
endforeach()

if(MSVC)
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/utf-8>)
    set(LFM_MSVC_SHOWINCLUDES_PREFIX "" CACHE STRING
        "Exact localized MSVC /showIncludes prefix detected by scripts/dev.py")
    if(LFM_MSVC_SHOWINCLUDES_PREFIX)
        set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "${LFM_MSVC_SHOWINCLUDES_PREFIX}")
        set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "${LFM_MSVC_SHOWINCLUDES_PREFIX}")
        set(CMAKE_CL_SHOWINCLUDES_PREFIX "${LFM_MSVC_SHOWINCLUDES_PREFIX}")
    endif()
endif()

add_library(lfm_base ${LFM_BASE_RUNTIME_SOURCES})
target_include_directories(lfm_base PUBLIC include)
if(NOT MSVC)
    target_compile_options(lfm_base PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(lfm_base PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(lfm_base PUBLIC winhttp)
endif()

add_library(lfm_runtime INTERFACE)
target_link_libraries(lfm_runtime INTERFACE lfm_base)
target_include_directories(lfm_runtime INTERFACE include)

add_library(lfm_cpu_backend ${LFM_CPU_BACKEND_SOURCES})
target_include_directories(lfm_cpu_backend PUBLIC include)
if(NOT MSVC)
    target_compile_options(lfm_cpu_backend PRIVATE -Wall -Wextra -Wpedantic)
else()
    set_source_files_properties(
        src/backend/cpu/kernels/quantized_dot_avx2_msvc.cpp
        src/backend/cpu/kernels/elementwise_avx2_msvc.cpp
        src/backend/cpu/kernels/gguf_avx2.cpp
        PROPERTIES COMPILE_FLAGS "/arch:AVX2"
    )
endif()
target_link_libraries(lfm_cpu_backend PUBLIC lfm_runtime Threads::Threads)

add_library(lfm25 SHARED ${LFM_API_SOURCES})
target_link_libraries(lfm25 PRIVATE lfm_cpu_backend)
target_include_directories(lfm25 PUBLIC include)
target_compile_definitions(lfm25 PRIVATE LFM25_API_BUILD=1)
set_target_properties(lfm25 PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    OUTPUT_NAME lfm25)

add_executable(lfm25-cpu-run src/app/cpu/main.cpp)
target_link_libraries(lfm25-cpu-run PRIVATE lfm_cpu_backend)
add_executable(lfm25-cpu-pack src/tools/cpu_pack.cpp)
target_link_libraries(lfm25-cpu-pack PRIVATE lfm_cpu_backend)
add_executable(lfm25-cpu-kernel-benchmark src/app/benchmark/cpu/kernels.cpp)
target_link_libraries(lfm25-cpu-kernel-benchmark PRIVATE lfm_cpu_backend)
add_executable(lfm25-c-api-example examples/api_example.c)
target_link_libraries(lfm25-c-api-example PRIVATE lfm25)
add_executable(lfm25-cpu-concurrent-benchmark src/app/benchmark/cpu/concurrent.cpp)
target_link_libraries(lfm25-cpu-concurrent-benchmark PRIVATE lfm_cpu_backend)
add_executable(lfm25-cpu-prefill-benchmark src/app/benchmark/cpu/prefill.cpp)
target_link_libraries(lfm25-cpu-prefill-benchmark PRIVATE lfm_cpu_backend)
add_executable(lfm25-cpu-prefix-cache-benchmark src/app/benchmark/cpu/prefix_cache.cpp)
target_link_libraries(lfm25-cpu-prefix-cache-benchmark PRIVATE lfm_cpu_backend)
add_executable(lfm25-cpu-compare-reference src/tools/cpu_compare_reference.cpp)
target_link_libraries(lfm25-cpu-compare-reference PRIVATE lfm_cpu_backend)
add_executable(lfm25-bench src/app/benchmark/cpu/lfm25_bench.cpp)
target_link_libraries(lfm25-bench PRIVATE lfm_cpu_backend)

add_executable(lfm25-download src/app/download/main.cpp)
target_link_libraries(lfm25-download PRIVATE lfm_base)

if(LFM_ENABLE_SERVE)
    include(FetchContent)
    FetchContent_Declare(
        glaze
        GIT_REPOSITORY https://github.com/stephenberry/glaze.git
        GIT_TAG v2.9.5
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(glaze)

    add_library(lfm_serve_protocol src/serve/protocol/mapping.cpp src/serve/protocol/openapi.cpp)
    target_include_directories(lfm_serve_protocol PUBLIC include)
    target_compile_definitions(lfm_serve_protocol PRIVATE
        LFM_OPENAPI_SPEC_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/serve/protocol/openapi.json")
    if(NOT MSVC)
        target_compile_options(lfm_serve_protocol PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    target_link_libraries(lfm_serve_protocol PUBLIC lfm_runtime glaze::glaze)
endif()

add_library(lfm_serve_core src/serve/generation_dispatcher.cpp)
target_include_directories(lfm_serve_core PUBLIC include)
if(NOT MSVC)
    target_compile_options(lfm_serve_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(lfm_serve_core PUBLIC Threads::Threads)

add_library(lfm_serve_cpu src/serve/cpu_inference_service.cpp)
target_include_directories(lfm_serve_cpu PUBLIC include)
if(NOT MSVC)
    target_compile_options(lfm_serve_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(lfm_serve_cpu PUBLIC lfm_cpu_backend)
target_link_libraries(lfm25 PRIVATE lfm_serve_cpu)

if(LFM_ENABLE_SERVE)
    # libuv is uSockets' event loop backend on platforms without epoll/kqueue
    # (i.e. Windows). Force it to build as a static lib so lfm25-serve has no
    # extra DLL to ship.
    set(_lfm_saved_build_shared_libs "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF)
    set(LIBUV_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        libuv
        GIT_REPOSITORY https://github.com/libuv/libuv.git
        GIT_TAG v1.49.2
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(libuv)
    set(BUILD_SHARED_LIBS "${_lfm_saved_build_shared_libs}")

    # uSockets has no CMake build of its own (upstream ships a plain
    # Makefile), so its sources are fetched and compiled as our own target.
    # Every backend-specific .c file self-guards with #ifdef on its own
    # macro, so compiling the whole src/ tree with only LIBUS_USE_LIBUV and
    # LIBUS_NO_SSL defined mirrors exactly what the upstream Makefile does
    # for a libuv, no-TLS build.
    FetchContent_Declare(
        usockets
        GIT_REPOSITORY https://github.com/uNetworking/uSockets.git
        GIT_TAG 86097c490263ab662d62e8e7b541390bdec7d149 # pinned by uWebSockets v20.79.0
        GIT_SHALLOW FALSE)
    FetchContent_Populate(usockets)
    file(GLOB LFM_USOCKETS_SOURCES
        "${usockets_SOURCE_DIR}/src/*.c"
        "${usockets_SOURCE_DIR}/src/eventing/*.c"
        "${usockets_SOURCE_DIR}/src/crypto/*.c"
        "${usockets_SOURCE_DIR}/src/io_uring/*.c")
    add_library(usockets STATIC ${LFM_USOCKETS_SOURCES})
    target_include_directories(usockets PUBLIC "${usockets_SOURCE_DIR}/src")
    target_compile_definitions(usockets PUBLIC LIBUS_USE_LIBUV LIBUS_NO_SSL)
    target_link_libraries(usockets PUBLIC libuv::libuv)

    # uWebSockets itself is header-only C++ on top of uSockets.
    FetchContent_Declare(
        uwebsockets
        GIT_REPOSITORY https://github.com/uNetworking/uWebSockets.git
        GIT_TAG v20.79.0
        GIT_SHALLOW TRUE)
    FetchContent_Populate(uwebsockets)
    add_library(uwebsockets INTERFACE)
    target_include_directories(uwebsockets INTERFACE "${uwebsockets_SOURCE_DIR}/src")
    target_compile_definitions(uwebsockets INTERFACE UWS_NO_ZLIB)
    target_link_libraries(uwebsockets INTERFACE usockets)

    # Swagger UI ships no CMake build; it's pure static assets (dist/), so we
    # only need to fetch the tree and point the server at it. Served straight
    # out of the fetched source dir rather than embedded, since minified
    # swagger-ui-bundle.js exceeds MSVC's string-literal size limit.
    FetchContent_Declare(
        swaggerui
        GIT_REPOSITORY https://github.com/swagger-api/swagger-ui.git
        GIT_TAG v5.17.14
        GIT_SHALLOW TRUE)
    FetchContent_Populate(swaggerui)

    add_executable(lfm25-serve
        src/app/serve/main.cpp
        src/app/serve/static_assets.cpp
        src/app/serve/routes/health.cpp
        src/app/serve/routes/models.cpp
        src/app/serve/routes/docs.cpp
        src/app/serve/routes/tokenize.cpp
        src/app/serve/routes/chat_completions.cpp)
    target_link_libraries(lfm25-serve PRIVATE
        lfm_cpu_backend lfm_serve_cpu lfm_serve_core lfm_serve_protocol uwebsockets)
    target_compile_definitions(lfm25-serve PRIVATE
        LFM_SWAGGER_UI_DIR="${swaggerui_SOURCE_DIR}/dist"
        LFM_SERVE_STATIC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/app/serve/static")
endif()

include(CheckLanguage)
set(LFM_CUDA_AVAILABLE OFF)
if(LFM_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            # Unsupported newer MSVC toolsets can fail inside <chrono> under
            # NVCC's RelWithDebInfo host pass. Keep normal C/C++ symbols while
            # compiling CUDA translation units with the known-good optimized
            # host flags.
            set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELEASE}")
        endif()
        find_package(CUDAToolkit REQUIRED)
        set(LFM_CUDA_AVAILABLE ON)
        if(WIN32)
            set(_lfm_cuda_runtime_dirs
                "${CUDAToolkit_BIN_DIR}/x64"
                "${CUDAToolkit_BIN_DIR}")
            set(LFM_CUDA_RUNTIME_DLLS)
            foreach(_runtime_dir IN LISTS _lfm_cuda_runtime_dirs)
                if(IS_DIRECTORY "${_runtime_dir}")
                    file(GLOB _runtime_dlls LIST_DIRECTORIES FALSE
                        "${_runtime_dir}/cudart64_*.dll"
                        "${_runtime_dir}/cublas64_*.dll"
                        "${_runtime_dir}/cublasLt64_*.dll")
                    list(APPEND LFM_CUDA_RUNTIME_DLLS ${_runtime_dlls})
                endif()
            endforeach()
            list(REMOVE_DUPLICATES LFM_CUDA_RUNTIME_DLLS)
            if(NOT LFM_CUDA_RUNTIME_DLLS)
                message(FATAL_ERROR
                    "CUDA runtime DLLs were not found under ${CUDAToolkit_BIN_DIR}. "
                    "Check that the selected CUDA toolkit includes cuBLAS.")
            endif()
            add_custom_target(lfm25_cuda_runtime ALL
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${LFM_CUDA_RUNTIME_DLLS} "${CMAKE_BINARY_DIR}"
                COMMAND_EXPAND_LISTS
                COMMENT "Staging CUDA runtime DLLs beside Windows executables")
        endif()
    else()
        message(WARNING "CUDA compiler not found: building the CPU backend only")
    endif()
endif()

if(LFM_CUDA_AVAILABLE)
    add_library(lfm_cuda_backend ${LFM_CUDA_BACKEND_SOURCES})
    target_include_directories(lfm_cuda_backend PUBLIC include)
    target_link_libraries(lfm_cuda_backend PUBLIC lfm_runtime CUDA::cudart CUDA::cublas CUDA::cublasLt Threads::Threads)
    target_compile_options(lfm_cuda_backend PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    if(LFM_FAST_MATH)
        target_compile_options(lfm_cuda_backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    endif()
    if(LFM_DEBUG_SYNC)
        target_compile_definitions(lfm_base PUBLIC LFM_DEBUG_SYNC=1)
        target_compile_definitions(lfm_cuda_backend PUBLIC LFM_DEBUG_SYNC=1)
    endif()

    add_executable(lfm25-run src/app/cuda/main.cpp)
    target_link_libraries(lfm25-run PRIVATE lfm_cuda_backend)
    if(TARGET lfm25_cuda_runtime)
        add_dependencies(lfm25-run lfm25_cuda_runtime)
    endif()

    target_link_libraries(lfm25 PRIVATE lfm_cuda_backend)
    target_compile_definitions(lfm25 PRIVATE LFM25_API_WITH_CUDA=1)
    if(TARGET lfm25_cuda_runtime)
        add_dependencies(lfm25 lfm25_cuda_runtime)
    endif()

    add_executable(lfm25-concurrent-benchmark src/app/benchmark/cuda/concurrent.cpp)
    target_link_libraries(lfm25-concurrent-benchmark PRIVATE lfm_cuda_backend)
    add_executable(lfm25-prefix-cache-benchmark src/app/benchmark/cuda/prefix_cache.cpp)
    target_link_libraries(lfm25-prefix-cache-benchmark PRIVATE lfm_cuda_backend)

    add_library(lfm_serve_cuda src/serve/cuda_inference_service.cpp)
    target_include_directories(lfm_serve_cuda PUBLIC include)
    target_link_libraries(lfm_serve_cuda PUBLIC lfm_cuda_backend)
    target_link_libraries(lfm25 PRIVATE lfm_serve_cuda)
    if(TARGET lfm25-serve)
        target_link_libraries(lfm25-serve PRIVATE lfm_serve_cuda)
        target_compile_definitions(lfm25-serve PRIVATE LFM_SERVE_CUDA=1)
    endif()
endif()

add_executable(lfm25-compare-logits src/tools/compare_logits.cpp)

add_executable(lfm25-pack-experts src/tools/pack_experts.cpp)
target_link_libraries(lfm25-pack-experts PRIVATE lfm_base)
target_include_directories(lfm25-pack-experts PRIVATE include)

include(GNUInstallDirs)
set(LFM_INSTALL_TARGETS lfm_base lfm_runtime lfm_cpu_backend lfm25 lfm25-cpu-run lfm25-cpu-pack lfm25-cpu-kernel-benchmark lfm25-c-api-example
    lfm25-cpu-concurrent-benchmark
    lfm25-cpu-prefill-benchmark lfm25-cpu-prefix-cache-benchmark
    lfm25-cpu-compare-reference lfm25-bench lfm25-compare-logits lfm25-download lfm25-pack-experts
    lfm_serve_core lfm_serve_cpu)
if(LFM_CUDA_AVAILABLE)
    list(APPEND LFM_INSTALL_TARGETS lfm_cuda_backend lfm25-run
        lfm25-concurrent-benchmark lfm25-prefix-cache-benchmark
        lfm_serve_cuda
        )
endif()
if(LFM_ENABLE_SERVE)
    list(APPEND LFM_INSTALL_TARGETS lfm_serve_protocol lfm25-serve)
endif()
install(TARGETS ${LFM_INSTALL_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(LFM_BUILD_TESTS)
    enable_testing()

    add_executable(json_test tests/json_test.cpp src/checkpoint/formats/json.cpp)
    target_include_directories(json_test PRIVATE include)
    add_test(NAME json_test COMMAND json_test)
    add_executable(config_test tests/config_test.cpp src/model/config/config.cpp src/model/config/shape.cpp src/model/config/variant.cpp src/checkpoint/formats/json.cpp)
    target_include_directories(config_test PRIVATE include)
    add_test(NAME config_test COMMAND config_test)
    add_executable(model_variant_test tests/model_variant_test.cpp src/model/config/shape.cpp src/model/config/variant.cpp)
    target_include_directories(model_variant_test PRIVATE include)
    add_test(NAME model_variant_test COMMAND model_variant_test)
    add_executable(chat_template_test tests/chat_template_test.cpp src/text/chat_template.cpp)
    target_include_directories(chat_template_test PRIVATE include)
    add_test(NAME chat_template_test COMMAND chat_template_test)
    add_executable(reference_test tests/reference_test.cpp src/model/reference.cpp)
    target_include_directories(reference_test PRIVATE include)
    add_test(NAME reference_test COMMAND reference_test)
    add_executable(policy_test tests/policy_test.cpp)
    target_include_directories(policy_test PRIVATE include)
    add_test(NAME policy_test COMMAND policy_test)
    add_executable(quantization_test tests/quantization_test.cpp src/model/weights/quantization.cpp)
    target_include_directories(quantization_test PRIVATE include)
    add_test(NAME quantization_test COMMAND quantization_test)
    add_executable(tokenizer_test tests/tokenizer_test.cpp src/text/tokenizer.cpp src/text/chat_template.cpp src/checkpoint/formats/json.cpp src/checkpoint/formats/gguf.cpp)
    target_include_directories(tokenizer_test PRIVATE include)
    add_test(NAME tokenizer_test COMMAND tokenizer_test)
    add_executable(safetensors_test tests/safetensors_test.cpp src/checkpoint/formats/safetensors.cpp src/checkpoint/formats/json.cpp)
    target_include_directories(safetensors_test PRIVATE include)
    add_test(NAME safetensors_test COMMAND safetensors_test)
    add_executable(safetensors_repo_test tests/safetensors_repo_test.cpp src/checkpoint/formats/safetensors.cpp src/checkpoint/repositories/safetensors.cpp src/checkpoint/formats/json.cpp)
    target_include_directories(safetensors_repo_test PRIVATE include)
    add_test(NAME safetensors_repo_test COMMAND safetensors_repo_test)
    add_executable(gguf_parse_test tests/gguf_parse_test.cpp src/checkpoint/formats/gguf.cpp)
    target_include_directories(gguf_parse_test PRIVATE include)
    add_test(NAME gguf_parse_test COMMAND gguf_parse_test)
    add_executable(moe_router_test tests/moe_router_test.cpp)
    target_link_libraries(moe_router_test PRIVATE lfm_runtime)
    target_include_directories(moe_router_test PRIVATE include)
    add_test(NAME moe_router_test COMMAND moe_router_test)

    add_executable(concurrent_policy_test tests/concurrent_policy_test.cpp src/runtime/concurrency/policy.cpp)
    target_include_directories(concurrent_policy_test PRIVATE include)
    add_test(NAME concurrent_policy_test COMMAND concurrent_policy_test)
    add_executable(prefix_radix_test tests/prefix_radix_test.cpp src/runtime/cache/prefix_radix.cpp)
    target_include_directories(prefix_radix_test PRIVATE include)
    add_test(NAME prefix_radix_test COMMAND prefix_radix_test)
    add_executable(page_lease_test tests/page_lease_test.cpp)
    target_include_directories(page_lease_test PRIVATE include)
    add_test(NAME page_lease_test COMMAND page_lease_test)
    add_executable(api_header_test tests/api_header_test.c)
    target_include_directories(api_header_test PRIVATE include)
    add_test(NAME api_header_test COMMAND api_header_test)
    add_executable(concurrent_metrics_test tests/concurrent_metrics_test.cpp)
    target_link_libraries(concurrent_metrics_test PRIVATE lfm_runtime)
    add_test(NAME concurrent_metrics_test COMMAND concurrent_metrics_test)
    add_executable(execution_plan_test tests/execution_plan_test.cpp)
    target_link_libraries(execution_plan_test PRIVATE lfm_runtime)
    add_test(NAME execution_plan_test COMMAND execution_plan_test)
    add_executable(prefix_cache_test tests/prefix_cache_test.cpp)
    target_link_libraries(prefix_cache_test PRIVATE lfm_runtime)
    add_test(NAME prefix_cache_test COMMAND prefix_cache_test)
    add_executable(public_facade_test tests/public_facade_test.cpp)
    target_link_libraries(public_facade_test PRIVATE lfm_runtime)
    add_test(NAME public_facade_test COMMAND public_facade_test)
    add_executable(request_registry_test tests/request_registry_test.cpp)
    target_link_libraries(request_registry_test PRIVATE lfm_runtime)
    add_test(NAME request_registry_test COMMAND request_registry_test)
    add_executable(generation_dispatcher_test tests/generation_dispatcher_test.cpp)
    target_link_libraries(generation_dispatcher_test PRIVATE lfm_serve_core)
    add_test(NAME generation_dispatcher_test COMMAND generation_dispatcher_test)
    if(LFM_ENABLE_SERVE)
        add_executable(chat_protocol_mapping_test tests/chat_protocol_mapping_test.cpp)
        target_link_libraries(chat_protocol_mapping_test PRIVATE lfm_serve_protocol)
        add_test(NAME chat_protocol_mapping_test COMMAND chat_protocol_mapping_test)
    endif()
    add_executable(batch_planner_test tests/batch_planner_test.cpp)
    target_link_libraries(batch_planner_test PRIVATE lfm_runtime)
    add_test(NAME batch_planner_test COMMAND batch_planner_test)
    add_executable(engine_worker_test tests/engine_worker_test.cpp)
    target_link_libraries(engine_worker_test PRIVATE lfm_runtime)
    add_test(NAME engine_worker_test COMMAND engine_worker_test)
    add_executable(expert_offload_test tests/expert_offload_test.cpp)
    target_link_libraries(expert_offload_test PRIVATE lfm_runtime)
    target_include_directories(expert_offload_test PRIVATE include)
    add_test(NAME expert_offload_test COMMAND expert_offload_test)

    add_executable(pinned_expert_cache_test tests/pinned_expert_cache_test.cpp)
    target_link_libraries(pinned_expert_cache_test PRIVATE lfm_runtime)
    target_include_directories(pinned_expert_cache_test PRIVATE include)
    add_test(NAME pinned_expert_cache_test COMMAND pinned_expert_cache_test)

    add_executable(cpu_isa_test tests/cpu_isa_test.cpp)
    target_link_libraries(cpu_isa_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_isa_test COMMAND cpu_isa_test)
    add_executable(cpu_thread_pool_test tests/cpu_thread_pool_test.cpp)
    target_link_libraries(cpu_thread_pool_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_thread_pool_test COMMAND cpu_thread_pool_test)
    add_executable(cpu_topology_test tests/cpu_topology_test.cpp)
    target_link_libraries(cpu_topology_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_topology_test COMMAND cpu_topology_test)
    add_executable(cpu_quantization_test tests/cpu_quantization_test.cpp)
    target_link_libraries(cpu_quantization_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_quantization_test COMMAND cpu_quantization_test)
    add_executable(cpu_kernels_test tests/cpu_kernels_test.cpp)
    target_link_libraries(cpu_kernels_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_kernels_test COMMAND cpu_kernels_test)
    add_executable(cpu_gguf_kernels_test tests/cpu_gguf_kernels_test.cpp)
    target_link_libraries(cpu_gguf_kernels_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_gguf_kernels_test COMMAND cpu_gguf_kernels_test)
    add_executable(cpu_kv_cache_test tests/cpu_kv_cache_test.cpp)
    target_link_libraries(cpu_kv_cache_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_kv_cache_test COMMAND cpu_kv_cache_test)
    add_executable(cpu_paged_kv_test tests/cpu_paged_kv_test.cpp)
    target_link_libraries(cpu_paged_kv_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_paged_kv_test COMMAND cpu_paged_kv_test)
    add_executable(cpu_parallel_attention_test tests/cpu_parallel_attention_test.cpp)
    target_link_libraries(cpu_parallel_attention_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_parallel_attention_test COMMAND cpu_parallel_attention_test)
    add_executable(cpu_prefix_cache_test tests/cpu_prefix_cache_test.cpp)
    target_link_libraries(cpu_prefix_cache_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_prefix_cache_test COMMAND cpu_prefix_cache_test)
    add_executable(cpu_numa_test tests/cpu_numa_test.cpp)
    target_link_libraries(cpu_numa_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_numa_test COMMAND cpu_numa_test)
    add_executable(cpu_concurrent_metrics_test tests/cpu_concurrent_metrics_test.cpp)
    target_link_libraries(cpu_concurrent_metrics_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_concurrent_metrics_test COMMAND cpu_concurrent_metrics_test)
    add_executable(cpu_model_options_test tests/cpu_model_options_test.cpp)
    target_link_libraries(cpu_model_options_test PRIVATE lfm_cpu_backend)
    add_test(NAME cpu_model_options_test COMMAND cpu_model_options_test)
    add_executable(cpu_c_api_smoke_test tests/api_smoke_test.c)
    target_link_libraries(cpu_c_api_smoke_test PRIVATE lfm25)
    target_include_directories(cpu_c_api_smoke_test PRIVATE include)
    add_test(NAME cpu_c_api_smoke_test COMMAND cpu_c_api_smoke_test)

    if(LFM_CUDA_AVAILABLE)
        add_executable(cuda_kernels_test tests/cuda_kernels_test.cu)
        target_link_libraries(cuda_kernels_test PRIVATE lfm_cuda_backend)
        target_include_directories(cuda_kernels_test PRIVATE include)
        if(LFM_RUN_CUDA_TESTS)
            add_test(NAME cuda_kernels_test COMMAND cuda_kernels_test)
        endif()

        add_executable(cuda_gguf_kernels_test tests/cuda_gguf_kernels_test.cu)
        target_link_libraries(cuda_gguf_kernels_test PRIVATE lfm_cuda_backend)
        target_include_directories(cuda_gguf_kernels_test PRIVATE include)
        if(LFM_RUN_CUDA_TESTS)
            add_test(NAME cuda_gguf_kernels_test COMMAND cuda_gguf_kernels_test)
        endif()

        add_executable(moe_route_test tests/moe_route_test.cu)
        target_link_libraries(moe_route_test PRIVATE lfm_cuda_backend)
        target_include_directories(moe_route_test PRIVATE include)
        if(LFM_RUN_CUDA_TESTS)
            add_test(NAME moe_route_test COMMAND moe_route_test)
        endif()

        add_executable(moe_ffn_test tests/moe_ffn_test.cu)
        target_link_libraries(moe_ffn_test PRIVATE lfm_cuda_backend)
        target_include_directories(moe_ffn_test PRIVATE include)
        if(LFM_RUN_CUDA_TESTS)
            add_test(NAME moe_ffn_test COMMAND moe_ffn_test)
        endif()

        add_executable(moe_integration_test tests/moe_integration_test.cu)
        target_link_libraries(moe_integration_test PRIVATE lfm_cuda_backend)
        target_include_directories(moe_integration_test PRIVATE include)
        if(LFM_RUN_CUDA_TESTS)
            add_test(NAME moe_integration_test COMMAND moe_integration_test)
        endif()

        add_executable(expert_residency_test tests/expert_residency_test.cu)
        target_link_libraries(expert_residency_test PRIVATE lfm_cuda_backend)
        target_include_directories(expert_residency_test PRIVATE include)
        if(LFM_RUN_CUDA_TESTS)
            add_test(NAME expert_residency_test COMMAND expert_residency_test)
        endif()
    endif()
endif()
