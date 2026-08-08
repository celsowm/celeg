cmake_minimum_required(VERSION 3.24)
project(celeg VERSION 0.0.21 LANGUAGES C CXX)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CELEG_ENABLE_CUDA "Build the NVIDIA CUDA backend when a CUDA compiler is available" ON)
option(CELEG_BUILD_TESTS "Build host, CPU and CUDA tests" ON)
option(CELEG_FAST_MATH "Enable CUDA --use_fast_math" OFF)
option(CELEG_DEBUG_SYNC "Synchronize after every CUDA kernel" OFF)
option(CELEG_RUN_CUDA_TESTS "Register CUDA runtime tests with CTest" ON)
option(CELEG_ENABLE_SERVE "Build celeg-serve and its OpenAI-compatible protocol layer (fetches Glaze)" ON)

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
foreach(_celeg_variable IN ITEMS
        CELEG_BASE_RUNTIME_SOURCES CELEG_CPU_BACKEND_SOURCES
        CELEG_CUDA_BACKEND_SOURCES CELEG_API_SOURCES)
    foreach(_celeg_source IN LISTS ${_celeg_variable})
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_celeg_source}")
            message(FATAL_ERROR
                "${_celeg_variable} declares missing source: ${_celeg_source}")
        endif()
    endforeach()
endforeach()

file(GLOB_RECURSE _celeg_public_headers CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "include/celeg/*.h" "include/celeg/*.hpp" "include/celeg/*.cuh")
foreach(_celeg_header IN LISTS _celeg_public_headers)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_celeg_header}")
        message(FATAL_ERROR "Public header disappeared: ${_celeg_header}")
    endif()
endforeach()

if(MSVC)
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/utf-8>)
    # CUDA 13.1's host-config guard predates the currently installed MSVC
    # 19.50 toolset.  Keep the override limited to CUDA translation units and
    # only newer toolsets; supported MSVC versions retain CUDA's normal check.
    if(CELEG_ENABLE_CUDA AND MSVC_VERSION GREATER 1942)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)
    endif()
    set(CELEG_MSVC_SHOWINCLUDES_PREFIX "" CACHE STRING
        "Exact localized MSVC /showIncludes prefix detected by scripts/dev.py")
    if(CELEG_MSVC_SHOWINCLUDES_PREFIX)
        set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "${CELEG_MSVC_SHOWINCLUDES_PREFIX}")
        set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "${CELEG_MSVC_SHOWINCLUDES_PREFIX}")
        set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CELEG_MSVC_SHOWINCLUDES_PREFIX}")
    endif()
endif()

add_library(celeg_base ${CELEG_BASE_RUNTIME_SOURCES})
target_include_directories(celeg_base PUBLIC include)
if(NOT MSVC)
    target_compile_options(celeg_base PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(celeg_base PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(celeg_base PUBLIC winhttp windowscodecs ole32 shlwapi)
endif()

add_library(celeg_runtime INTERFACE)
target_link_libraries(celeg_runtime INTERFACE celeg_base)
target_include_directories(celeg_runtime INTERFACE include)

add_library(celeg_cpu_backend ${CELEG_CPU_BACKEND_SOURCES})
target_include_directories(celeg_cpu_backend PUBLIC include)
if(NOT MSVC)
    target_compile_options(celeg_cpu_backend PRIVATE -Wall -Wextra -Wpedantic)
else()
    set_source_files_properties(
        src/backend/cpu/kernels/quantized_dot_avx2_msvc.cpp
        src/backend/cpu/kernels/elementwise_avx2_msvc.cpp
        src/backend/cpu/kernels/gguf_avx2.cpp
        PROPERTIES COMPILE_FLAGS "/arch:AVX2"
    )
endif()
target_link_libraries(celeg_cpu_backend PUBLIC celeg_runtime Threads::Threads)

add_library(celeg SHARED ${CELEG_API_SOURCES})
target_link_libraries(celeg PRIVATE celeg_cpu_backend)
target_include_directories(celeg PUBLIC include)
target_compile_definitions(celeg PRIVATE CELEG_API_BUILD=1)
set_target_properties(celeg PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    OUTPUT_NAME celeg)

add_executable(celeg-cpu-run src/app/cpu/main.cpp)
target_link_libraries(celeg-cpu-run PRIVATE celeg_cpu_backend)
add_executable(celeg-cpu-pack src/tools/cpu_pack.cpp)
target_link_libraries(celeg-cpu-pack PRIVATE celeg_cpu_backend)
add_executable(celeg-cpu-kernel-benchmark src/app/benchmark/cpu/kernels.cpp)
target_link_libraries(celeg-cpu-kernel-benchmark PRIVATE celeg_cpu_backend)
add_executable(celeg-c-api-example examples/api_example.c)
target_link_libraries(celeg-c-api-example PRIVATE celeg)
add_executable(celeg-cpu-concurrent-benchmark src/app/benchmark/cpu/concurrent.cpp)
target_link_libraries(celeg-cpu-concurrent-benchmark PRIVATE celeg_cpu_backend)
add_executable(celeg-cpu-prefill-benchmark src/app/benchmark/cpu/prefill.cpp)
target_link_libraries(celeg-cpu-prefill-benchmark PRIVATE celeg_cpu_backend)
add_executable(celeg-cpu-prefix-cache-benchmark src/app/benchmark/cpu/prefix_cache.cpp)
target_link_libraries(celeg-cpu-prefix-cache-benchmark PRIVATE celeg_cpu_backend)
add_executable(celeg-cpu-compare-reference src/tools/cpu_compare_reference.cpp)
target_link_libraries(celeg-cpu-compare-reference PRIVATE celeg_cpu_backend)
add_executable(celeg-bench src/app/benchmark/cpu/celeg_bench.cpp)
target_link_libraries(celeg-bench PRIVATE celeg_cpu_backend)

add_executable(celeg-download src/app/download/main.cpp)
target_link_libraries(celeg-download PRIVATE celeg_base)

if(CELEG_ENABLE_SERVE)
    include(FetchContent)
    # Keep Celeg's dependency sub-builds isolated from a reused CMake build
    # directory. CMake's --fresh resets the top-level cache but intentionally
    # preserves the default `_deps` tree, whose sub-build cache can belong to
    # another checkout at the same path.
    set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_celeg_deps")
    FetchContent_Declare(
        glaze
        URL https://github.com/stephenberry/glaze/archive/refs/tags/v2.9.5.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(glaze)

    add_library(celeg_serve_protocol src/serve/protocol/mapping.cpp src/serve/protocol/openapi.cpp
        src/serve/chat_generation.cpp)
    target_include_directories(celeg_serve_protocol PUBLIC include)
    target_compile_definitions(celeg_serve_protocol PRIVATE
        CELEG_OPENAPI_SPEC_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/serve/protocol/openapi.json")
    if(NOT MSVC)
        target_compile_options(celeg_serve_protocol PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    target_link_libraries(celeg_serve_protocol PUBLIC celeg_runtime glaze::glaze)

endif()

add_library(celeg_serve_core src/serve/generation_dispatcher.cpp)
target_include_directories(celeg_serve_core PUBLIC include)
if(NOT MSVC)
    target_compile_options(celeg_serve_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(celeg_serve_core PUBLIC Threads::Threads)

add_library(celeg_serve_cpu src/serve/cpu_inference_service.cpp)
target_include_directories(celeg_serve_cpu PUBLIC include)
if(NOT MSVC)
    target_compile_options(celeg_serve_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(celeg_serve_cpu PUBLIC celeg_cpu_backend)
target_link_libraries(celeg PRIVATE celeg_serve_cpu)

if(CELEG_ENABLE_SERVE)
    # libuv is uSockets' event loop backend on platforms without epoll/kqueue
    # (i.e. Windows). Force it to build as a static lib so celeg-serve has no
    # extra DLL to ship.
    set(_celeg_saved_build_shared_libs "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF)
    set(LIBUV_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        libuv
        URL https://github.com/libuv/libuv/archive/refs/tags/v1.49.2.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(libuv)
    set(BUILD_SHARED_LIBS "${_celeg_saved_build_shared_libs}")

    # uSockets has no CMake build of its own (upstream ships a plain
    # Makefile), so its sources are fetched and compiled as our own target.
    # Every backend-specific .c file self-guards with #ifdef on its own
    # macro, so compiling the whole src/ tree with only LIBUS_USE_LIBUV and
    # LIBUS_NO_SSL defined mirrors exactly what the upstream Makefile does
    # for a libuv, no-TLS build.
    FetchContent_Declare(
        usockets
        URL https://github.com/uNetworking/uSockets/archive/86097c490263ab662d62e8e7b541390bdec7d149.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_Populate(usockets)
    file(GLOB CELEG_USOCKETS_SOURCES
        "${usockets_SOURCE_DIR}/src/*.c"
        "${usockets_SOURCE_DIR}/src/eventing/*.c"
        "${usockets_SOURCE_DIR}/src/crypto/*.c"
        "${usockets_SOURCE_DIR}/src/io_uring/*.c")
    add_library(usockets STATIC ${CELEG_USOCKETS_SOURCES})
    target_include_directories(usockets PUBLIC "${usockets_SOURCE_DIR}/src")
    target_compile_definitions(usockets PUBLIC LIBUS_USE_LIBUV LIBUS_NO_SSL)
    target_link_libraries(usockets PUBLIC libuv::libuv)

    # uWebSockets itself is header-only C++ on top of uSockets.
    FetchContent_Declare(
        uwebsockets
        URL https://github.com/uNetworking/uWebSockets/archive/refs/tags/v20.79.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
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
        URL https://github.com/swagger-api/swagger-ui/archive/refs/tags/v5.17.14.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_Populate(swaggerui)

    add_executable(celeg-serve
        src/app/serve/main.cpp
        src/app/serve/static_assets.cpp
        src/app/serve/routes/health.cpp
        src/app/serve/routes/models.cpp
        src/app/serve/routes/chat_ui.cpp
        src/app/serve/routes/docs.cpp
        src/app/serve/routes/tokenize.cpp
        src/app/serve/routes/chat_completions.cpp)
    target_link_libraries(celeg-serve PRIVATE
        celeg_cpu_backend celeg_serve_cpu celeg_serve_core celeg_serve_protocol uwebsockets)
    add_custom_command(TARGET celeg-serve POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf "$<TARGET_FILE_DIR:celeg-serve>/celeg-serve-assets"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:celeg-serve>/celeg-serve-assets/chat"
            "$<TARGET_FILE_DIR:celeg-serve>/celeg-serve-assets/docs"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/web/chat/dist"
            "$<TARGET_FILE_DIR:celeg-serve>/celeg-serve-assets/chat"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/src/app/serve/static/docs.html"
            "$<TARGET_FILE_DIR:celeg-serve>/celeg-serve-assets/docs/index.html"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${swaggerui_SOURCE_DIR}/dist/swagger-ui-bundle.js"
            "${swaggerui_SOURCE_DIR}/dist/swagger-ui-standalone-preset.js"
            "${swaggerui_SOURCE_DIR}/dist/swagger-ui.css"
            "$<TARGET_FILE_DIR:celeg-serve>/celeg-serve-assets/docs"
        COMMENT "Staging celeg-serve web assets")
endif()

include(CheckLanguage)
set(CELEG_CUDA_AVAILABLE OFF)
if(CELEG_ENABLE_CUDA)
    # CUDA 13.1 rejects the installed VS 18 host toolset during compiler
    # identification.  The same flag must be present before check_language;
    # applying it only to target compile options is too late for CMake's
    # compiler probe.
    if(MSVC AND MSVC_VERSION GREATER 1942)
        set(CMAKE_CUDA_FLAGS_INIT
            "${CMAKE_CUDA_FLAGS_INIT} -allow-unsupported-compiler"
            CACHE STRING "Initial CUDA flags" FORCE)
    endif()
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
        set(CELEG_CUDA_AVAILABLE ON)
        if(WIN32)
            set(_celeg_cuda_runtime_dirs
                "${CUDAToolkit_BIN_DIR}/x64"
                "${CUDAToolkit_BIN_DIR}")
            set(CELEG_CUDA_RUNTIME_DLLS)
            foreach(_runtime_dir IN LISTS _celeg_cuda_runtime_dirs)
                if(IS_DIRECTORY "${_runtime_dir}")
                    file(GLOB _runtime_dlls LIST_DIRECTORIES FALSE
                        "${_runtime_dir}/cudart64_*.dll"
                        "${_runtime_dir}/cublas64_*.dll"
                        "${_runtime_dir}/cublasLt64_*.dll")
                    list(APPEND CELEG_CUDA_RUNTIME_DLLS ${_runtime_dlls})
                endif()
            endforeach()
            list(REMOVE_DUPLICATES CELEG_CUDA_RUNTIME_DLLS)
            if(NOT CELEG_CUDA_RUNTIME_DLLS)
                message(FATAL_ERROR
                    "CUDA runtime DLLs were not found under ${CUDAToolkit_BIN_DIR}. "
                    "Check that the selected CUDA toolkit includes cuBLAS.")
            endif()
            add_custom_target(celeg_cuda_runtime ALL
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CELEG_CUDA_RUNTIME_DLLS} "${CMAKE_BINARY_DIR}"
                COMMAND_EXPAND_LISTS
                COMMENT "Staging CUDA runtime DLLs beside Windows executables")
        endif()
    else()
        message(WARNING "CUDA compiler not found: building the CPU backend only")
    endif()
endif()

if(CELEG_CUDA_AVAILABLE)
    add_library(celeg_cuda_backend ${CELEG_CUDA_BACKEND_SOURCES})
    target_include_directories(celeg_cuda_backend PUBLIC include)
    target_link_libraries(celeg_cuda_backend PUBLIC celeg_runtime CUDA::cudart CUDA::cublas CUDA::cublasLt Threads::Threads)
    target_compile_options(celeg_cuda_backend PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    if(CELEG_FAST_MATH)
        target_compile_options(celeg_cuda_backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    endif()
    if(CELEG_DEBUG_SYNC)
        target_compile_definitions(celeg_base PUBLIC CELEG_DEBUG_SYNC=1)
        target_compile_definitions(celeg_cuda_backend PUBLIC CELEG_DEBUG_SYNC=1)
    endif()

    add_executable(celeg-run src/app/cuda/main.cpp)
    target_link_libraries(celeg-run PRIVATE celeg_cuda_backend)
    if(TARGET celeg_cuda_runtime)
        add_dependencies(celeg-run celeg_cuda_runtime)
    endif()

    target_link_libraries(celeg PRIVATE celeg_cuda_backend)
    target_compile_definitions(celeg PRIVATE CELEG_API_WITH_CUDA=1)
    if(TARGET celeg_cuda_runtime)
        add_dependencies(celeg celeg_cuda_runtime)
    endif()

    add_executable(celeg-concurrent-benchmark src/app/benchmark/cuda/concurrent.cpp)
    target_link_libraries(celeg-concurrent-benchmark PRIVATE celeg_cuda_backend)
    add_executable(celeg-prefix-cache-benchmark src/app/benchmark/cuda/prefix_cache.cpp)
    target_link_libraries(celeg-prefix-cache-benchmark PRIVATE celeg_cuda_backend)
    add_executable(celeg-decode-gemv-benchmark src/app/benchmark/cuda/decode_gemv.cu)
    target_link_libraries(celeg-decode-gemv-benchmark PRIVATE celeg_cuda_backend)

    add_library(celeg_serve_cuda src/serve/cuda_inference_service.cpp)
    target_include_directories(celeg_serve_cuda PUBLIC include)
    target_link_libraries(celeg_serve_cuda PUBLIC celeg_cuda_backend)
    target_link_libraries(celeg PRIVATE celeg_serve_cuda)
    if(TARGET celeg-serve)
        target_link_libraries(celeg-serve PRIVATE celeg_serve_cuda)
        target_compile_definitions(celeg-serve PRIVATE CELEG_SERVE_CUDA=1)
    endif()
endif()

add_executable(celeg-compare-logits src/tools/compare_logits.cpp)

add_executable(celeg-pack-experts src/tools/pack_experts.cpp)
target_link_libraries(celeg-pack-experts PRIVATE celeg_base)
target_include_directories(celeg-pack-experts PRIVATE include)

include(GNUInstallDirs)
set(CELEG_INSTALL_TARGETS celeg_base celeg_runtime celeg_cpu_backend celeg celeg-cpu-run celeg-cpu-pack celeg-cpu-kernel-benchmark celeg-c-api-example
    celeg-cpu-concurrent-benchmark
    celeg-cpu-prefill-benchmark celeg-cpu-prefix-cache-benchmark
    celeg-cpu-compare-reference celeg-bench celeg-compare-logits celeg-download celeg-pack-experts
    celeg_serve_core celeg_serve_cpu)
if(CELEG_CUDA_AVAILABLE)
    list(APPEND CELEG_INSTALL_TARGETS celeg_cuda_backend celeg-run
        celeg-concurrent-benchmark celeg-prefix-cache-benchmark
        celeg_serve_cuda
        )
endif()
if(CELEG_ENABLE_SERVE)
    list(APPEND CELEG_INSTALL_TARGETS celeg_serve_protocol celeg-serve)
endif()
install(TARGETS ${CELEG_INSTALL_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(CELEG_ENABLE_SERVE)
    install(DIRECTORY web/chat/dist/
        DESTINATION ${CMAKE_INSTALL_BINDIR}/celeg-serve-assets/chat)
    install(FILES src/app/serve/static/docs.html
        DESTINATION ${CMAKE_INSTALL_BINDIR}/celeg-serve-assets/docs
        RENAME index.html)
    install(FILES
        "${swaggerui_SOURCE_DIR}/dist/swagger-ui-bundle.js"
        "${swaggerui_SOURCE_DIR}/dist/swagger-ui-standalone-preset.js"
        "${swaggerui_SOURCE_DIR}/dist/swagger-ui.css"
        DESTINATION ${CMAKE_INSTALL_BINDIR}/celeg-serve-assets/docs)
endif()
# Keep the installed surface explicit.  The repository contains implementation
# headers under celeg/detail, CUDA device helpers under celeg/backend/cuda, and
# model-loading internals under celeg/model/weights; none of those are consumer
# contracts.  Installing the directory wholesale made internal layout part of
# the package ABI and also forced CPU-only consumers to discover CUDA headers.
set(CELEG_PUBLIC_HEADERS
    include/celeg/api.h
    include/celeg/backend/cpu/concurrent.hpp
    include/celeg/backend/cpu/gguf.hpp
    include/celeg/backend/cpu/isa.hpp
    include/celeg/backend/cpu/kernels.hpp
    include/celeg/backend/cpu/model.hpp
    include/celeg/backend/cpu/numa.hpp
    include/celeg/backend/cpu/paged_kv.hpp
    include/celeg/backend/cpu/prefix_cache.hpp
    include/celeg/backend/cpu/quantization.hpp
    include/celeg/backend/cpu/runtime_types.hpp
    include/celeg/backend/cpu/thread_pool.hpp
    include/celeg/backend/cpu/weight_codec.hpp
    include/celeg/backend/cpu/topology.hpp
    include/celeg/checkpoint/downloader.hpp
    include/celeg/checkpoint/formats/gguf.hpp
    include/celeg/checkpoint/formats/json.hpp
    include/celeg/checkpoint/formats/safetensors.hpp
    include/celeg/checkpoint/metadata.hpp
    include/celeg/checkpoint/gguf_blocks.hpp
    include/celeg/checkpoint/tokenizer.hpp
    include/celeg/checkpoint/repositories/gguf.hpp
    include/celeg/checkpoint/repositories/safetensors.hpp
    include/celeg/checkpoint/tensor_names.hpp
    include/celeg/model/definition.hpp
    include/celeg/model/architecture.hpp
    include/celeg/model/graph.hpp
    include/celeg/model/profiles.hpp
    include/celeg/model/resolved.hpp
    include/celeg/backend/cuda/execution_plan.hpp
    include/celeg/backend/cuda/runtime_types.hpp
    include/celeg/backend/cuda/weight_layout.hpp
    include/celeg/backend/cuda/weight_setup_support.hpp
    include/celeg/backend/cuda/weight_policy.hpp
    include/celeg/backend/cuda/weights_loader.hpp
    include/celeg/model/runtime_types.hpp
    include/celeg/model/visual_embeddings.hpp
    include/celeg/model/session.hpp
    include/celeg/model/weights/roles.hpp
    include/celeg/backend/cuda/model.hpp
    include/celeg/model/reference.hpp
    include/celeg/runtime/cache/kv_page_allocator.hpp
    include/celeg/runtime/cache/page_layout.hpp
    include/celeg/runtime/cache/page_lease.hpp
    include/celeg/runtime/cache/host_expert_cache.hpp
    include/celeg/runtime/cache/prefix_cache_index.hpp
    include/celeg/runtime/cache/prefix_cache.hpp
    include/celeg/runtime/cache/prefix_radix.hpp
    include/celeg/runtime/concurrency/metrics.hpp
    include/celeg/runtime/concurrency/policy_types.hpp
    include/celeg/runtime/concurrency/policy.hpp
    include/celeg/runtime/request_types.hpp
    include/celeg/backend/cuda/concurrency.hpp
    include/celeg/backend/cuda/moe.hpp
    include/celeg/backend/cuda/moe/offload.hpp
    include/celeg/backend/cuda/moe/expert_residency.hpp
    include/celeg/runtime/weights_topology.hpp
    include/celeg/serve/cpu_inference_service.hpp
    include/celeg/backend/cuda/cuda_inference_service.hpp
    include/celeg/serve/generation_dispatcher.hpp
    include/celeg/serve/chat_generation.hpp
    include/celeg/serve/inference_service.hpp
    include/celeg/serve/protocol/chat.hpp
    include/celeg/serve/protocol/json.hpp
    include/celeg/serve/protocol/mapping.hpp
    include/celeg/serve/protocol/openapi.hpp
    include/celeg/serve/types.hpp
    include/celeg/text/chat_template.hpp
    include/celeg/text/chat_profile.hpp
    include/celeg/text/tool_call.hpp
    include/celeg/text/conversation.hpp
    include/celeg/text/tokenizer.hpp)
foreach(_celeg_public_header IN LISTS CELEG_PUBLIC_HEADERS)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_celeg_public_header}")
        message(FATAL_ERROR "Public header manifest entry is missing: ${_celeg_public_header}")
    endif()
endforeach()
foreach(_celeg_public_header IN LISTS CELEG_PUBLIC_HEADERS)
    file(RELATIVE_PATH _celeg_header_relative
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/${_celeg_public_header}")
    get_filename_component(_celeg_header_directory
        "${_celeg_header_relative}" DIRECTORY)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${_celeg_public_header}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${_celeg_header_directory}")
endforeach()

if(CELEG_BUILD_TESTS)
    enable_testing()

    add_executable(json_test tests/json_test.cpp src/checkpoint/formats/json.cpp)
    target_include_directories(json_test PRIVATE include)
    add_test(NAME json_test COMMAND json_test)
    add_executable(packed_int8_test tests/packed_int8_test.cpp
        src/checkpoint/packed_int8.cpp)
    target_include_directories(packed_int8_test PRIVATE include tests)
    add_test(NAME packed_int8_test COMMAND packed_int8_test)
    add_executable(architecture_resolution_test tests/architecture_resolution_test.cpp)
    target_include_directories(architecture_resolution_test PRIVATE include)
    target_link_libraries(architecture_resolution_test PRIVATE celeg_base)
    add_test(NAME architecture_resolution_test COMMAND architecture_resolution_test)
    add_executable(lfm2_layer_decoder_test tests/lfm2_layer_decoder_test.cpp
        src/checkpoint/metadata.cpp src/checkpoint/formats/json.cpp
        src/models/lfm2/layer_decoder.cpp)
    target_include_directories(lfm2_layer_decoder_test PRIVATE include)
    add_test(NAME lfm2_layer_decoder_test COMMAND lfm2_layer_decoder_test)
    set(CELEG_CHAT_TEMPLATE_SOURCES
        src/composition/builtin_chat_profiles.cpp
        src/text/chat_template.cpp src/text/chat_template_support.cpp
        src/models/lfm2/chat_template.cpp src/models/lfm2/tool_call_codec.cpp
        src/models/granite/chat_template.cpp
        src/models/gemma4/chat_template.cpp src/models/gemma4/tool_call_codec.cpp
        src/models/minicpm5/chat_template.cpp src/models/minicpm5/tool_call_codec.cpp
        src/models/nanbeige/chat_template.cpp src/models/nanbeige/tool_call_codec.cpp
        src/models/smollm3/chat_template.cpp src/models/smollm3/tool_call_codec.cpp
        src/models/qwen35/chat_template.cpp
        src/models/nemotron_h/chat_template.cpp)
    add_executable(chat_template_test tests/chat_template_test.cpp
        ${CELEG_CHAT_TEMPLATE_SOURCES}
        )
    target_include_directories(chat_template_test PRIVATE include)
    add_test(NAME chat_template_test COMMAND chat_template_test)
    add_executable(protocol_utils_test tests/protocol_utils_test.cpp)
    target_include_directories(protocol_utils_test PRIVATE include tests)
    add_test(NAME protocol_utils_test COMMAND protocol_utils_test)
    add_executable(conversation_test tests/conversation_test.cpp)
    target_include_directories(conversation_test PRIVATE include)
    add_test(NAME conversation_test COMMAND conversation_test)
    add_executable(reference_test tests/reference_test.cpp src/model/reference.cpp)
    target_include_directories(reference_test PRIVATE include)
    add_test(NAME reference_test COMMAND reference_test)
    add_executable(tensor_resolver_test tests/tensor_resolver_test.cpp
        src/model/weights/roles.cpp)
    target_include_directories(tensor_resolver_test PRIVATE include)
    add_test(NAME tensor_resolver_test COMMAND tensor_resolver_test)
add_executable(policy_test tests/policy_test.cpp)
target_include_directories(policy_test PRIVATE include tests)
target_link_libraries(policy_test PRIVATE celeg_base)

add_executable(weight_plan_test tests/weight_plan_test.cpp)
target_include_directories(weight_plan_test PRIVATE include tests)
target_link_libraries(weight_plan_test PRIVATE celeg_base)
    add_test(NAME policy_test COMMAND policy_test)
    add_executable(quantization_test tests/quantization_test.cpp src/model/weights/quantization.cpp)
    target_include_directories(quantization_test PRIVATE include)
    add_test(NAME quantization_test COMMAND quantization_test)
    add_executable(tokenizer_test tests/tokenizer_test.cpp src/text/tokenizer.cpp
        src/text/tokenizer_json_loader.cpp
        ${CELEG_CHAT_TEMPLATE_SOURCES}
        src/checkpoint/formats/json.cpp src/checkpoint/formats/gguf.cpp)
    target_include_directories(tokenizer_test PRIVATE include)
    add_test(NAME tokenizer_test COMMAND tokenizer_test)
    add_executable(tokenizer_benchmark tests/tokenizer_benchmark.cpp src/text/tokenizer.cpp
        src/text/tokenizer_json_loader.cpp
        ${CELEG_CHAT_TEMPLATE_SOURCES}
        src/checkpoint/formats/json.cpp src/checkpoint/formats/gguf.cpp)
    target_include_directories(tokenizer_benchmark PRIVATE include)
    add_executable(tokenizer_probe tests/tokenizer_probe.cpp src/text/tokenizer.cpp
        src/text/tokenizer_json_loader.cpp
        ${CELEG_CHAT_TEMPLATE_SOURCES}
        src/checkpoint/formats/json.cpp src/checkpoint/formats/gguf.cpp)
    target_include_directories(tokenizer_probe PRIVATE include)
    add_executable(safetensors_test tests/safetensors_test.cpp src/checkpoint/formats/safetensors.cpp src/checkpoint/formats/json.cpp)
    target_include_directories(safetensors_test PRIVATE include)
    add_test(NAME safetensors_test COMMAND safetensors_test)
    add_executable(safetensors_repo_test tests/safetensors_repo_test.cpp src/checkpoint/formats/safetensors.cpp src/checkpoint/repositories/safetensors.cpp src/checkpoint/formats/json.cpp)
    target_include_directories(safetensors_repo_test PRIVATE include)
    add_test(NAME safetensors_repo_test COMMAND safetensors_repo_test)
    add_executable(gguf_parse_test tests/gguf_parse_test.cpp src/checkpoint/formats/gguf.cpp)
    target_include_directories(gguf_parse_test PRIVATE include)
    add_test(NAME gguf_parse_test COMMAND gguf_parse_test)
    add_executable(gguf_repository_test tests/gguf_repository_test.cpp)
    target_link_libraries(gguf_repository_test PRIVATE celeg_base)
    target_include_directories(gguf_repository_test PRIVATE include
        src/checkpoint/repositories/detail)
    add_test(NAME gguf_repository_test COMMAND gguf_repository_test)
    add_executable(moe_router_test tests/moe_router_test.cpp)
    target_link_libraries(moe_router_test PRIVATE celeg_runtime)
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
    add_executable(neutral_headers_compile_test tests/neutral_headers_compile_test.cpp)
    target_include_directories(neutral_headers_compile_test PRIVATE include)
    add_test(NAME neutral_headers_compile_test COMMAND neutral_headers_compile_test)

    add_executable(model_compiler_test tests/model_compiler_test.cpp
        src/backend/cpu/compiler.cpp src/backend/cuda/compiler.cpp)
    target_include_directories(model_compiler_test PRIVATE include tests)
    target_link_libraries(model_compiler_test PRIVATE celeg_base)
    add_test(NAME model_compiler_test COMMAND model_compiler_test)
    add_executable(fake_repository_backend_boundary_test
        tests/fake_repository_backend_boundary_test.cpp
        src/backend/cpu/compiler.cpp src/backend/cuda/compiler.cpp)
    target_include_directories(fake_repository_backend_boundary_test PRIVATE include tests)
    target_link_libraries(fake_repository_backend_boundary_test PRIVATE celeg_base)
    add_test(NAME fake_repository_backend_boundary_test
        COMMAND fake_repository_backend_boundary_test)
    add_test(NAME architecture_boundary_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_architecture_boundaries.py
                --root ${CMAKE_CURRENT_SOURCE_DIR})
    add_executable(numerical_compare_test tests/numerical_compare_test.cpp)
    target_include_directories(numerical_compare_test PRIVATE tests)
    add_test(NAME numerical_compare_test COMMAND numerical_compare_test)
    add_executable(unsupported_moe_policy_test tests/quant/unsupported_moe_policy_test.cpp)
    target_include_directories(unsupported_moe_policy_test PRIVATE include tests)
    add_test(NAME unsupported_moe_policy_test COMMAND unsupported_moe_policy_test)
    add_executable(checkpoint_capabilities_test tests/checkpoint_capabilities_test.cpp)
    target_include_directories(checkpoint_capabilities_test PRIVATE include tests)
    add_test(NAME checkpoint_capabilities_test COMMAND checkpoint_capabilities_test)
    add_executable(checkpoint_format_catalog_test tests/checkpoint_format_catalog_test.cpp)
    target_include_directories(checkpoint_format_catalog_test PRIVATE include tests)
    target_link_libraries(checkpoint_format_catalog_test PRIVATE celeg_base)
    add_test(NAME checkpoint_format_catalog_test COMMAND checkpoint_format_catalog_test)
    add_executable(runtime_context_test tests/runtime_context_test.cpp)
    target_include_directories(runtime_context_test PRIVATE include tests)
    target_link_libraries(runtime_context_test PRIVATE celeg_base)
    add_test(NAME runtime_context_test COMMAND runtime_context_test)
    add_executable(concurrent_metrics_test tests/concurrent_metrics_test.cpp)
    target_link_libraries(concurrent_metrics_test PRIVATE celeg_runtime)
    add_test(NAME concurrent_metrics_test COMMAND concurrent_metrics_test)
    add_executable(execution_plan_test tests/execution_plan_test.cpp
        src/backend/cuda/execution_plan.cpp)
    target_link_libraries(execution_plan_test PRIVATE celeg_runtime)
    add_test(NAME execution_plan_test COMMAND execution_plan_test)
    add_executable(prefix_cache_test tests/prefix_cache_test.cpp)
    target_link_libraries(prefix_cache_test PRIVATE celeg_runtime)
    add_test(NAME prefix_cache_test COMMAND prefix_cache_test)
    add_executable(public_facade_test tests/public_facade_test.cpp)
    target_link_libraries(public_facade_test PRIVATE celeg_runtime)
    add_test(NAME public_facade_test COMMAND public_facade_test)
    add_executable(request_registry_test tests/request_registry_test.cpp)
    target_link_libraries(request_registry_test PRIVATE celeg_runtime)
    add_test(NAME request_registry_test COMMAND request_registry_test)
    add_executable(generation_dispatcher_test tests/generation_dispatcher_test.cpp)
    target_link_libraries(generation_dispatcher_test PRIVATE celeg_serve_core)
    add_test(NAME generation_dispatcher_test COMMAND generation_dispatcher_test)
    add_executable(request_lifecycle_test tests/request_lifecycle_test.cpp)
    target_include_directories(request_lifecycle_test PRIVATE include tests)
    add_test(NAME request_lifecycle_test COMMAND request_lifecycle_test)
    if(CELEG_ENABLE_SERVE)
        add_executable(chat_protocol_mapping_test tests/chat_protocol_mapping_test.cpp)
        target_link_libraries(chat_protocol_mapping_test PRIVATE celeg_serve_protocol)
        add_test(NAME chat_protocol_mapping_test COMMAND chat_protocol_mapping_test)
    endif()
    add_executable(visual_request_test tests/visual_request_test.cpp)
    target_link_libraries(visual_request_test PRIVATE celeg_runtime)
    target_include_directories(visual_request_test PRIVATE include tests)
    add_test(NAME visual_request_test COMMAND visual_request_test)
    add_executable(gemma4_vision_test tests/gemma4_vision_test.cpp)
    target_link_libraries(gemma4_vision_test PRIVATE celeg_base)
    target_include_directories(gemma4_vision_test PRIVATE include tests)
    add_test(NAME gemma4_vision_test COMMAND gemma4_vision_test)
    add_executable(qwen35_vision_test tests/qwen35_vision_test.cpp)
    target_link_libraries(qwen35_vision_test PRIVATE celeg_base)
    target_include_directories(qwen35_vision_test PRIVATE include tests)
    add_test(NAME qwen35_vision_test COMMAND qwen35_vision_test)
    add_executable(batch_planner_test tests/batch_planner_test.cpp)
    target_link_libraries(batch_planner_test PRIVATE celeg_runtime)
    add_test(NAME batch_planner_test COMMAND batch_planner_test)
    add_executable(engine_worker_test tests/engine_worker_test.cpp)
    target_link_libraries(engine_worker_test PRIVATE celeg_runtime)
    add_test(NAME engine_worker_test COMMAND engine_worker_test)
    add_executable(expert_offload_test tests/expert_offload_test.cpp)
    target_link_libraries(expert_offload_test PRIVATE celeg_runtime)
    target_include_directories(expert_offload_test PRIVATE include)
    add_test(NAME expert_offload_test COMMAND expert_offload_test)

    add_executable(host_expert_cache_test tests/host_expert_cache_test.cpp)
    target_link_libraries(host_expert_cache_test PRIVATE celeg_runtime)
    target_include_directories(host_expert_cache_test PRIVATE include)
    add_test(NAME host_expert_cache_test COMMAND host_expert_cache_test)
    add_executable(expert_policy_test tests/expert_policy_test.cpp)
    target_include_directories(expert_policy_test PRIVATE include tests)
    add_test(NAME expert_policy_test COMMAND expert_policy_test)

    add_executable(cpu_isa_test tests/cpu_isa_test.cpp)
    target_link_libraries(cpu_isa_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_isa_test COMMAND cpu_isa_test)
    add_executable(cpu_thread_pool_test tests/cpu_thread_pool_test.cpp)
    target_link_libraries(cpu_thread_pool_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_thread_pool_test COMMAND cpu_thread_pool_test)
    add_executable(cpu_topology_test tests/cpu_topology_test.cpp)
    target_link_libraries(cpu_topology_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_topology_test COMMAND cpu_topology_test)
    add_executable(cpu_kv_topology_test tests/cpu_kv_topology_test.cpp)
    target_link_libraries(cpu_kv_topology_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_kv_topology_test COMMAND cpu_kv_topology_test)
    add_executable(cpu_quantization_test tests/cpu_quantization_test.cpp)
    target_link_libraries(cpu_quantization_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_quantization_test COMMAND cpu_quantization_test)
    add_executable(cpu_kernels_test tests/cpu_kernels_test.cpp)
    target_link_libraries(cpu_kernels_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_kernels_test COMMAND cpu_kernels_test)
    add_executable(cpu_gguf_kernels_test tests/cpu_gguf_kernels_test.cpp)
    target_link_libraries(cpu_gguf_kernels_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_gguf_kernels_test COMMAND cpu_gguf_kernels_test)
    add_executable(cpu_kv_cache_test tests/cpu_kv_cache_test.cpp)
    target_link_libraries(cpu_kv_cache_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_kv_cache_test COMMAND cpu_kv_cache_test)
    add_executable(cpu_paged_kv_test tests/cpu_paged_kv_test.cpp)
    target_link_libraries(cpu_paged_kv_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_paged_kv_test COMMAND cpu_paged_kv_test)
    add_executable(cpu_parallel_attention_test tests/cpu_parallel_attention_test.cpp)
    target_link_libraries(cpu_parallel_attention_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_parallel_attention_test COMMAND cpu_parallel_attention_test)
    add_executable(cpu_prefix_cache_test tests/cpu_prefix_cache_test.cpp)
    target_link_libraries(cpu_prefix_cache_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_prefix_cache_test COMMAND cpu_prefix_cache_test)
    add_executable(cpu_numa_test tests/cpu_numa_test.cpp)
    target_link_libraries(cpu_numa_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_numa_test COMMAND cpu_numa_test)
    add_executable(cpu_concurrent_metrics_test tests/cpu_concurrent_metrics_test.cpp)
    target_link_libraries(cpu_concurrent_metrics_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_concurrent_metrics_test COMMAND cpu_concurrent_metrics_test)
    add_executable(cpu_model_options_test tests/cpu_model_options_test.cpp)
    target_link_libraries(cpu_model_options_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_model_options_test COMMAND cpu_model_options_test)
    add_executable(cpu_moe_route_test tests/cpu_moe_route_test.cpp)
    target_include_directories(cpu_moe_route_test PRIVATE include src/backend/cpu tests)
    target_link_libraries(cpu_moe_route_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_moe_route_test COMMAND cpu_moe_route_test)
    add_executable(cpu_sampler_test tests/cpu_sampler_test.cpp)
    target_link_libraries(cpu_sampler_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_sampler_test COMMAND cpu_sampler_test)
    add_executable(cpu_granite_test tests/cpu_granite_test.cpp)
    target_link_libraries(cpu_granite_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_granite_test COMMAND cpu_granite_test)
    add_executable(cpu_mrope_test tests/cpu_mrope_test.cpp)
    target_link_libraries(cpu_mrope_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_mrope_test COMMAND cpu_mrope_test)
    add_executable(cpu_c_api_smoke_test tests/api_smoke_test.c)
    target_link_libraries(cpu_c_api_smoke_test PRIVATE celeg)
    target_include_directories(cpu_c_api_smoke_test PRIVATE include)
    add_test(NAME cpu_c_api_smoke_test COMMAND cpu_c_api_smoke_test)
    add_executable(runtime_backend_extension_test tests/runtime_backend_extension_test.cpp)
    target_link_libraries(runtime_backend_extension_test PRIVATE celeg_base)
    target_include_directories(runtime_backend_extension_test PRIVATE include tests)
    add_test(NAME runtime_backend_extension_test COMMAND runtime_backend_extension_test)
    add_executable(image_codec_test tests/image_codec_test.cpp)
    target_link_libraries(image_codec_test PRIVATE celeg_base)
    target_include_directories(image_codec_test PRIVATE include tests)
    add_test(NAME image_codec_test COMMAND image_codec_test)

    if(CELEG_CUDA_AVAILABLE)
        add_executable(cuda_kernels_test tests/cuda_kernels_test.cu)
        target_link_libraries(cuda_kernels_test PRIVATE celeg_cuda_backend)
        target_include_directories(cuda_kernels_test PRIVATE include)
        if(CELEG_RUN_CUDA_TESTS)
            add_test(NAME cuda_kernels_test COMMAND cuda_kernels_test)
        endif()

        add_executable(cuda_gguf_kernels_test tests/cuda_gguf_kernels_test.cu)
        target_link_libraries(cuda_gguf_kernels_test PRIVATE celeg_cuda_backend)
        target_include_directories(cuda_gguf_kernels_test PRIVATE include)
        if(CELEG_RUN_CUDA_TESTS)
            add_test(NAME cuda_gguf_kernels_test COMMAND cuda_gguf_kernels_test)
        endif()

        add_executable(packed_workspace_test tests/packed_workspace_test.cpp)
        target_link_libraries(packed_workspace_test PRIVATE celeg_cuda_backend)
        target_include_directories(packed_workspace_test PRIVATE include)
        add_test(NAME packed_workspace_test COMMAND packed_workspace_test)
        add_executable(packed_commit_test tests/packed_commit_test.cpp)
        target_link_libraries(packed_commit_test PRIVATE celeg_cuda_backend)
        target_include_directories(packed_commit_test PRIVATE include tests)
        add_test(NAME packed_commit_test COMMAND packed_commit_test)

        add_executable(moe_route_test tests/moe_route_test.cu)
        target_link_libraries(moe_route_test PRIVATE celeg_cuda_backend)
        target_include_directories(moe_route_test PRIVATE include)
        if(CELEG_RUN_CUDA_TESTS)
            add_test(NAME moe_route_test COMMAND moe_route_test)
        endif()

        add_executable(moe_router_materialization_test tests/quant/moe_router_materialization_test.cpp)
        target_link_libraries(moe_router_materialization_test PRIVATE celeg_cuda_backend)
        target_include_directories(moe_router_materialization_test PRIVATE include tests)
        if(CELEG_RUN_CUDA_TESTS)
            add_test(NAME moe_router_materialization_test COMMAND moe_router_materialization_test)
        endif()

        add_executable(moe_ffn_test tests/moe_ffn_test.cu)
        target_link_libraries(moe_ffn_test PRIVATE celeg_cuda_backend)
        target_include_directories(moe_ffn_test PRIVATE include)
        if(CELEG_RUN_CUDA_TESTS)
            add_test(NAME moe_ffn_test COMMAND moe_ffn_test)
        endif()

        add_executable(moe_integration_test tests/moe_integration_test.cu)
        target_link_libraries(moe_integration_test PRIVATE celeg_cuda_backend)
        target_include_directories(moe_integration_test PRIVATE include)
        if(CELEG_RUN_CUDA_TESTS)
            add_test(NAME moe_integration_test COMMAND moe_integration_test)
        endif()

        add_executable(expert_residency_test tests/expert_residency_test.cu)
        target_link_libraries(expert_residency_test PRIVATE celeg_cuda_backend)
        target_include_directories(expert_residency_test PRIVATE include)
        if(CELEG_RUN_CUDA_TESTS)
            add_test(NAME expert_residency_test COMMAND expert_residency_test)
        endif()

        add_executable(cuda_granite_test tests/cuda_granite_test.cu)
        target_link_libraries(cuda_granite_test PRIVATE celeg_cuda_backend)
        target_include_directories(cuda_granite_test PRIVATE include)
        if(CELEG_RUN_CUDA_TESTS)
            add_test(NAME cuda_granite_test COMMAND cuda_granite_test)
        endif()
    endif()
endif()
