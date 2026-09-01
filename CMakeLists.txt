cmake_minimum_required(VERSION 3.24)
project(celeg VERSION 0.0.21 LANGUAGES C CXX)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
include(GNUInstallDirs)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CELEG_ENABLE_CUDA "Build the NVIDIA CUDA backend when a CUDA compiler is available" ON)
if(APPLE)
    set(_celeg_metal_default ON)
else()
    set(_celeg_metal_default OFF)
endif()
option(CELEG_ENABLE_METAL "Build the Apple Metal backend on Apple hosts" ${_celeg_metal_default})
option(CELEG_BUILD_TESTS "Build host, CPU and CUDA tests" ON)
option(CELEG_FAST_MATH "Enable CUDA --use_fast_math" OFF)
option(CELEG_DEBUG_SYNC "Synchronize after every CUDA kernel" OFF)
option(CELEG_RUN_CUDA_TESTS "Register CUDA runtime tests with CTest" ON)
option(CELEG_ENABLE_SERVE "Build celeg-serve and its OpenAI-compatible protocol layer (fetches Glaze)" ON)
option(CELEG_RUN_METAL_TESTS "Register Metal runtime tests with CTest" ${APPLE})

find_package(Threads REQUIRED)

include(src/checkpoint/CMakeLists.txt)
include(src/model/CMakeLists.txt)
include(src/text/CMakeLists.txt)
include(src/quantization/CMakeLists.txt)
include(src/composition/CMakeLists.txt)
include(src/runtime/CMakeLists.txt)
include(src/backend/cpu/CMakeLists.txt)
include(src/backend/cuda/CMakeLists.txt)
include(src/backend/metal/CMakeLists.txt)
include(src/api/CMakeLists.txt)
include(apps/CMakeLists.txt)
include(tools/CMakeLists.txt)
include(tests/CMakeLists.txt)

# Fail during a clean configure when a source manifest drifts from the
# checkout. This is intentionally performed before any target is created so
# the diagnostic names the broken manifest entry directly.
foreach(_celeg_variable IN ITEMS
        CELEG_BASE_RUNTIME_SOURCES CELEG_CPU_BACKEND_SOURCES
        CELEG_CHECKPOINT_SOURCES CELEG_MODEL_SOURCES CELEG_TEXT_SOURCES
        CELEG_QUANTIZATION_SOURCES CELEG_COMPOSITION_SOURCES
        CELEG_RUNTIME_SOURCES
        CELEG_CUDA_BACKEND_SOURCES CELEG_METAL_BACKEND_SOURCES
        CELEG_METAL_MODEL_SOURCES CELEG_METAL_SHADER_SOURCES
        CELEG_METAL_APP_SOURCES CELEG_API_SOURCES)
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

# AddressSanitizer instrumentation for debugging memory-corruption bugs.
# Applies to C/C++ translation units only (MSVC's /fsanitize=address is
# incompatible with CUDA host compilation), so it must be used with
# CELEG_ENABLE_CUDA=OFF. Enabled via `-DCELEG_ASAN=ON` or `python
# scripts/dev.py verify --asan`.
option(CELEG_ASAN "Build C/C++ targets with AddressSanitizer (/fsanitize=address)" OFF)
if(CELEG_ASAN)
    if(CELEG_ENABLE_CUDA)
        message(FATAL_ERROR "CELEG_ASAN is incompatible with CELEG_ENABLE_CUDA=ON")
    endif()
    add_compile_options(
        $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/fsanitize=address>
        $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/Zi>)
    add_link_options(/fsanitize=address)
endif()

add_library(celeg_base ${CELEG_BASE_RUNTIME_SOURCES})
target_include_directories(celeg_base PUBLIC include)
target_include_directories(celeg_base PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(celeg_base PUBLIC
    CELEG_DESCRIPTOR_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/descriptors"
    CELEG_DESCRIPTOR_INSTALL_DIRECTORY="${CMAKE_INSTALL_FULL_DATADIR}/celeg/descriptors")
if(NOT MSVC)
    target_compile_options(celeg_base PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(celeg_base PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(celeg_base PUBLIC winhttp windowscodecs ole32 shlwapi)
else()
    # The POSIX HuggingFace HTTP transport (hf_http.cpp) uses libcurl for
    # TLS, redirects, and resumable range downloads. Windows keeps WinHTTP.
    find_package(CURL REQUIRED)
    target_link_libraries(celeg_base PUBLIC CURL::libcurl)
endif()

add_library(celeg_runtime INTERFACE)
target_link_libraries(celeg_runtime INTERFACE celeg_base)
target_include_directories(celeg_runtime INTERFACE include)

add_library(celeg_cpu_backend ${CELEG_CPU_BACKEND_SOURCES})
target_include_directories(celeg_cpu_backend PUBLIC include)
target_include_directories(celeg_cpu_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
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
target_include_directories(celeg PUBLIC include)
target_include_directories(celeg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cuda/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cuda/model)
target_compile_definitions(celeg PRIVATE CELEG_API_BUILD=1)
set_target_properties(celeg PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    OUTPUT_NAME celeg)

add_executable(celeg-cpu-run apps/cpu/main.cpp)
target_include_directories(celeg-cpu-run PRIVATE src)
target_link_libraries(celeg-cpu-run PRIVATE celeg_cpu_backend)
add_executable(celeg-cpu-pack tools/cpu_pack.cpp)
target_link_libraries(celeg-cpu-pack PRIVATE celeg_cpu_backend)
add_executable(celeg-cpu-kernel-benchmark apps/benchmark/cpu/kernels.cpp)
target_link_libraries(celeg-cpu-kernel-benchmark PRIVATE celeg_cpu_backend)
add_executable(celeg-c-api-example examples/api_example.c)
target_link_libraries(celeg-c-api-example PRIVATE celeg)
add_executable(celeg-cpu-concurrent-benchmark apps/benchmark/cpu/concurrent.cpp)
target_include_directories(celeg-cpu-concurrent-benchmark PRIVATE src)
target_link_libraries(celeg-cpu-concurrent-benchmark PRIVATE celeg_cpu_backend)
add_executable(celeg-cpu-prefill-benchmark apps/benchmark/cpu/prefill.cpp)
target_link_libraries(celeg-cpu-prefill-benchmark PRIVATE celeg_cpu_backend)
add_executable(celeg-cpu-prefix-cache-benchmark apps/benchmark/cpu/prefix_cache.cpp)
target_include_directories(celeg-cpu-prefix-cache-benchmark PRIVATE src)
target_link_libraries(celeg-cpu-prefix-cache-benchmark PRIVATE celeg_cpu_backend)
add_executable(celeg-cpu-compare-reference tools/cpu_compare_reference.cpp)
target_link_libraries(celeg-cpu-compare-reference PRIVATE celeg_cpu_backend)
add_executable(celeg-bench apps/benchmark/cpu/celeg_bench.cpp)
target_link_libraries(celeg-bench PRIVATE celeg_cpu_backend)

add_executable(celeg-download apps/download/main.cpp)
target_link_libraries(celeg-download PRIVATE celeg_base)

if(CELEG_ENABLE_SERVE)
    include(FetchContent)
    if(POLICY CMP0218)
        cmake_policy(PUSH)
        cmake_policy(SET CMP0218 OLD)
    endif()
    set(_celeg_saved_cmake_warn_deprecated "${CMAKE_WARN_DEPRECATED}")
    set(_celeg_saved_policy_version_minimum "${CMAKE_POLICY_VERSION_MINIMUM}")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
    set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
    if(COMMAND cmake_diagnostic)
        cmake_diagnostic(PUSH)
        cmake_diagnostic(SET CMD_AUTHOR IGNORE RECURSE)
        cmake_diagnostic(SET CMD_DEPRECATED IGNORE RECURSE)
        cmake_diagnostic(SET CMD_POLICY IGNORE RECURSE)
    else()
        set(_celeg_saved_warn_deprecated "${CMAKE_WARN_DEPRECATED}")
        set(_celeg_saved_error_deprecated "${CMAKE_ERROR_DEPRECATED}")
        set(_celeg_saved_suppress_developer_warnings "${CMAKE_SUPPRESS_DEVELOPER_WARNINGS}")
        set(CMAKE_WARN_DEPRECATED OFF)
        set(CMAKE_ERROR_DEPRECATED OFF)
        set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE)
    endif()
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
    target_include_directories(celeg_serve_protocol PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(celeg_serve_protocol PRIVATE
        CELEG_OPENAPI_SPEC_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/serve/protocol/openapi.json")
    if(NOT MSVC)
        target_compile_options(celeg_serve_protocol PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    target_link_libraries(celeg_serve_protocol PUBLIC celeg_runtime glaze::glaze)

endif()

set(CELEG_METAL_AVAILABLE OFF)
if(CELEG_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "CELEG_ENABLE_METAL requires an Apple host")
    endif()
    include(CheckLanguage)
    check_language(OBJCXX)
    if(NOT CMAKE_OBJCXX_COMPILER)
        message(FATAL_ERROR "CELEG_ENABLE_METAL requires an Objective-C++ compiler")
    endif()
    enable_language(OBJCXX)
    set(CELEG_METAL_INFERENCE_SOURCES
        src/backend/metal/kernels/inference/common.metal
        src/backend/metal/kernels/inference/vector.metal
        src/backend/metal/kernels/inference/state.metal
        src/backend/metal/kernels/inference/projection.metal
        src/backend/metal/kernels/inference/batch.metal
        src/backend/metal/kernels/inference/pair.metal)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/kernels/probe.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/kernels/inference.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/kernels/tensor.metal"
        ${CELEG_METAL_INFERENCE_SOURCES})
    find_library(CELEG_METAL_FRAMEWORK Metal REQUIRED)
    find_library(CELEG_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/kernels/probe.metal"
        CELEG_METAL_PROBE_SOURCE)
    set(CELEG_METAL_INFERENCE_SOURCE "")
    foreach(CELEG_METAL_INFERENCE_SOURCE_FILE IN LISTS CELEG_METAL_INFERENCE_SOURCES)
        file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${CELEG_METAL_INFERENCE_SOURCE_FILE}"
            CELEG_METAL_INFERENCE_SOURCE_PART)
        string(APPEND CELEG_METAL_INFERENCE_SOURCE
            "${CELEG_METAL_INFERENCE_SOURCE_PART}\n")
    endforeach()
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/kernels/tensor.metal"
        CELEG_METAL_TENSOR_SOURCE)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/metal_probe_source.hpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/metal_probe_source.hpp"
        @ONLY)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/metal_inference_source.hpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/metal_inference_source.hpp"
        @ONLY)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/metal_tensor_source.hpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/metal_tensor_source.hpp"
        @ONLY)

    add_library(celeg_metal_backend ${CELEG_METAL_BACKEND_SOURCES})
    target_include_directories(celeg_metal_backend PUBLIC include)
    target_include_directories(celeg_metal_backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_compile_definitions(celeg_metal_backend PUBLIC CELEG_API_WITH_METAL=1)
    target_link_libraries(celeg_metal_backend PUBLIC
        celeg_runtime "${CELEG_METAL_FRAMEWORK}" "${CELEG_FOUNDATION_FRAMEWORK}"
        Threads::Threads)
    set_source_files_properties(src/backend/metal/runtime/device.mm
        PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set(CELEG_METAL_AVAILABLE ON)

    add_library(celeg_serve_metal src/serve/metal_inference_service.cpp)
    target_sources(celeg_serve_metal PRIVATE ${CELEG_METAL_MODEL_SOURCES})
    target_include_directories(celeg_serve_metal PUBLIC include)
    target_include_directories(celeg_serve_metal PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(celeg_serve_metal PUBLIC
        celeg_metal_backend)
    target_compile_definitions(celeg_serve_metal PUBLIC CELEG_SERVE_METAL=1)
    target_link_libraries(celeg PRIVATE celeg_serve_metal)
    if(TARGET celeg-serve)
        target_link_libraries(celeg-serve PRIVATE celeg_serve_metal)
        target_compile_definitions(celeg-serve PRIVATE CELEG_SERVE_METAL=1)
    endif()

    add_executable(celeg-metal-run apps/metal/main.cpp)
    target_link_libraries(celeg-metal-run PRIVATE celeg_serve_metal)
    add_executable(celeg-metal-bench ${CELEG_METAL_APP_SOURCES})
    target_link_libraries(celeg-metal-bench PRIVATE celeg_serve_metal)
    add_executable(celeg-metal-kernel-benchmark apps/benchmark/metal/kernels.mm)
    target_include_directories(celeg-metal-kernel-benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(celeg-metal-kernel-benchmark PRIVATE celeg_serve_metal)
endif()

add_library(celeg_serve_core src/serve/generation_dispatcher.cpp)
target_include_directories(celeg_serve_core PUBLIC include)
target_include_directories(celeg_serve_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
    target_compile_options(celeg_serve_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(celeg_serve_core PUBLIC Threads::Threads)

add_library(celeg_serve_cpu src/serve/cpu_inference_service.cpp)
target_include_directories(celeg_serve_cpu PUBLIC include)
target_include_directories(celeg_serve_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
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
    cmake_policy(PUSH)
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(usockets)
    cmake_policy(POP)
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
    cmake_policy(PUSH)
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(uwebsockets)
    cmake_policy(POP)
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
    cmake_policy(PUSH)
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(swaggerui)
    cmake_policy(POP)

    if(COMMAND cmake_diagnostic)
        cmake_diagnostic(POP)
    else()
        set(CMAKE_WARN_DEPRECATED "${_celeg_saved_warn_deprecated}")
        set(CMAKE_ERROR_DEPRECATED "${_celeg_saved_error_deprecated}")
        set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS "${_celeg_saved_suppress_developer_warnings}")
    endif()
    set(CMAKE_WARN_DEPRECATED "${_celeg_saved_cmake_warn_deprecated}" CACHE BOOL "" FORCE)
    set(CMAKE_POLICY_VERSION_MINIMUM "${_celeg_saved_policy_version_minimum}")
    if(POLICY CMP0218)
        cmake_policy(POP)
    endif()

    add_executable(celeg-serve
        apps/serve/main.cpp
        apps/serve/static_assets.cpp
        apps/serve/routes/health.cpp
        apps/serve/routes/models.cpp
        apps/serve/routes/chat_ui.cpp
        apps/serve/routes/docs.cpp
        apps/serve/routes/tokenize.cpp
        apps/serve/routes/chat_completions.cpp)
    target_include_directories(celeg-serve PRIVATE src)
    target_link_libraries(celeg-serve PRIVATE
        celeg_serve_cpu celeg_serve_core celeg_serve_protocol uwebsockets)
    if(CELEG_METAL_AVAILABLE)
        target_link_libraries(celeg-serve PRIVATE celeg_serve_metal)
        target_compile_definitions(celeg-serve PRIVATE CELEG_SERVE_METAL=1)
    endif()
    add_custom_command(TARGET celeg-serve POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf "$<TARGET_FILE_DIR:celeg-serve>/celeg-serve-assets"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:celeg-serve>/celeg-serve-assets/chat"
            "$<TARGET_FILE_DIR:celeg-serve>/celeg-serve-assets/docs"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/web/chat/dist"
            "$<TARGET_FILE_DIR:celeg-serve>/celeg-serve-assets/chat"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/apps/serve/static/docs.html"
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
target_include_directories(celeg_cuda_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cuda/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cuda/model)
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

    add_executable(celeg-run apps/cuda/main.cpp)
    target_include_directories(celeg-run PRIVATE src)
    target_link_libraries(celeg-run PRIVATE celeg_cuda_backend)
    if(TARGET celeg_cuda_runtime)
        add_dependencies(celeg-run celeg_cuda_runtime)
    endif()

    target_link_libraries(celeg PRIVATE celeg_cuda_backend)
    target_compile_definitions(celeg PRIVATE CELEG_API_WITH_CUDA=1)
    if(TARGET celeg_cuda_runtime)
        add_dependencies(celeg celeg_cuda_runtime)
    endif()

    add_executable(celeg-concurrent-benchmark apps/benchmark/cuda/concurrent.cpp)
    target_link_libraries(celeg-concurrent-benchmark PRIVATE celeg_cuda_backend)
    add_executable(celeg-prefix-cache-benchmark apps/benchmark/cuda/prefix_cache.cpp)
    target_link_libraries(celeg-prefix-cache-benchmark PRIVATE celeg_cuda_backend)
    add_executable(celeg-decode-gemv-benchmark apps/benchmark/cuda/decode_gemv.cu)
    target_include_directories(celeg-decode-gemv-benchmark PRIVATE
        src/backend/cuda src/backend/cuda/kernels)
    target_link_libraries(celeg-decode-gemv-benchmark PRIVATE celeg_cuda_backend)

    add_library(celeg_serve_cuda src/serve/cuda_inference_service.cpp)
target_include_directories(celeg_serve_cuda PUBLIC include)
target_include_directories(celeg_serve_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cuda/model)
    target_link_libraries(celeg_serve_cuda PUBLIC celeg_cuda_backend)
    target_link_libraries(celeg PRIVATE celeg_serve_cuda)
    if(TARGET celeg-serve)
        target_link_libraries(celeg-serve PRIVATE celeg_serve_cuda)
        target_compile_definitions(celeg-serve PRIVATE CELEG_SERVE_CUDA=1)
    endif()
endif()

add_executable(celeg-compare-logits tools/compare_logits.cpp)

add_executable(celeg-pack-experts tools/pack_experts.cpp)
target_link_libraries(celeg-pack-experts PRIVATE celeg_base)
target_include_directories(celeg-pack-experts PRIVATE include src)

foreach(_celeg_internal_consumer IN ITEMS
    celeg-cpu-run celeg-cpu-pack celeg-cpu-kernel-benchmark
    celeg-cpu-concurrent-benchmark celeg-cpu-prefill-benchmark
    celeg-cpu-prefix-cache-benchmark celeg-cpu-compare-reference celeg-bench
    celeg-download celeg-compare-logits celeg-pack-experts celeg-run
    celeg-concurrent-benchmark celeg-prefix-cache-benchmark
    celeg-decode-gemv-benchmark celeg-serve celeg-metal-run celeg-metal-bench
    celeg-metal-kernel-benchmark)
    if(TARGET ${_celeg_internal_consumer})
        target_include_directories(${_celeg_internal_consumer} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
    endif()
endforeach()

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
if(CELEG_METAL_AVAILABLE)
    list(APPEND CELEG_INSTALL_TARGETS celeg_metal_backend celeg_serve_metal
        celeg-metal-run celeg-metal-bench celeg-metal-kernel-benchmark)
endif()
install(TARGETS ${CELEG_INSTALL_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(CELEG_ENABLE_SERVE)
    install(DIRECTORY web/chat/dist/
        DESTINATION ${CMAKE_INSTALL_BINDIR}/celeg-serve-assets/chat)
    install(FILES apps/serve/static/docs.html
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
    include/celeg/api.h)
foreach(_celeg_public_header IN LISTS CELEG_PUBLIC_HEADERS)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_celeg_public_header}")
        message(FATAL_ERROR "Public header manifest entry is missing: ${_celeg_public_header}")
    endif()
endforeach()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/descriptors/"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/celeg/descriptors")
foreach(_celeg_public_header IN LISTS CELEG_PUBLIC_HEADERS)
    file(RELATIVE_PATH _celeg_header_relative
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/${_celeg_public_header}")
    get_filename_component(_celeg_header_directory
        "${_celeg_header_relative}" DIRECTORY)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${_celeg_public_header}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${_celeg_header_directory}")
endforeach()

include(tests/Targets.cmake)
