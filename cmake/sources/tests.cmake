# Test source ownership is declared separately from target registration so the
# test graph can remain readable while source paths stay centralized.
set(CELEG_TESTS_ROOT tests)

function(celeg_register_cpu_expert_cache_test)
    if(NOT CELEG_BUILD_TESTS)
        return()
    endif()
    add_executable(cpu_expert_cache_test
        tests/cpu_expert_cache_test.cpp)
    target_include_directories(cpu_expert_cache_test PRIVATE
        include src/backend/cpu)
    target_link_libraries(cpu_expert_cache_test PRIVATE
        celeg_cpu_backend Threads::Threads)
    add_test(NAME cpu_expert_cache_test COMMAND cpu_expert_cache_test)
endfunction()

# This manifest is included before targets are declared. Defer registration so
# celeg_cpu_backend exists and the test remains owned by the test manifest.
cmake_language(DEFER CALL celeg_register_cpu_expert_cache_test)
