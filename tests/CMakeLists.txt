# Test source ownership is declared separately from target registration so the
# test graph can remain readable while source paths stay centralized.
set(CELEG_TESTS_ROOT tests)

function(celeg_register_cpu_expert_cache_test)
    if(NOT CELEG_BUILD_TESTS)
        return()
    endif()
    add_executable(cpu_expert_cache_test
        tests/cpu/cpu_expert_cache_test.cpp)
    target_include_directories(cpu_expert_cache_test PRIVATE
        include src src/backend/cpu)
    target_link_libraries(cpu_expert_cache_test PRIVATE
        celeg_cpu_backend Threads::Threads)
    add_test(NAME cpu_expert_cache_test COMMAND cpu_expert_cache_test)
endfunction()

function(celeg_register_attention_semantics_test)
    if(NOT CELEG_BUILD_TESTS)
        return()
    endif()
    add_executable(attention_semantics_test
        tests/cpu/attention_semantics_test.cpp)
    target_include_directories(attention_semantics_test PRIVATE
        include src/backend/cpu src tests)
    target_link_libraries(attention_semantics_test PRIVATE
        celeg_cpu_backend Threads::Threads)
    add_test(NAME attention_semantics_test COMMAND attention_semantics_test)
endfunction()

function(celeg_register_latent_prefill_path_test)
    if(NOT CELEG_BUILD_TESTS)
        return()
    endif()
    add_executable(latent_prefill_path_test
        tests/model/latent_prefill_path_test.cpp)
    target_include_directories(latent_prefill_path_test PRIVATE
        include src src/backend/cuda/model/prefill_batched/attention)
    target_link_libraries(latent_prefill_path_test PRIVATE
        celeg_base Threads::Threads)
    add_test(NAME latent_prefill_path_test COMMAND latent_prefill_path_test)
endfunction()

function(celeg_register_program_fingerprint_test)
    if(NOT CELEG_BUILD_TESTS)
        return()
    endif()
    add_executable(program_fingerprint_test
        tests/model/program_fingerprint_test.cpp)
    target_include_directories(program_fingerprint_test PRIVATE
        include tests)
    target_link_libraries(program_fingerprint_test PRIVATE
        celeg_base Threads::Threads)
    add_test(NAME program_fingerprint_test COMMAND program_fingerprint_test)
endfunction()

function(celeg_register_layer_inference_rule_test)
    if(NOT CELEG_BUILD_TESTS)
        return()
    endif()
    add_executable(layer_inference_rule_test
        tests/model/layer_inference_rule_test.cpp)
    target_include_directories(layer_inference_rule_test PRIVATE
        include src/model tests)
    target_link_libraries(layer_inference_rule_test PRIVATE
        celeg_base Threads::Threads)
    add_test(NAME layer_inference_rule_test COMMAND layer_inference_rule_test)
endfunction()

# This manifest is included before targets are declared. Defer registration so
# celeg_cpu_backend exists and the tests remain owned by the test manifest.
cmake_language(DEFER CALL celeg_register_cpu_expert_cache_test)
cmake_language(DEFER CALL celeg_register_attention_semantics_test)
cmake_language(DEFER CALL celeg_register_latent_prefill_path_test)
cmake_language(DEFER CALL celeg_register_program_fingerprint_test)
cmake_language(DEFER CALL celeg_register_layer_inference_rule_test)
