foreach(test_name IN ITEMS
    architecture_resolution_test
    automatic_inference_test
    layer_dispatch_exhaustiveness_test
    layer_inference_rule_test
    model_compiler_test
    policy_test
    program_fingerprint_test
    quantization_test
    reference_test
    semantic_identity_test
    tensor_resolver_test
    weight_plan_test)
    add_executable(${test_name} tests/model/${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE include src tests)
    target_link_libraries(${test_name} PRIVATE celeg_base Threads::Threads)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

add_executable(latent_prefill_path_test tests/model/latent_prefill_path_test.cpp)
target_include_directories(latent_prefill_path_test PRIVATE
    include src tests src/backend/cuda/model/prefill_batched/attention)
target_link_libraries(latent_prefill_path_test PRIVATE celeg_base Threads::Threads)
add_test(NAME latent_prefill_path_test COMMAND latent_prefill_path_test)

if(TARGET celeg_cuda_backend)
    add_executable(expert_weight_test tests/model/expert_weight_test.cpp)
    target_include_directories(expert_weight_test PRIVATE
        include src tests src/backend/cuda)
    target_link_libraries(expert_weight_test PRIVATE celeg_cuda_backend)
    if(CELEG_RUN_CUDA_TESTS)
        add_test(NAME expert_weight_test COMMAND expert_weight_test)
    endif()
endif()
