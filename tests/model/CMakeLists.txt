foreach(test_name IN ITEMS
    architecture_resolution_test
    attention_backend_capabilities_test
    attention_bias_validation_test
    attention_norm_weight_plan_test
    attention_semantic_capability_test
    attention_shared_kv_contract_test
    automatic_inference_test
    cpu_attention_semantic_capability_test
    cuda_packed_attention_capability_test
    cuda_relative_bias_semantic_test
    layer_dispatch_exhaustiveness_test
    layer_inference_rule_test
    metal_shared_kv_capability_test
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

target_include_directories(layer_inference_rule_test PRIVATE src/model)
target_sources(model_compiler_test PRIVATE
    src/backend/cpu/compiler.cpp
    src/backend/cuda/compiler.cpp)
target_sources(cpu_attention_semantic_capability_test PRIVATE
    src/backend/cpu/compiler.cpp)
target_include_directories(cuda_packed_attention_capability_test PRIVATE
    src/backend/cuda)
target_sources(cuda_packed_attention_capability_test PRIVATE
    src/backend/cuda/packed/layer_program.cpp)
target_sources(cuda_relative_bias_semantic_test PRIVATE
    src/backend/cuda/compiler.cpp)

if(TARGET celeg_cuda_backend)
    add_executable(cuda_factorized_attention_binding_test
        tests/model/cuda_factorized_attention_binding_test.cpp)
    target_include_directories(cuda_factorized_attention_binding_test PRIVATE
        include src tests src/backend/cuda src/backend/cuda/model)
    target_link_libraries(cuda_factorized_attention_binding_test PRIVATE
        celeg_cuda_backend Threads::Threads)
    add_test(NAME cuda_factorized_attention_binding_test
             COMMAND cuda_factorized_attention_binding_test)

    add_executable(latent_prefill_path_test tests/model/latent_prefill_path_test.cpp)
    target_include_directories(latent_prefill_path_test PRIVATE
        include src tests src/backend/cuda/model/prefill_batched/attention)
    target_link_libraries(latent_prefill_path_test PRIVATE celeg_cuda_backend Threads::Threads)
    if(CELEG_RUN_CUDA_TESTS)
        add_test(NAME latent_prefill_path_test COMMAND latent_prefill_path_test)
    endif()

    add_executable(expert_weight_test tests/model/expert_weight_test.cpp)
    target_include_directories(expert_weight_test PRIVATE
        include src tests src/backend/cuda src/backend/cuda/model)
    target_link_libraries(expert_weight_test PRIVATE celeg_cuda_backend)
    if(CELEG_RUN_CUDA_TESTS)
        add_test(NAME expert_weight_test COMMAND expert_weight_test)
    endif()
endif()