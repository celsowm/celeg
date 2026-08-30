foreach(test_name IN ITEMS
    architecture_resolution_test
    automatic_inference_test
    layer_dispatch_exhaustiveness_test
    model_compiler_test
    policy_test
    quantization_test
    reference_test
    semantic_identity_test
    tensor_resolver_test
    weight_plan_test)
    add_executable(${test_name} tests/model/${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE include src tests)
    target_link_libraries(${test_name} PRIVATE celeg_base)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

if(TARGET celeg_cuda_backend)
    add_executable(expert_weight_test tests/model/expert_weight_test.cpp)
    target_include_directories(expert_weight_test PRIVATE
        include src tests src/backend/cuda)
    target_link_libraries(expert_weight_test PRIVATE celeg_cuda_backend)
    if(CELEG_RUN_CUDA_TESTS)
        add_test(NAME expert_weight_test COMMAND expert_weight_test)
    endif()
endif()
