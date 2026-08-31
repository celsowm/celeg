if(TARGET celeg_cuda_backend)
    set(CELEG_CUDA_TEST_PRIVATE_INCLUDE_DIRS
        include
        src
        tests
        src/backend/cuda
        src/backend/cuda/kernels)

    add_executable(cuda_attention_capability_test
        tests/cuda/cuda_attention_capability_test.cpp)
    target_include_directories(cuda_attention_capability_test PRIVATE
        ${CELEG_CUDA_TEST_PRIVATE_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(cuda_attention_capability_test PRIVATE celeg_base)
    add_test(NAME cuda_attention_capability_test COMMAND cuda_attention_capability_test)

    add_executable(cuda_model_options_test tests/cuda/cuda_model_options_test.cpp)
    target_include_directories(cuda_model_options_test PRIVATE include src tests)
    target_link_libraries(cuda_model_options_test PRIVATE celeg_base)
    add_test(NAME cuda_model_options_test COMMAND cuda_model_options_test)

    add_executable(cuda_kernels_test
        tests/cuda/cuda_kernels_test.cu
        tests/cuda/sampling_tests.cu
        tests/cuda/attention_tests.cu)
    target_include_directories(cuda_kernels_test PRIVATE
        ${CELEG_CUDA_TEST_PRIVATE_INCLUDE_DIRS})
    target_link_libraries(cuda_kernels_test PRIVATE
        celeg_cuda_backend celeg_cpu_backend)
    if(CELEG_RUN_CUDA_TESTS)
        add_test(NAME cuda_kernels_test COMMAND cuda_kernels_test)
    endif()

    foreach(test_name IN ITEMS
        cuda_bidirectional_attention_test
        cuda_block_sparse_attention_test
        cuda_dynamic_sparse_attention_test
        cuda_gguf_kernels_test
        moe_route_test
        moe_ffn_test
        moe_integration_test
        expert_residency_test
        cuda_granite_test)
        add_executable(${test_name} tests/cuda/${test_name}.cu)
        target_include_directories(${test_name} PRIVATE
            ${CELEG_CUDA_TEST_PRIVATE_INCLUDE_DIRS})
        target_link_libraries(${test_name} PRIVATE celeg_cuda_backend)
        if(CELEG_RUN_CUDA_TESTS)
            add_test(NAME ${test_name} COMMAND ${test_name})
        endif()
    endforeach()
endif()
