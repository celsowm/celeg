if(TARGET celeg_cuda_backend)
    add_executable(unsupported_moe_policy_test tests/quant/unsupported_moe_policy_test.cpp)
    target_include_directories(unsupported_moe_policy_test PRIVATE
        include src tests src/backend/cuda)
    target_link_libraries(unsupported_moe_policy_test PRIVATE celeg_cuda_backend)
    if(CELEG_RUN_CUDA_TESTS)
        add_test(NAME unsupported_moe_policy_test COMMAND unsupported_moe_policy_test)
    endif()

    add_executable(moe_router_materialization_test tests/quant/moe_router_materialization_test.cpp)
    target_include_directories(moe_router_materialization_test PRIVATE
        include src tests src/backend/cuda src/backend/cuda/model ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(moe_router_materialization_test PRIVATE celeg_cuda_backend)
    if(CELEG_RUN_CUDA_TESTS)
        add_test(NAME moe_router_materialization_test COMMAND moe_router_materialization_test)
    endif()
endif()
