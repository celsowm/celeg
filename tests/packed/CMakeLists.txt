foreach(test_name IN ITEMS
    fp8_test
    int4_test
    int8_test
    nvfp4_test)
    add_executable(${test_name} tests/packed/${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE include src tests)
    target_link_libraries(${test_name} PRIVATE celeg_base)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

if(CELEG_ENABLE_CUDA)
    foreach(test_name IN ITEMS commit_test workspace_test)
        add_executable(${test_name} tests/packed/${test_name}.cpp)
        target_include_directories(${test_name} PRIVATE include src tests src/backend/cuda)
        target_link_libraries(${test_name} PRIVATE celeg_base celeg_cuda_backend)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()
