if(TARGET celeg_metal_backend)
    add_executable(metal_c_api_smoke_test tests/metal/metal_c_api_smoke_test.c)
    target_include_directories(metal_c_api_smoke_test PRIVATE include)
    target_link_libraries(metal_c_api_smoke_test PRIVATE celeg)

    foreach(test_name IN ITEMS
        metal_attention_gate_test
        metal_device_test
        metal_inference_test
        metal_linear_binding_test
        metal_normalization_test
        metal_quantization_test
        metal_recurrent_test
        metal_tensor_test)
        set(test_source tests/metal/${test_name}.mm)
        if(test_name STREQUAL "metal_linear_binding_test")
            set(test_source tests/metal/${test_name}.cpp)
        endif()
        add_executable(${test_name} ${test_source})
        target_include_directories(${test_name} PRIVATE
            include src tests src/backend/metal "${CMAKE_BINARY_DIR}/generated")
        target_link_libraries(${test_name} PRIVATE celeg_base celeg_metal_backend)
        if(CELEG_RUN_METAL_TESTS AND
           NOT test_name STREQUAL "metal_inference_test" AND
           NOT test_name STREQUAL "metal_quantization_test")
            add_test(NAME ${test_name} COMMAND ${test_name})
        endif()
    endforeach()

    add_executable(metal_rope_table_benchmark apps/benchmark/metal/rope_table.mm)
    target_include_directories(metal_rope_table_benchmark PRIVATE
        include src tests src/backend/metal "${CMAKE_BINARY_DIR}/generated")
    target_link_libraries(metal_rope_table_benchmark PRIVATE celeg_base celeg_metal_backend)

    if(TARGET celeg_serve_metal)
        target_link_libraries(metal_inference_test PRIVATE celeg_cpu_backend celeg_serve_metal)

        add_executable(metal_service_test tests/metal/metal_service_test.mm)
        target_include_directories(metal_service_test PRIVATE include src tests)
        target_link_libraries(metal_service_test PRIVATE celeg_base celeg_metal_backend celeg_serve_metal)
    endif()
endif()
