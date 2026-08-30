if(TARGET celeg_metal_backend)
    add_executable(metal_c_api_smoke_test tests/metal/metal_c_api_smoke_test.c)
    target_include_directories(metal_c_api_smoke_test PRIVATE include)
    target_link_libraries(metal_c_api_smoke_test PRIVATE celeg)
    add_test(NAME metal_c_api_smoke_test COMMAND metal_c_api_smoke_test)

    foreach(test_name IN ITEMS
        metal_device_test
        metal_inference_test
        metal_linear_binding_test
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
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()

    if(TARGET celeg_serve_metal)
        target_link_libraries(metal_inference_test PRIVATE celeg_cpu_backend celeg_serve_metal)

        add_executable(metal_service_test tests/metal/metal_service_test.mm)
        target_include_directories(metal_service_test PRIVATE include src tests)
        target_link_libraries(metal_service_test PRIVATE celeg_base celeg_metal_backend celeg_serve_metal)
        add_test(NAME metal_service_test COMMAND metal_service_test)
    endif()
endif()
