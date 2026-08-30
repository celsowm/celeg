foreach(test_name IN ITEMS
    cpu_isa_test
    cpu_thread_pool_test
    cpu_topology_test
    cpu_kv_topology_test
    cpu_quantization_test
    cpu_kernels_test
    cpu_gguf_kernels_test
    cpu_kv_cache_test
    cpu_paged_kv_test
    cpu_parallel_attention_test
    cpu_prefix_cache_test
    cpu_numa_test
    cpu_concurrent_metrics_test
    cpu_model_options_test
    cpu_sampler_test
    cpu_granite_test
    cpu_mrope_test)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpu/${test_name}.cpp")
        add_executable(${test_name} tests/cpu/${test_name}.cpp)
        target_include_directories(${test_name} PRIVATE include src tests)
        target_link_libraries(${test_name} PRIVATE celeg_cpu_backend)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endforeach()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpu/cpu_batch_scheduler_test.cpp")
    add_executable(cpu_batch_scheduler_test tests/cpu/cpu_batch_scheduler_test.cpp)
    target_include_directories(cpu_batch_scheduler_test PRIVATE include src src/backend/cpu tests)
    target_link_libraries(cpu_batch_scheduler_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_batch_scheduler_test COMMAND cpu_batch_scheduler_test)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpu/cpu_moe_route_test.cpp")
    add_executable(cpu_moe_route_test tests/cpu/cpu_moe_route_test.cpp)
    target_include_directories(cpu_moe_route_test PRIVATE include src src/backend/cpu tests)
    target_link_libraries(cpu_moe_route_test PRIVATE celeg_cpu_backend)
    add_test(NAME cpu_moe_route_test COMMAND cpu_moe_route_test)
endif()
