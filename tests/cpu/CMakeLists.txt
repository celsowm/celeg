foreach(test_name IN ITEMS
    cpu_isa_test
    cpu_thread_pool_test
    cpu_topology_test
    cpu_kv_topology_test
    cpu_quantization_test
    cpu_linear_kernels_test
    cpu_segmented_linear_test
    cpu_elementwise_kernels_test
    cpu_recurrent_kernels_test
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
    cpu_mrope_test
    cpu_shared_kv_test
    cpu_projected_latent_test)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpu/${test_name}.cpp")
        add_executable(${test_name} tests/cpu/${test_name}.cpp)
        target_include_directories(${test_name} PRIVATE include src tests)
        target_link_libraries(${test_name} PRIVATE celeg_cpu_backend)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endforeach()

foreach(test_name IN ITEMS
    attention_semantics_test
    cpu_attention_gate_test
    cpu_expert_cache_test
    cpu_batch_scheduler_test
    cpu_moe_route_test)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpu/${test_name}.cpp")
        add_executable(${test_name} tests/cpu/${test_name}.cpp)
        target_include_directories(${test_name} PRIVATE
            include src src/backend/cpu tests)
        target_link_libraries(${test_name} PRIVATE
            celeg_cpu_backend Threads::Threads)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endforeach()
