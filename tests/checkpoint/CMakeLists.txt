foreach(test_name IN ITEMS
    checkpoint_capabilities_test
    checkpoint_format_catalog_test
    gguf_parse_test
    gguf_repository_test
    json_test
    safetensors_repo_test
    safetensors_test
    tensor_codec_test
    tokenizer_test)
    add_executable(${test_name} tests/checkpoint/${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE include src tests)
    target_link_libraries(${test_name} PRIVATE celeg_base)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

target_include_directories(gguf_repository_test PRIVATE
    src/checkpoint/repositories/detail)
