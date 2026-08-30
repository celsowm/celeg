foreach(test_name IN ITEMS
    gemma4_vision_test
    image_codec_test
    qwen35_vision_test)
    add_executable(${test_name} tests/vision/${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE include src tests)
    target_link_libraries(${test_name} PRIVATE celeg_base)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
