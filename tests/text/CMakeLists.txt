foreach(test_name IN ITEMS
    chat_template_resolution_test
    chat_template_test
    conversation_test
    protocol_utils_test)
    add_executable(${test_name} tests/text/${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE include src tests)
    target_link_libraries(${test_name} PRIVATE celeg_base)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
