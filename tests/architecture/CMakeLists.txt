add_executable(neutral_headers_compile_test tests/architecture/neutral_headers_compile_test.cpp)
target_include_directories(neutral_headers_compile_test PRIVATE include src)
add_test(NAME neutral_headers_compile_test COMMAND neutral_headers_compile_test)

add_executable(fake_repository_backend_boundary_test
    tests/architecture/fake_repository_backend_boundary_test.cpp
    src/backend/cpu/compiler.cpp
    src/backend/cuda/compiler.cpp)
target_include_directories(fake_repository_backend_boundary_test PRIVATE include src tests)
target_link_libraries(fake_repository_backend_boundary_test PRIVATE celeg_base)
add_test(NAME fake_repository_backend_boundary_test COMMAND fake_repository_backend_boundary_test)

add_test(NAME architecture_boundary_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_architecture_boundaries.py
            --root ${CMAKE_CURRENT_SOURCE_DIR})
