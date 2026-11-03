cmake_minimum_required(VERSION 3.20)
project(chapter07_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../chapter06_generation
    ${CMAKE_CURRENT_BINARY_DIR}/chapter06_generation
)

add_library(chapter07_benchmark
    src/benchmark.cpp
    src/test_helpers.cpp
)

target_include_directories(chapter07_benchmark
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chapter07_benchmark PUBLIC cxx_std_17)

add_executable(chapter07_benchmark_demo
    examples/main_benchmark.cpp
)

add_executable(chapter07_runtime_benchmark
    examples/runtime_benchmark.cpp
)

add_executable(chapter07_multi_request_benchmark
    examples/multi_request_benchmark.cpp
)

if(CMAKE_BUILD_TYPE)
    set(CHAPTER07_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
else()
    set(CHAPTER07_BUILD_TYPE "unspecified")
endif()

target_compile_definitions(chapter07_multi_request_benchmark
    PRIVATE
        CHAPTER07_BUILD_TYPE="${CHAPTER07_BUILD_TYPE}"
        CHAPTER07_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        CHAPTER07_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)

target_link_libraries(chapter07_runtime_benchmark
    PRIVATE chapter06_generation
)

target_link_libraries(chapter07_multi_request_benchmark
    PRIVATE chapter06_generation
)

target_link_libraries(chapter07_benchmark_demo
    PRIVATE chapter07_benchmark
)

add_executable(chapter07_benchmark_test
    tests/chapter07_benchmark_test.cpp
)

target_link_libraries(chapter07_benchmark_test
    PRIVATE chapter07_benchmark
)
cpp_llm_prepare_test(chapter07_benchmark_test)

include(CTest)
enable_testing()

add_test(NAME chapter07_benchmark_test COMMAND chapter07_benchmark_test)
cpp_llm_set_test_labels(chapter07_benchmark_test unit benchmark)
add_test(
    NAME chapter07_multi_request_result_schema
    COMMAND ${CMAKE_COMMAND}
        -DBENCHMARK_EXECUTABLE=$<TARGET_FILE:chapter07_multi_request_benchmark>
        -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/chapter07_multi_request_test.json
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_multi_request_result.cmake
)
cpp_llm_set_test_labels(
    chapter07_multi_request_result_schema contract benchmark result-schema)
