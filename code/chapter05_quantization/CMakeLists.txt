cmake_minimum_required(VERSION 3.20)
project(chapter05_quantization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

if(NOT TARGET cpp_llm_engine_core)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../core
        ${CMAKE_CURRENT_BINARY_DIR}/core
    )
endif()

if(NOT TARGET chapter02_tensor_ops)
    set(CHAPTER02_BUILD_STANDALONE_TARGETS OFF)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../chapter02_tensor_ops
        ${CMAKE_CURRENT_BINARY_DIR}/chapter02_tensor_ops
        EXCLUDE_FROM_ALL
    )
    unset(CHAPTER02_BUILD_STANDALONE_TARGETS)
endif()

add_library(chapter05_quantization
    src/dequantize_op.cpp
    src/linear_layer.cpp
    src/quantized_linear.cpp
    src/quantized_linear_view_adapter.cpp
    src/q4k_dequantize.cpp
    src/q4k_kernel.cpp
    src/q4k_linear.cpp
    src/q6k_dequantize.cpp
    src/q6k_kernel.cpp
    src/q6k_linear.cpp
    src/q8_0_dequantize.cpp
    src/q8_0_linear.cpp
    src/quantized_linear_policy.cpp
    src/tensor.cpp
)

target_include_directories(chapter05_quantization
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chapter05_quantization
    PUBLIC
        cpp_llm_engine_core
)

target_compile_features(chapter05_quantization PUBLIC cxx_std_17)

add_executable(chapter05_quantization_demo
    examples/main_quantization.cpp
)

add_executable(q4k_benchmark
    examples/q4k_benchmark.cpp
)

target_link_libraries(q4k_benchmark
    PRIVATE chapter05_quantization
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    add_library(chapter05_quantization_avx2
        src/q4k_linear_avx2.cpp
        src/q6k_linear_avx2.cpp
    )
    target_include_directories(chapter05_quantization_avx2
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_options(chapter05_quantization_avx2 PRIVATE -mavx2 -mfma)
    target_link_libraries(chapter05_quantization_avx2 PUBLIC chapter05_quantization)
    target_compile_definitions(q4k_benchmark PRIVATE CHAPTER05_HAS_AVX2=1)
    target_link_libraries(q4k_benchmark PRIVATE chapter05_quantization_avx2)
endif()

target_link_libraries(chapter05_quantization_demo
    PRIVATE chapter05_quantization
)

add_executable(chapter05_quantization_test
    tests/chapter05_quantization_test.cpp
)

add_executable(chapter02_tensor_quantized_linear_integration_test
    tests/chapter02_tensor_quantized_linear_integration_test.cpp
)

target_link_libraries(chapter05_quantization_test
    PRIVATE chapter05_quantization
)
cpp_llm_prepare_test(chapter05_quantization_test)

target_link_libraries(chapter02_tensor_quantized_linear_integration_test
    PRIVATE chapter02_tensor_ops chapter05_quantization
)
cpp_llm_prepare_test(chapter02_tensor_quantized_linear_integration_test)

if(TARGET chapter05_quantization_avx2)
    target_compile_definitions(chapter05_quantization_test PRIVATE CHAPTER05_HAS_AVX2=1)
    target_link_libraries(chapter05_quantization_test PRIVATE chapter05_quantization_avx2)
endif()

include(CTest)
enable_testing()

add_test(NAME chapter05_quantization_test COMMAND chapter05_quantization_test)
add_test(
    NAME chapter02_tensor_quantized_linear_integration_test
    COMMAND chapter02_tensor_quantized_linear_integration_test
)
cpp_llm_set_test_labels(chapter05_quantization_test unit quantization)
cpp_llm_set_test_labels(
    chapter02_tensor_quantized_linear_integration_test integration tensor quantization)
