# CUDA 基础层：构建错误契约、内存、stream、event 和真实设备测试。
# 该目录独立于 CPU 默认回归，只有 GPU 学习路径显式启用。
cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build")
endif()

project(cpp_llm_engine_cuda LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT TARGET cpp_llm_engine_core)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../core
        ${CMAKE_CURRENT_BINARY_DIR}/core
    )
endif()

find_package(CUDAToolkit REQUIRED)

add_library(cpp_llm_engine_cuda
    src/cublas_matmul_backend.cpp
    src/cublas_status.cpp
    src/cuda_event.cpp
    src/cuda_execution_context.cpp
    src/cuda_decoder_batch_layer.cpp
    src/cuda_decoder_layer.cpp
    src/cuda_kv_cache.cpp
    src/cuda_paged_kv_cache.cpp
    src/cuda_quantized_matrix.cpp
    src/cuda_status.cpp
    src/cuda_stream.cpp
    src/cuda_tensor.cpp
    src/cuda_workspace.cpp
    src/device_buffer.cpp
    src/decoder_elementwise_backend.cu
    src/elementwise_backend.cu
    src/gqa_attention_backend.cu
    src/pinned_host_buffer.cpp
    src/quantized_dequantize_backend.cu
    src/quantized_gemv_backend.cu
    src/rope_backend.cu
)
target_include_directories(cpp_llm_engine_cuda
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(cpp_llm_engine_cuda PUBLIC cxx_std_17)
target_link_libraries(cpp_llm_engine_cuda
    PUBLIC cpp_llm_engine_core CUDA::cudart CUDA::cublas
)

add_executable(cpp_llm_engine_cuda_test
    tests/cuda_runtime_test.cu
)
target_link_libraries(cpp_llm_engine_cuda_test PRIVATE cpp_llm_engine_cuda)

add_executable(cpp_llm_engine_cuda_elementwise_test
    tests/cuda_elementwise_test.cu
)
target_link_libraries(
    cpp_llm_engine_cuda_elementwise_test PRIVATE cpp_llm_engine_cuda)

add_executable(cpp_llm_engine_cuda_cublas_matmul_test
    tests/cuda_cublas_matmul_test.cu
)
target_link_libraries(
    cpp_llm_engine_cuda_cublas_matmul_test PRIVATE cpp_llm_engine_cuda)

add_executable(cpp_llm_engine_cuda_execution_context_test
    tests/cuda_execution_context_test.cu
)
target_link_libraries(
    cpp_llm_engine_cuda_execution_context_test PRIVATE cpp_llm_engine_cuda)

add_executable(cpp_llm_engine_cuda_quantized_linear_test
    tests/cuda_quantized_linear_test.cu
    ../chapter05_quantization/src/q4k_dequantize.cpp
    ../chapter05_quantization/src/q6k_dequantize.cpp
)
target_include_directories(cpp_llm_engine_cuda_quantized_linear_test
    PRIVATE ../chapter05_quantization/include
)
target_link_libraries(
    cpp_llm_engine_cuda_quantized_linear_test PRIVATE cpp_llm_engine_cuda)

add_executable(cpp_llm_engine_cuda_rope_test
    tests/cuda_rope_test.cu
)
target_link_libraries(
    cpp_llm_engine_cuda_rope_test PRIVATE cpp_llm_engine_cuda)

add_executable(cpp_llm_engine_cuda_kv_cache_test
    tests/cuda_kv_cache_test.cu
)
target_link_libraries(
    cpp_llm_engine_cuda_kv_cache_test PRIVATE cpp_llm_engine_cuda)

if(NOT TARGET chapter07_kv_cache)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../chapter07_kv_cache
        ${CMAKE_CURRENT_BINARY_DIR}/chapter07_kv_cache
    )
endif()
add_executable(cpp_llm_engine_cuda_paged_kv_cache_test
    tests/cuda_paged_kv_cache_test.cu
)
target_link_libraries(
    cpp_llm_engine_cuda_paged_kv_cache_test
    PRIVATE cpp_llm_engine_cuda chapter07_kv_cache)

add_executable(cpp_llm_engine_cuda_gqa_attention_test
    tests/cuda_gqa_attention_test.cu
)
target_link_libraries(
    cpp_llm_engine_cuda_gqa_attention_test PRIVATE cpp_llm_engine_cuda)

add_executable(cpp_llm_engine_cuda_decoder_elementwise_test
    tests/cuda_decoder_elementwise_test.cu
)
target_link_libraries(
    cpp_llm_engine_cuda_decoder_elementwise_test PRIVATE cpp_llm_engine_cuda)

add_executable(cpp_llm_engine_cuda_decoder_layer_test
    tests/cuda_decoder_layer_test.cu
)
target_link_libraries(
    cpp_llm_engine_cuda_decoder_layer_test PRIVATE cpp_llm_engine_cuda)

cpp_llm_prepare_test(cpp_llm_engine_cuda_test)
cpp_llm_prepare_test(cpp_llm_engine_cuda_elementwise_test)
cpp_llm_prepare_test(cpp_llm_engine_cuda_cublas_matmul_test)
cpp_llm_prepare_test(cpp_llm_engine_cuda_execution_context_test)
cpp_llm_prepare_test(cpp_llm_engine_cuda_quantized_linear_test)
cpp_llm_prepare_test(cpp_llm_engine_cuda_rope_test)
cpp_llm_prepare_test(cpp_llm_engine_cuda_kv_cache_test)
cpp_llm_prepare_test(cpp_llm_engine_cuda_paged_kv_cache_test)
cpp_llm_prepare_test(cpp_llm_engine_cuda_gqa_attention_test)
cpp_llm_prepare_test(cpp_llm_engine_cuda_decoder_elementwise_test)
cpp_llm_prepare_test(cpp_llm_engine_cuda_decoder_layer_test)

include(CTest)
enable_testing()
add_test(NAME cpp_llm_engine_cuda_test COMMAND cpp_llm_engine_cuda_test)
add_test(
    NAME cpp_llm_engine_cuda_elementwise_test
    COMMAND cpp_llm_engine_cuda_elementwise_test)
add_test(
    NAME cpp_llm_engine_cuda_cublas_matmul_test
    COMMAND cpp_llm_engine_cuda_cublas_matmul_test)
add_test(
    NAME cpp_llm_engine_cuda_execution_context_test
    COMMAND cpp_llm_engine_cuda_execution_context_test)
add_test(
    NAME cpp_llm_engine_cuda_quantized_linear_test
    COMMAND cpp_llm_engine_cuda_quantized_linear_test)
add_test(
    NAME cpp_llm_engine_cuda_rope_test
    COMMAND cpp_llm_engine_cuda_rope_test)
add_test(
    NAME cpp_llm_engine_cuda_kv_cache_test
    COMMAND cpp_llm_engine_cuda_kv_cache_test)
add_test(
    NAME cpp_llm_engine_cuda_paged_kv_cache_test
    COMMAND cpp_llm_engine_cuda_paged_kv_cache_test)
add_test(
    NAME cpp_llm_engine_cuda_gqa_attention_test
    COMMAND cpp_llm_engine_cuda_gqa_attention_test)
add_test(
    NAME cpp_llm_engine_cuda_decoder_elementwise_test
    COMMAND cpp_llm_engine_cuda_decoder_elementwise_test)
add_test(
    NAME cpp_llm_engine_cuda_decoder_layer_test
    COMMAND cpp_llm_engine_cuda_decoder_layer_test)
cpp_llm_set_test_labels(cpp_llm_engine_cuda_test unit cuda runtime)
cpp_llm_set_test_labels(cpp_llm_engine_cuda_elementwise_test unit cuda operator)
cpp_llm_set_test_labels(cpp_llm_engine_cuda_cublas_matmul_test integration cuda matmul)
cpp_llm_set_test_labels(cpp_llm_engine_cuda_execution_context_test contract cuda runtime)
cpp_llm_set_test_labels(cpp_llm_engine_cuda_quantized_linear_test integration cuda quantization)
cpp_llm_set_test_labels(cpp_llm_engine_cuda_rope_test unit cuda rope)
cpp_llm_set_test_labels(cpp_llm_engine_cuda_kv_cache_test component cuda cache)
cpp_llm_set_test_labels(
    cpp_llm_engine_cuda_paged_kv_cache_test integration cuda cache paged)
cpp_llm_set_test_labels(cpp_llm_engine_cuda_gqa_attention_test component cuda attention)
cpp_llm_set_test_labels(cpp_llm_engine_cuda_decoder_elementwise_test component cuda decoder)
cpp_llm_set_test_labels(cpp_llm_engine_cuda_decoder_layer_test integration cuda decoder)
