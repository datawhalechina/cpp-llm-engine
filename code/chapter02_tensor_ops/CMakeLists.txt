# Chapter 02：Tensor view 构建入口
# 构建：生成 Tensor library、demo 和 CTest；不依赖外部模型或 GPU。
cmake_minimum_required(VERSION 3.20)
project(chapter02_tensor_ops LANGUAGES CXX)

option(
    CHAPTER02_BUILD_STANDALONE_TARGETS
    "Build the Chapter 02 demo and standalone test"
    ON
)

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

add_library(chapter02_tensor_ops
    src/tensor.cpp
)

target_include_directories(chapter02_tensor_ops
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chapter02_tensor_ops PUBLIC cxx_std_17)
target_link_libraries(chapter02_tensor_ops PUBLIC cpp_llm_engine_core)

if(CHAPTER02_BUILD_STANDALONE_TARGETS)
    add_executable(chapter02_tensor_ops_demo
        examples/main_tensor_ops.cpp
    )

    target_link_libraries(chapter02_tensor_ops_demo
        PRIVATE chapter02_tensor_ops
    )

    add_executable(chapter02_tensor_ops_test
        tests/chapter02_tensor_ops_test.cpp
    )

    target_link_libraries(chapter02_tensor_ops_test
        PRIVATE chapter02_tensor_ops
    )
    cpp_llm_prepare_test(chapter02_tensor_ops_test)

    add_executable(chapter02_nd_tensor_test
        tests/chapter02_nd_tensor_test.cpp
    )

    target_link_libraries(chapter02_nd_tensor_test
        PRIVATE chapter02_tensor_ops
    )
    cpp_llm_prepare_test(chapter02_nd_tensor_test)

    include(CTest)
    enable_testing()

    add_test(NAME chapter02_tensor_ops_test COMMAND chapter02_tensor_ops_test)
    add_test(NAME chapter02_nd_tensor_test COMMAND chapter02_nd_tensor_test)
    cpp_llm_set_test_labels(chapter02_tensor_ops_test unit tensor)
    cpp_llm_set_test_labels(chapter02_nd_tensor_test contract tensor)
endif()
