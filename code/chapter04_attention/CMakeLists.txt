# Chapter 04：Attention/Decoder CPU 构建入口
# 构建：生成 core、Attention library、Decoder demo 和联合 CTest。
cmake_minimum_required(VERSION 3.20)
project(chapter04_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CMAKE_CURRENT_BINARY_DIR}/core
)

if(NOT TARGET chapter02_tensor_ops)
    set(CHAPTER02_BUILD_STANDALONE_TARGETS OFF)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../chapter02_tensor_ops
        ${CMAKE_CURRENT_BINARY_DIR}/chapter02_tensor_ops
        EXCLUDE_FROM_ALL
    )
    unset(CHAPTER02_BUILD_STANDALONE_TARGETS)
endif()

add_library(chapter04_attention
    src/attention.cpp
    src/attention_backend.cpp
    src/attention_kernel.cpp
    src/decoder_context.cpp
    src/decoder_layer.cpp
    src/feed_forward.cpp
    src/layer_norm.cpp
    src/linear_layer.cpp
    src/mla_attention.cpp
    src/moe_feed_forward.cpp
    src/kv_cache.cpp
    src/tensor.cpp
    src/tensor_attention_adapter.cpp
)

target_include_directories(chapter04_attention
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chapter04_attention
    PUBLIC
        cpp_llm_engine_core
        chapter02_tensor_ops
)

target_compile_features(chapter04_attention PUBLIC cxx_std_17)

add_executable(chapter04_attention_demo
    examples/main_attention.cpp
)

target_link_libraries(chapter04_attention_demo
    PRIVATE chapter04_attention
)

add_executable(chapter04_attention_test
    tests/chapter04_attention_test.cpp
)

add_executable(chapter04_online_softmax_test
    tests/online_softmax_test.cpp
)

add_executable(chapter04_moe_feed_forward_test
    tests/moe_feed_forward_test.cpp
)

add_executable(chapter04_mla_attention_test
    tests/mla_attention_test.cpp
)

add_executable(chapter02_tensor_attention_integration_test
    tests/chapter02_tensor_attention_integration_test.cpp
)

target_link_libraries(chapter02_tensor_attention_integration_test
    PRIVATE chapter04_attention
)
cpp_llm_prepare_test(chapter02_tensor_attention_integration_test)

target_link_libraries(chapter04_attention_test
    PRIVATE chapter04_attention
)
cpp_llm_prepare_test(chapter04_attention_test)

target_link_libraries(chapter04_online_softmax_test
    PRIVATE chapter04_attention
)
cpp_llm_prepare_test(chapter04_online_softmax_test)

target_link_libraries(chapter04_moe_feed_forward_test
    PRIVATE chapter04_attention
)
cpp_llm_prepare_test(chapter04_moe_feed_forward_test)

target_link_libraries(chapter04_mla_attention_test
    PRIVATE chapter04_attention
)
cpp_llm_prepare_test(chapter04_mla_attention_test)

include(CTest)
enable_testing()

add_test(NAME chapter04_attention_test COMMAND chapter04_attention_test)
add_test(NAME chapter04_online_softmax_test COMMAND chapter04_online_softmax_test)
add_test(NAME chapter04_moe_feed_forward_test COMMAND chapter04_moe_feed_forward_test)
add_test(NAME chapter04_mla_attention_test COMMAND chapter04_mla_attention_test)
add_test(
    NAME chapter02_tensor_attention_integration_test
    COMMAND chapter02_tensor_attention_integration_test
)
cpp_llm_set_test_labels(chapter04_attention_test attention decoder)
cpp_llm_set_test_contract(
    chapter04_attention_test OWNER 04 LEVEL component STATUS active)
cpp_llm_set_test_labels(chapter04_online_softmax_test attention softmax online)
cpp_llm_set_test_contract(
    chapter04_online_softmax_test OWNER 04 LEVEL component STATUS active)
cpp_llm_set_test_labels(chapter04_moe_feed_forward_test moe router ffn)
cpp_llm_set_test_contract(
    chapter04_moe_feed_forward_test OWNER 04 LEVEL component STATUS active)
cpp_llm_set_test_labels(chapter04_mla_attention_test attention mla cache)
cpp_llm_set_test_contract(
    chapter04_mla_attention_test OWNER 04 LEVEL component STATUS active)
cpp_llm_set_test_labels(
    chapter02_tensor_attention_integration_test tensor attention)
cpp_llm_set_test_contract(
    chapter02_tensor_attention_integration_test
    OWNER 04 LEVEL integration STATUS active)
