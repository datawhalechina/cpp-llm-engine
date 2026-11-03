# Chapter 06：构建 Generation Runtime、示例和专项测试。
# 目标：让 Request、KV Cache、Scheduler 和 Runtime 保持可独立验证。
cmake_minimum_required(VERSION 3.20)
project(chapter06_generation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()
include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

add_library(chapter06_generation
    src/request.cpp
    src/sampler.cpp
    src/kv_cache.cpp
    src/runtime.cpp
    src/scheduler.cpp
    src/embedding.cpp
    src/position_encoding.cpp
    src/decoder_layer.cpp
    src/paged_kv_cache.cpp
    src/serving_preview.cpp
    src/transformer.cpp
)

target_include_directories(chapter06_generation
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chapter06_generation PUBLIC cxx_std_17)

function(add_chapter06_test target source labels)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE chapter06_generation)
    cpp_llm_prepare_test(${target})
    add_test(NAME ${target} COMMAND ${target})
    cpp_llm_set_test_labels(${target} ${labels})
endfunction()

add_executable(chapter06_generation_demo
    examples/main_generation.cpp
)

target_link_libraries(chapter06_generation_demo
    PRIVATE chapter06_generation
)

add_chapter06_test(
    chapter06_generation_test
    tests/chapter06_generation_test.cpp
    "unit;generation"
)
add_chapter06_test(
    chapter06_sampler_test
    tests/sampler_test.cpp
    "unit;sampler"
)
add_chapter06_test(
    chapter06_kv_cache_test
    tests/kv_cache_test.cpp
    "unit;cache"
)
add_chapter06_test(
    chapter06_scheduler_test
    tests/chapter06_scheduler_test.cpp
    "component;scheduler"
)
add_chapter06_test(
    chapter06_continuous_batch_scheduler_test
    tests/continuous_batch_scheduler_test.cpp
    "component;scheduler;batch"
)
add_chapter06_test(
    chapter06_block_pool_scheduling_test
    tests/block_pool_scheduling_test.cpp
    "component;scheduler;cache"
)
add_chapter06_test(
    chapter06_chunked_prefill_test
    tests/chunked_prefill_test.cpp
    "component;scheduler;cache;prefill"
)
add_chapter06_test(
    chapter06_execution_transaction_test
    tests/execution_transaction_test.cpp
    "contract;scheduler;cache;transaction"
)
add_chapter06_test(
    chapter06_scheduler_fairness_test
    tests/scheduler_fairness_test.cpp
    "contract;scheduler;fairness"
)
add_chapter06_test(
    chapter06_tensor_batch_test
    tests/tensor_batch_test.cpp
    "component;batch"
)
add_chapter06_test(
    chapter06_paged_kv_cache_test
    tests/paged_kv_cache_test.cpp
    "component;cache"
)
add_chapter06_test(
    chapter06_request_contract_test
    tests/request_contract_test.cpp
    "contract;request"
)
add_chapter06_test(
    chapter06_kv_block_pool_contract_test
    tests/kv_block_pool_contract_test.cpp
    "contract;cache"
)

add_executable(chapter06_streaming_demo
    examples/main_streaming.cpp
)

target_link_libraries(chapter06_streaming_demo
    PRIVATE chapter06_generation
)
