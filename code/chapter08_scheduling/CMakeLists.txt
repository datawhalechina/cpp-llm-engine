# Chapter 08：构建独立调度、批处理与跨章缓存事务测试。
cmake_minimum_required(VERSION 3.20)
project(chapter08_scheduling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()
include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

if(NOT TARGET chapter06_generation)
    add_subdirectory(
        ${CMAKE_CURRENT_LIST_DIR}/../chapter06_generation
        ${CMAKE_CURRENT_BINARY_DIR}/chapter06_generation
    )
endif()
if(NOT TARGET chapter07_kv_cache)
    add_subdirectory(
        ${CMAKE_CURRENT_LIST_DIR}/../chapter07_kv_cache
        ${CMAKE_CURRENT_BINARY_DIR}/chapter07_kv_cache
    )
endif()

add_library(chapter08_scheduling
    src/overlap_scheduler.cpp
    src/scheduler.cpp
    src/step_scheduler.cpp
)
target_include_directories(chapter08_scheduling
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(chapter08_scheduling
    PUBLIC
        chapter06_generation
        chapter07_kv_cache
)
target_compile_features(chapter08_scheduling PUBLIC cxx_std_17)

function(add_chapter08_test target source labels level)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE chapter08_scheduling)
    cpp_llm_prepare_test(${target})
    add_test(NAME ${target} COMMAND ${target})
    cpp_llm_set_test_labels(${target} ${labels})
    cpp_llm_set_test_contract(${target} OWNER 08 LEVEL ${level} STATUS active)
endfunction()

add_chapter08_test(
    chapter08_scheduler_test
    tests/scheduler_test.cpp
    "scheduler"
    component
)
add_chapter08_test(
    chapter08_overlap_scheduler_test
    tests/overlap_scheduler_test.cpp
    "scheduler;overlap;contract"
    contract
)
add_chapter08_test(
    chapter08_step_scheduler_test
    tests/step_scheduler_test.cpp
    "scheduler;batch"
    component
)
add_chapter08_test(
    chapter08_priority_aging_test
    tests/priority_aging_test.cpp
    "scheduler;priority;fairness;contract"
    contract
)
add_chapter08_test(
    chapter08_cache_transaction_test
    tests/cache_transaction_test.cpp
    "scheduler;cache;transaction"
    integration
)
add_chapter08_test(
    chapter08_tensor_batch_test
    tests/tensor_batch_test.cpp
    "scheduler;batch;tensor"
    integration
)
