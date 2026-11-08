# Chapter 07：构建独立的分页 K/V 控制面、数据面和专项测试。
cmake_minimum_required(VERSION 3.20)
project(chapter07_kv_cache LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()
include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

add_library(chapter07_kv_cache
    src/block_table.cpp
    src/paged_kv_cache.cpp
    src/radix_prefix_cache.cpp
)

target_include_directories(chapter07_kv_cache
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(chapter07_kv_cache PUBLIC cxx_std_17)

function(add_chapter07_test target source labels level)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE chapter07_kv_cache)
    cpp_llm_prepare_test(${target})
    add_test(NAME ${target} COMMAND ${target})
    cpp_llm_set_test_labels(${target} ${labels})
    cpp_llm_set_test_contract(${target} OWNER 07 LEVEL ${level} STATUS active)
endfunction()

add_chapter07_test(
    chapter07_block_pool_test
    tests/block_pool_test.cpp
    "cache"
    contract
)
add_chapter07_test(
    chapter07_paged_kv_cache_test
    tests/paged_kv_cache_test.cpp
    "cache;transaction"
    component
)
add_chapter07_test(
    chapter07_radix_prefix_cache_test
    tests/radix_prefix_cache_test.cpp
    "cache;prefix;radix"
    integration
)
