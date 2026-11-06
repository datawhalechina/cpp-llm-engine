# Chapter 06：构建单请求 Generation Core、示例和专项测试。
# 目标：让 Request、TokenHistory、Sampler、Executor 和 Runtime 可独立验证。
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
    src/token_history.cpp
    src/runtime.cpp
    src/embedding.cpp
    src/position_encoding.cpp
    src/decoder_layer.cpp
    src/transformer.cpp
)

target_include_directories(chapter06_generation
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chapter06_generation PUBLIC cxx_std_17)

function(add_chapter06_test target source labels owner level status)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE chapter06_generation)
    cpp_llm_prepare_test(${target})
    add_test(NAME ${target} COMMAND ${target})
    cpp_llm_set_test_labels(${target} ${labels})
    cpp_llm_set_test_contract(
        ${target} OWNER ${owner} LEVEL ${level} STATUS ${status})
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
    "generation"
    06 component active
)
add_chapter06_test(
    chapter06_sampler_test
    tests/sampler_test.cpp
    "sampler"
    06 unit active
)
add_chapter06_test(
    chapter06_token_history_test
    tests/token_history_test.cpp
    "token-history"
    06 unit active
)
add_chapter06_test(
    chapter06_request_contract_test
    tests/request_contract_test.cpp
    "request"
    06 contract active
)

add_executable(chapter06_streaming_demo
    examples/main_streaming.cpp
)

target_link_libraries(chapter06_streaming_demo
    PRIVATE chapter06_generation
)
