# Chapter 10：构建服务生命周期、HTTP/SSE 契约和跨章调度验证。
cmake_minimum_required(VERSION 3.20)
project(chapter10_serving LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()
include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

option(CHAPTER10_ENABLE_CPPHTTPLIB "Build the optional cpp-httplib transport" OFF)
set(CHAPTER10_CPPHTTPLIB_INCLUDE_DIR "" CACHE PATH
    "Directory containing httplib.h; skips FetchContent when set")

if(NOT TARGET chapter08_scheduling)
    add_subdirectory(
        ${CMAKE_CURRENT_LIST_DIR}/../chapter08_scheduling
        ${CMAKE_CURRENT_BINARY_DIR}/chapter08_scheduling
    )
endif()

add_library(chapter10_serving
    src/http_adapter.cpp
    src/request_worker.cpp
    src/service_core.cpp
)
target_include_directories(chapter10_serving PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(chapter10_serving
    PUBLIC
        chapter06_generation
        chapter07_kv_cache
        chapter08_scheduling
)
target_compile_features(chapter10_serving PUBLIC cxx_std_17)

function(add_chapter10_test target source labels level)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE chapter10_serving)
    cpp_llm_prepare_test(${target})
    add_test(NAME ${target} COMMAND ${target})
    cpp_llm_set_test_labels(${target} ${labels})
    cpp_llm_set_test_contract(
        ${target} OWNER 10 LEVEL ${level} STATUS active)
endfunction()

add_chapter10_test(
    chapter10_service_core_test
    tests/service_core_test.cpp
    "component;serving;lifecycle;scheduler"
    component
)
add_chapter10_test(
    chapter10_terminal_release_test
    tests/terminal_release_test.cpp
    "contract;serving;cache;cancellation"
    contract
)
add_chapter10_test(
    chapter10_http_contract_test
    tests/http_contract_test.cpp
    "integration;serving;http;streaming"
    integration
)
add_chapter10_test(
    chapter10_request_worker_test
    tests/request_worker_test.cpp
    "integration;serving;worker;concurrency"
    integration
)

if(CHAPTER10_ENABLE_CPPHTTPLIB)
    if(CHAPTER10_CPPHTTPLIB_INCLUDE_DIR)
        set(cpp_httplib_SOURCE_DIR "${CHAPTER10_CPPHTTPLIB_INCLUDE_DIR}")
    else()
        include(FetchContent)
        FetchContent_Declare(
            cpp_httplib
            GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
            GIT_TAG v0.18.3
            GIT_SHALLOW TRUE
        )
        FetchContent_Populate(cpp_httplib)
    endif()

    add_library(chapter10_cpp_httplib_transport src/cpp_httplib_transport.cpp)
    if(EXISTS "${cpp_httplib_SOURCE_DIR}/httplib.cpp")
        find_package(Threads REQUIRED)
        target_sources(chapter10_cpp_httplib_transport PRIVATE
            ${cpp_httplib_SOURCE_DIR}/httplib.cpp)
        target_link_libraries(chapter10_cpp_httplib_transport PRIVATE Threads::Threads)
    endif()
    target_include_directories(chapter10_cpp_httplib_transport
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${cpp_httplib_SOURCE_DIR}
    )
    target_link_libraries(chapter10_cpp_httplib_transport PUBLIC chapter10_serving)
    target_compile_features(chapter10_cpp_httplib_transport PUBLIC cxx_std_17)

    add_executable(chapter10_cpp_httplib_server examples/main_cpp_httplib.cpp)
    target_link_libraries(
        chapter10_cpp_httplib_server PRIVATE chapter10_cpp_httplib_transport)
endif()
