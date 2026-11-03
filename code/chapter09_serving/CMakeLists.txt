cmake_minimum_required(VERSION 3.20)
project(chapter09_serving LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

option(CHAPTER09_ENABLE_CPPHTTPLIB "Build the optional cpp-httplib transport" OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CHAPTER09_STANDALONE_DEFAULT ON)
else()
    set(CHAPTER09_STANDALONE_DEFAULT OFF)
endif()
option(CHAPTER09_BUILD_STANDALONE_ASSETS
    "Build Chapter09 demo and unit tests"
    ${CHAPTER09_STANDALONE_DEFAULT})
set(CHAPTER09_CPPHTTPLIB_INCLUDE_DIR "" CACHE PATH
    "Directory containing httplib.h; skips FetchContent when set")

if(NOT TARGET chapter06_generation)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../chapter06_generation
        ${CMAKE_CURRENT_BINARY_DIR}/chapter06_generation
    )
endif()

add_library(chapter09_serving
    src/serving_core.cpp
    src/http_adapter.cpp
)

target_include_directories(chapter09_serving
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(chapter09_serving PUBLIC chapter06_generation)
target_compile_features(chapter09_serving PUBLIC cxx_std_17)

if(CHAPTER09_BUILD_STANDALONE_ASSETS)
    add_executable(chapter09_serving_demo examples/main_serving.cpp)
    target_link_libraries(chapter09_serving_demo PRIVATE chapter09_serving)

    add_executable(chapter09_serving_test tests/chapter09_serving_test.cpp)
    target_link_libraries(chapter09_serving_test PRIVATE chapter09_serving)
    cpp_llm_prepare_test(chapter09_serving_test)

    add_executable(chapter09_step_dispatch_test tests/step_dispatch_test.cpp)
    target_link_libraries(chapter09_step_dispatch_test PRIVATE chapter09_serving)
    cpp_llm_prepare_test(chapter09_step_dispatch_test)

    add_executable(chapter09_parallel_dispatch_test tests/parallel_dispatch_test.cpp)
    target_link_libraries(chapter09_parallel_dispatch_test PRIVATE chapter09_serving)
    cpp_llm_prepare_test(chapter09_parallel_dispatch_test)

    add_executable(
        chapter09_cancellation_release_test tests/cancellation_release_test.cpp)
    target_link_libraries(chapter09_cancellation_release_test PRIVATE chapter09_serving)
    cpp_llm_prepare_test(chapter09_cancellation_release_test)

    add_executable(chapter09_service_metrics_test tests/service_metrics_test.cpp)
    target_link_libraries(chapter09_service_metrics_test PRIVATE chapter09_serving)
    cpp_llm_prepare_test(chapter09_service_metrics_test)

    add_executable(chapter09_fault_sequence_test tests/fault_sequence_test.cpp)
    target_link_libraries(chapter09_fault_sequence_test PRIVATE chapter09_serving)
    cpp_llm_prepare_test(chapter09_fault_sequence_test)

    add_executable(chapter09_http_adapter_test tests/http_adapter_test.cpp)
    target_link_libraries(chapter09_http_adapter_test PRIVATE chapter09_serving)
    cpp_llm_prepare_test(chapter09_http_adapter_test)

    add_executable(chapter09_response_writer_test tests/response_writer_test.cpp)
    target_link_libraries(chapter09_response_writer_test PRIVATE chapter09_serving)
    cpp_llm_prepare_test(chapter09_response_writer_test)

    add_executable(chapter09_http_adapter_writer_test tests/http_adapter_writer_test.cpp)
    target_link_libraries(chapter09_http_adapter_writer_test PRIVATE chapter09_serving)
    cpp_llm_prepare_test(chapter09_http_adapter_writer_test)

    include(CTest)
    enable_testing()
    add_test(NAME chapter09_serving_test COMMAND chapter09_serving_test)
    add_test(NAME chapter09_step_dispatch_test COMMAND chapter09_step_dispatch_test)
    add_test(NAME chapter09_parallel_dispatch_test COMMAND chapter09_parallel_dispatch_test)
    add_test(
        NAME chapter09_cancellation_release_test
        COMMAND chapter09_cancellation_release_test)
    add_test(NAME chapter09_service_metrics_test COMMAND chapter09_service_metrics_test)
    add_test(NAME chapter09_fault_sequence_test COMMAND chapter09_fault_sequence_test)
    add_test(NAME chapter09_http_adapter_test COMMAND chapter09_http_adapter_test)
    add_test(NAME chapter09_response_writer_test COMMAND chapter09_response_writer_test)
    add_test(NAME chapter09_http_adapter_writer_test COMMAND chapter09_http_adapter_writer_test)
    cpp_llm_set_test_labels(chapter09_serving_test component serving lifecycle)
    cpp_llm_set_test_labels(
        chapter09_step_dispatch_test component serving scheduler cache)
    cpp_llm_set_test_labels(
        chapter09_parallel_dispatch_test integration serving parallel)
    cpp_llm_set_test_labels(
        chapter09_cancellation_release_test contract serving cancellation cache)
    cpp_llm_set_test_labels(
        chapter09_service_metrics_test contract serving metrics latency)
    cpp_llm_set_test_labels(
        chapter09_fault_sequence_test contract serving fault sequence cache)
    cpp_llm_set_test_labels(chapter09_http_adapter_test unit http)
    cpp_llm_set_test_labels(chapter09_response_writer_test unit streaming)
    cpp_llm_set_test_labels(chapter09_http_adapter_writer_test integration http streaming)
endif()

if(CHAPTER09_ENABLE_CPPHTTPLIB)
    if(CHAPTER09_CPPHTTPLIB_INCLUDE_DIR)
        set(cpp_httplib_SOURCE_DIR "${CHAPTER09_CPPHTTPLIB_INCLUDE_DIR}")
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

    add_library(chapter09_cpp_httplib_transport
        src/cpp_httplib_transport.cpp
    )
    if(EXISTS "${cpp_httplib_SOURCE_DIR}/httplib.cpp")
        find_package(Threads REQUIRED)
        target_sources(chapter09_cpp_httplib_transport PRIVATE
            ${cpp_httplib_SOURCE_DIR}/httplib.cpp
        )
        target_link_libraries(chapter09_cpp_httplib_transport PRIVATE Threads::Threads)
    endif()
    target_include_directories(chapter09_cpp_httplib_transport
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${cpp_httplib_SOURCE_DIR}
    )
    target_link_libraries(chapter09_cpp_httplib_transport PUBLIC chapter09_serving)
    target_compile_features(chapter09_cpp_httplib_transport PUBLIC cxx_std_17)

    add_executable(chapter09_cpp_httplib_server
        examples/main_cpp_httplib.cpp
    )
    target_link_libraries(
        chapter09_cpp_httplib_server PRIVATE chapter09_cpp_httplib_transport)
    target_compile_features(chapter09_cpp_httplib_server PRIVATE cxx_std_17)
endif()
