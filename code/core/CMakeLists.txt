cmake_minimum_required(VERSION 3.20)
project(cpp_llm_engine_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

add_library(cpp_llm_engine_core INTERFACE)
target_include_directories(cpp_llm_engine_core
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(cpp_llm_engine_core_test
    tests/core_contract_test.cpp
)
target_link_libraries(cpp_llm_engine_core_test PRIVATE cpp_llm_engine_core)
cpp_llm_prepare_test(cpp_llm_engine_core_test)

include(CTest)
enable_testing()
add_test(NAME cpp_llm_engine_core_test COMMAND cpp_llm_engine_core_test)
cpp_llm_set_test_labels(cpp_llm_engine_core_test unit contract core)
