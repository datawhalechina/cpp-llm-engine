cmake_minimum_required(VERSION 3.20)

project(chapter00_quickstart LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()
include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

add_executable(chapter00_quickstart
    examples/main_quick.cpp
)

target_compile_features(chapter00_quickstart PRIVATE cxx_std_17)
cpp_llm_prepare_test(chapter00_quickstart)

add_test(NAME chapter00_quickstart_test COMMAND chapter00_quickstart --check)
cpp_llm_set_test_labels(chapter00_quickstart_test smoke quickstart)
cpp_llm_set_test_contract(
    chapter00_quickstart_test OWNER 00 LEVEL e2e STATUS active)

find_program(BASH_EXECUTABLE bash REQUIRED)
add_test(
    NAME chapter00_real_inference_script_test
    COMMAND ${BASH_EXECUTABLE}
            ${CMAKE_CURRENT_LIST_DIR}/tests/run_real_inference_contract.sh)
cpp_llm_set_test_labels(
    chapter00_real_inference_script_test quickstart script contract)
cpp_llm_set_test_contract(
    chapter00_real_inference_script_test OWNER 00 LEVEL contract STATUS active)
