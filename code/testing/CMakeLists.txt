# 测试基础设施自检：确认命名检查可运行，Release 构建仍启用断言。
cmake_minimum_required(VERSION 3.20)
project(cpp_llm_testing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/CppLlmTesting.cmake)
include(CTest)
enable_testing()

add_executable(cpp_llm_testing_self_test tests/testing_self_test.cpp)
cpp_llm_prepare_test(cpp_llm_testing_self_test)
add_test(NAME cpp_llm_testing_self_test COMMAND cpp_llm_testing_self_test)
cpp_llm_set_test_labels(cpp_llm_testing_self_test contract infrastructure)
