# Build the Chapter 08 evidence formatter and A/B comparison checks.
# Runtime execution remains in Chapter 06 and benchmark collection remains in Chapter 07.
cmake_minimum_required(VERSION 3.20)
project(chapter08_profiling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../../../testing/cmake/CppLlmTesting.cmake)

add_library(chapter08_profiling
    src/ab_compare.cpp
    src/profile_record.cpp
)

target_include_directories(chapter08_profiling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(chapter08_profiling PUBLIC cxx_std_17)

add_executable(chapter08_profile_report examples/main_profiling.cpp)
target_link_libraries(chapter08_profile_report PRIVATE chapter08_profiling)

add_executable(chapter08_profiling_test tests/chapter08_profiling_test.cpp)
target_link_libraries(chapter08_profiling_test PRIVATE chapter08_profiling)
cpp_llm_prepare_test(chapter08_profiling_test)

include(CTest)
enable_testing()
add_test(NAME chapter08_profiling_test COMMAND chapter08_profiling_test)
cpp_llm_set_test_labels(chapter08_profiling_test unit profiling)
