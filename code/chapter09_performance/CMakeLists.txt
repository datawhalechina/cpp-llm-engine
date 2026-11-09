# Chapter 09：构建统一性能证据、统计聚合与 A/B 决策测试。
cmake_minimum_required(VERSION 3.20)
project(chapter09_performance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()
include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

add_library(chapter09_performance
    src/comparison.cpp
    src/evidence_record.cpp
    src/statistics.cpp
)
target_include_directories(chapter09_performance PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(chapter09_performance PUBLIC cxx_std_17)

add_executable(chapter09_compare_evidence
    examples/compare_evidence.cpp
)
target_link_libraries(chapter09_compare_evidence PRIVATE chapter09_performance)

function(add_chapter09_test target source labels level)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE chapter09_performance)
    cpp_llm_prepare_test(${target})
    add_test(NAME ${target} COMMAND ${target})
    cpp_llm_set_test_labels(${target} ${labels})
    cpp_llm_set_test_contract(
        ${target} OWNER 09 LEVEL ${level} STATUS active)
endfunction()

add_chapter09_test(
    chapter09_statistics_test
    tests/statistics_test.cpp
    "unit;performance;statistics"
    unit
)

find_program(CHAPTER09_NODE_EXECUTABLE node)
if(CHAPTER09_NODE_EXECUTABLE)
    add_test(
        NAME chapter09_legacy_import_test
        COMMAND ${CHAPTER09_NODE_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/legacy_evidence_import_test.mjs
    )
    cpp_llm_set_test_labels(chapter09_legacy_import_test
        "contract;performance;migration")
    cpp_llm_set_test_contract(chapter09_legacy_import_test
        OWNER 09 LEVEL contract STATUS active)

    add_test(
        NAME chapter09_versioned_evidence_test
        COMMAND ${CHAPTER09_NODE_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/versioned_evidence_test.mjs
    )
    cpp_llm_set_test_labels(chapter09_versioned_evidence_test
        "contract;performance;evidence;versioned")
    cpp_llm_set_test_contract(chapter09_versioned_evidence_test
        OWNER 09 LEVEL contract STATUS active)

    add_test(
        NAME chapter09_real_evidence_bridge_test
        COMMAND ${CHAPTER09_NODE_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/real_evidence_bridge_test.mjs
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_evidence_pair.mjs
                $<TARGET_FILE:chapter09_compare_evidence>
    )
    cpp_llm_set_test_labels(chapter09_real_evidence_bridge_test
        "integration;performance;comparison;evidence")
    cpp_llm_set_test_contract(chapter09_real_evidence_bridge_test
        OWNER 09 LEVEL integration STATUS active)

    add_test(
        NAME chapter09_adaptive_gemv_evidence_test
        COMMAND ${CHAPTER09_NODE_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/adaptive_gemv_evidence_test.mjs
    )
    cpp_llm_set_test_labels(chapter09_adaptive_gemv_evidence_test
        "contract;performance;evidence;cuda")
    cpp_llm_set_test_contract(chapter09_adaptive_gemv_evidence_test
        OWNER 09 LEVEL contract STATUS active)
endif()
add_chapter09_test(
    chapter09_evidence_contract_test
    tests/evidence_contract_test.cpp
    "contract;performance;evidence"
    contract
)
add_chapter09_test(
    chapter09_comparison_test
    tests/comparison_test.cpp
    "unit;performance;comparison"
    unit
)
