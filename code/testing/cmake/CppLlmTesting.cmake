# cpp-llm-engine 公共测试配置。
# 旧测试迁移出 assert 前，统一保证 Release 构建仍执行断言。
include_guard(GLOBAL)

set(CPP_LLM_TESTING_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

function(cpp_llm_prepare_test target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "cpp_llm_prepare_test: unknown target '${target}'")
    endif()
    target_include_directories(${target} PRIVATE "${CPP_LLM_TESTING_INCLUDE_DIR}")
    if(MSVC)
        target_compile_options(${target} PRIVATE /UNDEBUG)
    else()
        target_compile_options(${target} PRIVATE -UNDEBUG)
    endif()
endfunction()

function(cpp_llm_set_test_labels test_name)
    set_tests_properties(${test_name} PROPERTIES LABELS "${ARGN}")
endfunction()
