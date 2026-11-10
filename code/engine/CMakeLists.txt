cmake_minimum_required(VERSION 3.20)
project(cpp_llm_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)
include(CTest)
enable_testing()

option(CHAPTER01_ENABLE_LLAMA_CPP "Build the real llama.cpp tokenizer backend" OFF)
option(CPP_LLM_ENGINE_ENABLE_CUDA "Build optional CUDA model adapters" OFF)
set(LLAMA_CPP_ROOT "" CACHE PATH "llama.cpp checkout containing include/ and build/bin/")
set(CPP_LLM_ENGINE_REAL_MODEL "" CACHE FILEPATH "Optional GGUF used by real-model CTest")
set(CPP_LLM_ENGINE_CUDA_REAL_MODEL "" CACHE FILEPATH
    "Optional GGUF used by the real CUDA layer CTest")
set(CPP_LLM_ENGINE_QWEN15_MOE_MODEL "" CACHE FILEPATH
    "Optional Qwen1.5-MoE GGUF used by the real CPU Runtime CTest")
set(CHAPTER05_SOURCE_DIR "" CACHE PATH "Optional Chapter 05 source directory for real-model execution")

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CMAKE_CURRENT_BINARY_DIR}/core
)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../chapter01_tensor_loader
    ${CMAKE_CURRENT_BINARY_DIR}/chapter01_tensor_loader
)

if(CPP_LLM_ENGINE_ENABLE_CUDA)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
        ${CMAKE_CURRENT_BINARY_DIR}/cuda
    )
endif()

if(CHAPTER05_SOURCE_DIR)
    if(NOT EXISTS "${CHAPTER05_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "CHAPTER05_SOURCE_DIR must point to Chapter 05")
    endif()
    add_subdirectory(
        ${CHAPTER05_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/chapter05_quantization
    )
endif()

if(CPP_LLM_ENGINE_ENABLE_CUDA OR TARGET chapter05_quantization)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../model
        ${CMAKE_CURRENT_BINARY_DIR}/model
    )
endif()

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../chapter06_generation
    ${CMAKE_CURRENT_BINARY_DIR}/chapter06_generation
)

if(TARGET cpp_llm_engine_model)
    add_library(cpp_llm_engine_qwen2_moe_runtime
        src/real_qwen2_moe_executor_factory.cpp
    )
    target_include_directories(cpp_llm_engine_qwen2_moe_runtime
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(cpp_llm_engine_qwen2_moe_runtime
        PUBLIC cpp_llm_engine_model chapter06_generation
    )
    target_compile_features(cpp_llm_engine_qwen2_moe_runtime PUBLIC cxx_std_17)

    add_executable(real_qwen15_moe_runtime_test
        tests/real_qwen15_moe_runtime_test.cpp
    )
    add_executable(real_qwen15_moe_layer_cache_benchmark
        examples/real_qwen15_moe_layer_cache_benchmark.cpp
    )
    add_executable(real_qwen15_moe_batch_runtime_benchmark
        examples/real_qwen15_moe_batch_runtime_benchmark.cpp
    )
    add_executable(real_qwen15_moe_scheduling_benchmark
        examples/real_qwen15_moe_scheduling_benchmark.cpp
    )
    add_executable(real_qwen15_moe_service_benchmark
        examples/real_qwen15_moe_service_benchmark.cpp
    )
    target_link_libraries(real_qwen15_moe_runtime_test
        PRIVATE cpp_llm_engine_qwen2_moe_runtime)
    target_link_libraries(real_qwen15_moe_layer_cache_benchmark
        PRIVATE cpp_llm_engine_qwen2_moe_runtime)
    target_link_libraries(real_qwen15_moe_batch_runtime_benchmark
        PRIVATE cpp_llm_engine_qwen2_moe_runtime)
    target_link_libraries(real_qwen15_moe_scheduling_benchmark
        PRIVATE
            cpp_llm_engine_qwen2_moe_runtime
            chapter08_scheduling)
    target_link_libraries(real_qwen15_moe_service_benchmark
        PRIVATE
            cpp_llm_engine_qwen2_moe_runtime
            chapter10_serving)
    cpp_llm_prepare_test(real_qwen15_moe_runtime_test)
    if(CPP_LLM_ENGINE_QWEN15_MOE_MODEL)
        if(NOT EXISTS "${CPP_LLM_ENGINE_QWEN15_MOE_MODEL}")
            message(FATAL_ERROR "CPP_LLM_ENGINE_QWEN15_MOE_MODEL does not exist")
        endif()
        add_test(
            NAME real_qwen15_moe_runtime_test
            COMMAND real_qwen15_moe_runtime_test
                "${CPP_LLM_ENGINE_QWEN15_MOE_MODEL}"
        )
        cpp_llm_set_test_labels(
            real_qwen15_moe_runtime_test
            real_model slow cpu moe generation integration)
        cpp_llm_set_test_contract(
            real_qwen15_moe_runtime_test
            OWNER 06 LEVEL real_model STATUS active)
    endif()
endif()

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../chapter10_serving
    ${CMAKE_CURRENT_BINARY_DIR}/chapter10_serving
)

if(TARGET cpp_llm_engine_qwen2_moe_runtime AND TARGET chapter08_scheduling)
    add_executable(real_qwen15_moe_scheduling_smoke
        examples/real_qwen15_moe_scheduling_smoke.cpp
    )
    target_link_libraries(real_qwen15_moe_scheduling_smoke
        PRIVATE cpp_llm_engine_qwen2_moe_runtime chapter08_scheduling)
    cpp_llm_prepare_test(real_qwen15_moe_scheduling_smoke)
    if(CPP_LLM_ENGINE_QWEN15_MOE_MODEL)
        add_test(
            NAME real_qwen15_moe_scheduling_smoke
            COMMAND real_qwen15_moe_scheduling_smoke
                "${CPP_LLM_ENGINE_QWEN15_MOE_MODEL}"
        )
        cpp_llm_set_test_labels(
            real_qwen15_moe_scheduling_smoke
            real_model slow cpu moe scheduler batch integration)
        cpp_llm_set_test_contract(
            real_qwen15_moe_scheduling_smoke
            OWNER 08 LEVEL real_model STATUS active)
    endif()
endif()

add_library(cpp_llm_engine
    src/embedding_request_executor.cpp
    src/engine.cpp
)

if(CPP_LLM_ENGINE_ENABLE_CUDA)
    add_library(cpp_llm_engine_cuda_runtime
        src/real_llama_cuda_executor_factory.cpp
    )
    target_include_directories(cpp_llm_engine_cuda_runtime
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(cpp_llm_engine_cuda_runtime
        PUBLIC cpp_llm_engine_model_cuda chapter06_generation chapter07_kv_cache
    )
    target_compile_features(cpp_llm_engine_cuda_runtime PUBLIC cxx_std_17)
    add_executable(cuda_llama_layer_weights_test
        tests/cuda_llama_layer_weights_test.cpp
    )
    target_link_libraries(cuda_llama_layer_weights_test
        PRIVATE cpp_llm_engine_model_cuda)
    cpp_llm_prepare_test(cuda_llama_layer_weights_test)
    if(TARGET chapter05_quantization)
        add_executable(real_tinyllama_cuda_first_layer_test
            tests/real_tinyllama_cuda_first_layer_test.cpp
        )
        add_executable(real_tinyllama_cuda_multilayer_test
            tests/real_tinyllama_cuda_multilayer_test.cpp
        )
        add_executable(real_tinyllama_cuda_runtime_test
            tests/real_tinyllama_cuda_runtime_test.cpp
        )
        add_executable(real_tinyllama_cuda_attention_benchmark
            examples/real_tinyllama_cuda_attention_benchmark.cpp
        )
        add_executable(real_tinyllama_cuda_paged_prefill_benchmark
            examples/real_tinyllama_cuda_paged_prefill_benchmark.cpp
        )
        add_executable(real_tinyllama_cuda_batch_benchmark
            examples/real_tinyllama_cuda_batch_benchmark.cpp
        )
        add_executable(real_tinyllama_cuda_adaptive_gemv_matrix
            examples/real_tinyllama_cuda_adaptive_gemv_matrix.cpp
        )
        add_executable(real_tinyllama_cuda_paged_attention_benchmark
            examples/real_tinyllama_cuda_paged_attention_benchmark.cpp
        )
        target_link_libraries(real_tinyllama_cuda_first_layer_test
            PRIVATE cpp_llm_engine_model_cuda chapter05_quantization)
        target_link_libraries(real_tinyllama_cuda_multilayer_test
            PRIVATE cpp_llm_engine_model cpp_llm_engine_model_cuda)
        target_link_libraries(real_tinyllama_cuda_runtime_test
            PRIVATE cpp_llm_engine_cuda_runtime)
        target_link_libraries(real_tinyllama_cuda_attention_benchmark
            PRIVATE cpp_llm_engine_cuda_runtime)
        target_link_libraries(real_tinyllama_cuda_paged_prefill_benchmark
            PRIVATE cpp_llm_engine_cuda_runtime)
        target_link_libraries(real_tinyllama_cuda_batch_benchmark
            PRIVATE cpp_llm_engine_cuda_runtime)
        target_link_libraries(real_tinyllama_cuda_adaptive_gemv_matrix
            PRIVATE cpp_llm_engine_cuda_runtime)
        target_link_libraries(real_tinyllama_cuda_paged_attention_benchmark
            PRIVATE cpp_llm_engine_cuda_runtime)
        cpp_llm_prepare_test(real_tinyllama_cuda_first_layer_test)
        cpp_llm_prepare_test(real_tinyllama_cuda_multilayer_test)
        cpp_llm_prepare_test(real_tinyllama_cuda_runtime_test)
    endif()
endif()

target_include_directories(cpp_llm_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(cpp_llm_engine
    PUBLIC
        cpp_llm_engine_core
        chapter01_tensor_loader
        chapter06_generation
        chapter07_kv_cache
)

if(CHAPTER01_ENABLE_LLAMA_CPP)
    if(NOT TARGET chapter01_real_tokenizer)
        message(FATAL_ERROR "CHAPTER01_ENABLE_LLAMA_CPP requires chapter01_real_tokenizer")
    endif()
    add_library(cpp_llm_engine_real_tokenizer
        src/real_tokenizer_backend.cpp
    )
    target_include_directories(cpp_llm_engine_real_tokenizer
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(cpp_llm_engine_real_tokenizer
        PUBLIC chapter01_real_tokenizer
    )
    target_compile_features(cpp_llm_engine_real_tokenizer PUBLIC cxx_std_17)
    target_link_libraries(cpp_llm_engine PUBLIC cpp_llm_engine_real_tokenizer)
    add_executable(real_tokenizer_engine_smoke
        examples/real_tokenizer_engine_smoke.cpp
    )
    target_link_libraries(real_tokenizer_engine_smoke PRIVATE cpp_llm_engine)
    set_target_properties(real_tokenizer_engine_smoke PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )
    if(TARGET cpp_llm_engine_qwen2_moe_runtime)
        add_executable(real_qwen15_moe_generation_smoke
            examples/real_qwen15_moe_generation_smoke.cpp
        )
        target_link_libraries(real_qwen15_moe_generation_smoke
            PRIVATE
                cpp_llm_engine_qwen2_moe_runtime
                cpp_llm_engine_real_tokenizer
        )
        set_target_properties(real_qwen15_moe_generation_smoke PROPERTIES
            BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
        )
    endif()
    if(TARGET cpp_llm_engine_cuda_runtime)
        target_sources(cpp_llm_engine_cuda_runtime PRIVATE
            src/real_llama_cuda_engine.cpp
        )
        target_link_libraries(cpp_llm_engine_cuda_runtime
            PUBLIC cpp_llm_engine_real_tokenizer
        )
        add_executable(real_tinyllama_cuda_generation_smoke
            examples/real_tinyllama_cuda_generation_smoke.cpp
        )
        add_executable(real_tinyllama_cuda_serving_smoke
            examples/real_tinyllama_cuda_serving_smoke.cpp
        )
        target_link_libraries(real_tinyllama_cuda_generation_smoke
            PRIVATE cpp_llm_engine_cuda_runtime
        )
        target_link_libraries(real_tinyllama_cuda_serving_smoke
            PRIVATE cpp_llm_engine_cuda_runtime chapter10_serving
        )
        cpp_llm_prepare_test(real_tinyllama_cuda_generation_smoke)
        cpp_llm_prepare_test(real_tinyllama_cuda_serving_smoke)
        set_target_properties(real_tinyllama_cuda_generation_smoke PROPERTIES
            BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
        )
        set_target_properties(real_tinyllama_cuda_serving_smoke PROPERTIES
            BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
        )
    endif()
    if(TARGET cpp_llm_engine_model)
        target_sources(cpp_llm_engine PRIVATE
            src/real_llama_engine.cpp
            src/real_llama_executor_factory.cpp
        )
        target_link_libraries(cpp_llm_engine PUBLIC cpp_llm_engine_model)
        add_executable(real_tinyllama_generation_smoke
            examples/real_tinyllama_generation_smoke.cpp
        )
        target_link_libraries(real_tinyllama_generation_smoke PRIVATE cpp_llm_engine)
        set_target_properties(real_tinyllama_generation_smoke PROPERTIES
            BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
        )
    endif()
endif()

add_executable(cpp_llm_engine_demo
    examples/main.cpp
)

target_link_libraries(cpp_llm_engine_demo
    PRIVATE
        cpp_llm_engine
)

add_executable(cpp_llm_engine_test
    tests/engine_smoke_test.cpp
)

add_executable(embedding_request_executor_test
    tests/embedding_request_executor_test.cpp
)

add_executable(embedding_request_service_test
    tests/embedding_request_service_test.cpp
)

target_link_libraries(embedding_request_executor_test
    PRIVATE cpp_llm_engine
)
cpp_llm_prepare_test(embedding_request_executor_test)

target_link_libraries(embedding_request_service_test
    PRIVATE cpp_llm_engine chapter10_serving
)
cpp_llm_prepare_test(embedding_request_service_test)

target_link_libraries(cpp_llm_engine_test
    PRIVATE
        cpp_llm_engine
)
cpp_llm_prepare_test(cpp_llm_engine_test)

add_test(NAME cpp_llm_engine_test COMMAND cpp_llm_engine_test)
add_test(NAME embedding_request_executor_test COMMAND embedding_request_executor_test)
add_test(NAME embedding_request_service_test COMMAND embedding_request_service_test)
cpp_llm_set_test_labels(cpp_llm_engine_test integration engine)
cpp_llm_set_test_labels(embedding_request_executor_test component engine executor)
cpp_llm_set_test_labels(embedding_request_service_test integration engine serving)
if(CPP_LLM_ENGINE_ENABLE_CUDA)
    add_test(NAME cuda_llama_layer_weights_test COMMAND cuda_llama_layer_weights_test)
    cpp_llm_set_test_labels(cuda_llama_layer_weights_test cuda integration)
    if(CPP_LLM_ENGINE_CUDA_REAL_MODEL)
        if(NOT EXISTS "${CPP_LLM_ENGINE_CUDA_REAL_MODEL}")
            message(FATAL_ERROR "CPP_LLM_ENGINE_CUDA_REAL_MODEL does not exist")
        endif()
        if(NOT TARGET real_tinyllama_cuda_first_layer_test)
            message(FATAL_ERROR
                "CPP_LLM_ENGINE_CUDA_REAL_MODEL requires CHAPTER05_SOURCE_DIR")
        endif()
        add_test(
            NAME real_tinyllama_cuda_first_layer_test
            COMMAND real_tinyllama_cuda_first_layer_test
                "${CPP_LLM_ENGINE_CUDA_REAL_MODEL}"
        )
        add_test(
            NAME real_tinyllama_cuda_multilayer_test
            COMMAND real_tinyllama_cuda_multilayer_test
                "${CPP_LLM_ENGINE_CUDA_REAL_MODEL}" 2
        )
        add_test(
            NAME real_tinyllama_cuda_full_decoder_test
            COMMAND real_tinyllama_cuda_multilayer_test
                "${CPP_LLM_ENGINE_CUDA_REAL_MODEL}" 22
        )
        add_test(
            NAME real_tinyllama_cuda_runtime_test
            COMMAND real_tinyllama_cuda_runtime_test
                "${CPP_LLM_ENGINE_CUDA_REAL_MODEL}"
        )
        cpp_llm_set_test_labels(
            real_tinyllama_cuda_first_layer_test
            real_model slow cuda decoder integration)
        cpp_llm_set_test_contract(
            real_tinyllama_cuda_first_layer_test
            OWNER 04 LEVEL real_model STATUS active)
        cpp_llm_set_test_labels(
            real_tinyllama_cuda_multilayer_test
            real_model slow cuda decoder integration)
        cpp_llm_set_test_contract(
            real_tinyllama_cuda_multilayer_test
            OWNER 04 LEVEL real_model STATUS active)
        cpp_llm_set_test_labels(
            real_tinyllama_cuda_full_decoder_test
            real_model slow cuda decoder integration)
        cpp_llm_set_test_contract(
            real_tinyllama_cuda_full_decoder_test
            OWNER 04 LEVEL real_model STATUS active)
        cpp_llm_set_test_labels(
            real_tinyllama_cuda_runtime_test
            real_model slow cuda generation integration)
        cpp_llm_set_test_contract(
            real_tinyllama_cuda_runtime_test
            OWNER 06 LEVEL real_model STATUS active)
        if(TARGET real_tinyllama_cuda_generation_smoke)
            add_test(
                NAME real_tinyllama_cuda_generation_smoke
                COMMAND real_tinyllama_cuda_generation_smoke
                    "${CPP_LLM_ENGINE_CUDA_REAL_MODEL}"
                    4
                    "Hello TinyLlama"
                    "29892,306,29915,29885"
            )
            cpp_llm_set_test_labels(
                real_tinyllama_cuda_generation_smoke
                real_model slow cuda generation integration tokenizer)
            cpp_llm_set_test_contract(
                real_tinyllama_cuda_generation_smoke
                OWNER 06 LEVEL real_model STATUS active)
            add_test(
                NAME real_tinyllama_cuda_serving_smoke
                COMMAND real_tinyllama_cuda_serving_smoke
                    "${CPP_LLM_ENGINE_CUDA_REAL_MODEL}"
            )
            cpp_llm_set_test_labels(
                real_tinyllama_cuda_serving_smoke
                real_model slow cuda serving http streaming integration)
            cpp_llm_set_test_contract(
                real_tinyllama_cuda_serving_smoke
                OWNER 10 LEVEL real_model STATUS active)
        endif()
    endif()
endif()

if(CPP_LLM_ENGINE_REAL_MODEL)
    if(NOT EXISTS "${CPP_LLM_ENGINE_REAL_MODEL}")
        message(FATAL_ERROR "CPP_LLM_ENGINE_REAL_MODEL does not exist")
    endif()
    if(NOT TARGET real_tinyllama_generation_smoke)
        message(FATAL_ERROR
            "CPP_LLM_ENGINE_REAL_MODEL requires CHAPTER01_ENABLE_LLAMA_CPP and CHAPTER05_SOURCE_DIR"
        )
    endif()
    add_test(
        NAME real_tinyllama_generation_test
        COMMAND real_tinyllama_generation_smoke
            "${CPP_LLM_ENGINE_REAL_MODEL}"
            4
            "Hello TinyLlama"
            "29892,306,29915,29885"
    )
    set_tests_properties(real_tinyllama_generation_test PROPERTIES
        LABELS "real_model;slow"
        PASS_REGULAR_EXPRESSION "golden_tokens_match: true"
    )
    cpp_llm_set_test_contract(
        real_tinyllama_generation_test
        OWNER 06 LEVEL real_model STATUS active)

    add_test(
        NAME real_tinyllama_first_layer_structure_test
        COMMAND gguf_first_layer_inspect "${CPP_LLM_ENGINE_REAL_MODEL}"
    )
    set_tests_properties(real_tinyllama_first_layer_structure_test PROPERTIES
        LABELS "real_model;slow;decoder;structure"
        PASS_REGULAR_EXPRESSION "first_layer_weight_contract=ready"
    )
    cpp_llm_set_test_contract(
        real_tinyllama_first_layer_structure_test
        OWNER 04 LEVEL real_model STATUS active)

    add_test(
        NAME gguf_decoder_multilayer
        COMMAND gguf_decoder_multilayer "${CPP_LLM_ENGINE_REAL_MODEL}" 22
    )
    cpp_llm_set_test_labels(
        gguf_decoder_multilayer real_model slow decoder quantization)
    cpp_llm_set_test_contract(
        gguf_decoder_multilayer OWNER 05 LEVEL real_model STATUS active)

    add_executable(real_tinyllama_paged_kv_smoke
        examples/real_tinyllama_paged_kv_smoke.cpp
    )
    target_link_libraries(real_tinyllama_paged_kv_smoke PRIVATE cpp_llm_engine)
    set_target_properties(real_tinyllama_paged_kv_smoke PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )
    add_test(
        NAME real_tinyllama_paged_kv_test
        COMMAND real_tinyllama_paged_kv_smoke "${CPP_LLM_ENGINE_REAL_MODEL}"
    )
    set_tests_properties(real_tinyllama_paged_kv_test PROPERTIES
        LABELS "real_model;slow;paged_kv"
        PASS_REGULAR_EXPRESSION "paged_kv_real_match: true"
    )
    cpp_llm_set_test_contract(
        real_tinyllama_paged_kv_test
        OWNER 07 LEVEL real_model STATUS active)

    add_executable(real_tinyllama_batch_smoke
        examples/real_tinyllama_batch_smoke.cpp
    )
    target_link_libraries(real_tinyllama_batch_smoke PRIVATE cpp_llm_engine)
    set_target_properties(real_tinyllama_batch_smoke PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )
    add_test(
        NAME real_tinyllama_batch_test
        COMMAND real_tinyllama_batch_smoke "${CPP_LLM_ENGINE_REAL_MODEL}"
    )
    set_tests_properties(real_tinyllama_batch_test PROPERTIES
        LABELS "real_model;slow;batch"
        PASS_REGULAR_EXPRESSION "batch_golden_match: true"
    )
    cpp_llm_set_test_contract(
        real_tinyllama_batch_test OWNER 08 LEVEL real_model STATUS active)

    add_executable(real_tinyllama_batch_benchmark
        examples/real_tinyllama_batch_benchmark.cpp
    )
    target_link_libraries(real_tinyllama_batch_benchmark PRIVATE cpp_llm_engine)
    target_compile_definitions(real_tinyllama_batch_benchmark PRIVATE
        CPP_LLM_ENGINE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    )
    set_target_properties(real_tinyllama_batch_benchmark PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )
    add_test(
        NAME real_tinyllama_batch_benchmark_test
        COMMAND real_tinyllama_batch_benchmark
            "${CPP_LLM_ENGINE_REAL_MODEL}"
            --warmup 0
            --iterations 1
            --output "${CMAKE_CURRENT_BINARY_DIR}/tinyllama_real_batch_test.json"
    )
    set_tests_properties(real_tinyllama_batch_benchmark_test PROPERTIES
        LABELS "real_model;slow;batch;benchmark"
        PASS_REGULAR_EXPRESSION "correctness=PASS"
    )

    add_executable(real_tinyllama_serving_smoke
        examples/real_tinyllama_serving_smoke.cpp
    )
    target_link_libraries(real_tinyllama_serving_smoke
        PRIVATE cpp_llm_engine chapter10_serving
    )
    set_target_properties(real_tinyllama_serving_smoke PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )
    add_test(
        NAME real_tinyllama_serving_test
        COMMAND real_tinyllama_serving_smoke "${CPP_LLM_ENGINE_REAL_MODEL}"
    )
    set_tests_properties(real_tinyllama_serving_test PROPERTIES
        LABELS "real_model;slow;serving"
        PASS_REGULAR_EXPRESSION "real_serving_match: true"
    )
    cpp_llm_set_test_contract(
        real_tinyllama_serving_test
        OWNER 10 LEVEL real_model STATUS active)

    add_executable(real_tinyllama_http_adapter_smoke
        examples/real_tinyllama_http_adapter_smoke.cpp
    )
    if(NOT TARGET chapter10_serving)
        add_subdirectory(
            ${CMAKE_CURRENT_SOURCE_DIR}/../chapter10_serving
            ${CMAKE_CURRENT_BINARY_DIR}/chapter10_serving
        )
    endif()
    target_link_libraries(real_tinyllama_http_adapter_smoke
        PRIVATE cpp_llm_engine chapter10_serving
    )
    set_target_properties(real_tinyllama_http_adapter_smoke PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )
    add_test(
        NAME real_tinyllama_http_adapter_test
        COMMAND real_tinyllama_http_adapter_smoke "${CPP_LLM_ENGINE_REAL_MODEL}"
    )
    set_tests_properties(real_tinyllama_http_adapter_test PROPERTIES
        LABELS "real_model;slow;serving;http"
        PASS_REGULAR_EXPRESSION "sync_golden_match: true"
    )
    cpp_llm_set_test_contract(
        real_tinyllama_http_adapter_test
        OWNER 10 LEVEL real_model STATUS active)

    if(TARGET chapter10_cpp_httplib_transport)
        add_executable(real_tinyllama_http_server
            examples/real_tinyllama_http_server.cpp
        )
        target_link_libraries(real_tinyllama_http_server
            PRIVATE cpp_llm_engine chapter10_cpp_httplib_transport
        )
        set_target_properties(real_tinyllama_http_server PROPERTIES
            BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
        )
    endif()

    add_executable(real_tinyllama_decoder_profile
        examples/real_tinyllama_decoder_profile.cpp
    )
    target_link_libraries(real_tinyllama_decoder_profile PRIVATE cpp_llm_engine)
    target_compile_definitions(real_tinyllama_decoder_profile PRIVATE
        CPP_LLM_ENGINE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    )
    set_target_properties(real_tinyllama_decoder_profile PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )
    add_test(
        NAME real_tinyllama_decoder_profile_test
        COMMAND real_tinyllama_decoder_profile "${CPP_LLM_ENGINE_REAL_MODEL}" 1
    )
    set_tests_properties(real_tinyllama_decoder_profile_test PROPERTIES
        LABELS "real_model;slow;profiling"
        PASS_REGULAR_EXPRESSION "workload_id=WL-CPU-TINYLLAMA-DECODER-PHASE-V1"
    )

    add_executable(real_tinyllama_serving_benchmark
        examples/real_tinyllama_serving_benchmark.cpp
    )
    target_link_libraries(real_tinyllama_serving_benchmark
        PRIVATE cpp_llm_engine chapter10_serving
    )
    target_compile_definitions(real_tinyllama_serving_benchmark PRIVATE
        CPP_LLM_ENGINE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    )
    set_target_properties(real_tinyllama_serving_benchmark PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )
    add_test(
        NAME real_tinyllama_serving_benchmark_test
        COMMAND real_tinyllama_serving_benchmark
            "${CPP_LLM_ENGINE_REAL_MODEL}"
            --warmup 0
            --iterations 1
            --output "${CMAKE_CURRENT_BINARY_DIR}/tinyllama_service_benchmark_test.json"
    )
    set_tests_properties(real_tinyllama_serving_benchmark_test PROPERTIES
        LABELS "real_model;slow;serving;benchmark"
        PASS_REGULAR_EXPRESSION "correctness=PASS"
    )
endif()
