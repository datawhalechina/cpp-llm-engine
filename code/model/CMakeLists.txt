# 真实模型执行模块：连接 Chapter01 模型资产与 Chapter05 量化后端。
cmake_minimum_required(VERSION 3.20)
project(cpp_llm_engine_model LANGUAGES CXX)

if(NOT TARGET chapter01_tensor_loader)
    message(FATAL_ERROR "cpp_llm_engine_model requires chapter01_tensor_loader")
endif()

if(TARGET chapter05_quantization)
    add_library(cpp_llm_engine_model
        src/packed_expert_view.cpp
        src/real_deepseek2_cpu_model.cpp
        src/real_deepseek2_dense_layer.cpp
        src/real_moe_feed_forward.cpp
        src/real_qwen2_moe_layer.cpp
        src/real_qwen2_moe_cpu_model.cpp
        src/real_qwen2_moe_output.cpp
        src/real_llama_cpu_decoder.cpp
    )
    target_include_directories(cpp_llm_engine_model
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(cpp_llm_engine_model
        PUBLIC
            chapter01_tensor_loader
            chapter05_quantization
    )
    target_compile_features(cpp_llm_engine_model PUBLIC cxx_std_17)

    add_executable(packed_expert_view_test tests/packed_expert_view_test.cpp)
    target_link_libraries(packed_expert_view_test PRIVATE cpp_llm_engine_model)
    cpp_llm_prepare_test(packed_expert_view_test)
    add_test(NAME packed_expert_view_test COMMAND packed_expert_view_test)
    cpp_llm_set_test_labels(packed_expert_view_test model moe tensor)
    cpp_llm_set_test_contract(
        packed_expert_view_test OWNER 05 LEVEL component STATUS active)

    add_executable(real_moe_feed_forward_test
        tests/real_moe_feed_forward_test.cpp)
    target_link_libraries(real_moe_feed_forward_test PRIVATE cpp_llm_engine_model)
    cpp_llm_prepare_test(real_moe_feed_forward_test)
    add_test(NAME real_moe_feed_forward_test COMMAND real_moe_feed_forward_test)
    cpp_llm_set_test_labels(real_moe_feed_forward_test model moe quantization)
    cpp_llm_set_test_contract(
        real_moe_feed_forward_test OWNER 05 LEVEL component STATUS active)

    add_executable(real_qwen2_moe_layer_test
        tests/real_qwen2_moe_layer_test.cpp)
    target_link_libraries(real_qwen2_moe_layer_test PRIVATE cpp_llm_engine_model)
    cpp_llm_prepare_test(real_qwen2_moe_layer_test)
    add_test(NAME real_qwen2_moe_layer_test COMMAND real_qwen2_moe_layer_test)
    cpp_llm_set_test_labels(
        real_qwen2_moe_layer_test model moe attention prefill)
    cpp_llm_set_test_contract(
        real_qwen2_moe_layer_test OWNER 05 LEVEL component STATUS active)

    add_executable(real_deepseek2_dense_layer_test
        tests/real_deepseek2_dense_layer_test.cpp)
    target_link_libraries(real_deepseek2_dense_layer_test
        PRIVATE cpp_llm_engine_model)
    cpp_llm_prepare_test(real_deepseek2_dense_layer_test)
    add_test(NAME real_deepseek2_dense_layer_test
        COMMAND real_deepseek2_dense_layer_test)
    cpp_llm_set_test_labels(
        real_deepseek2_dense_layer_test model deepseek attention quantization)
    cpp_llm_set_test_contract(
        real_deepseek2_dense_layer_test OWNER 05 LEVEL integration STATUS active)

    add_executable(qwen15_moe_context_smoke
        examples/qwen15_moe_context_smoke.cpp)
    target_link_libraries(qwen15_moe_context_smoke
        PRIVATE cpp_llm_engine_model)
    set_target_properties(qwen15_moe_context_smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

    if(TARGET chapter05_quantization_avx2)
        target_compile_definitions(cpp_llm_engine_model
            PRIVATE
                CHAPTER01_HAS_Q4K_AVX2=1
                CHAPTER01_HAS_Q6K_AVX2=1
        )
        target_link_libraries(cpp_llm_engine_model
            PUBLIC chapter05_quantization_avx2
        )
    endif()
endif()

if(TARGET cpp_llm_engine_cuda)
    add_library(cpp_llm_engine_model_cuda
        src/cuda_llama_layer_weights.cpp
        src/real_llama_cuda_model.cpp
    )
    target_include_directories(cpp_llm_engine_model_cuda
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(cpp_llm_engine_model_cuda
        PUBLIC
            chapter01_tensor_loader
            cpp_llm_engine_cuda
    )
    target_compile_features(cpp_llm_engine_model_cuda PUBLIC cxx_std_17)
endif()

if(TARGET cpp_llm_engine_model)
    add_executable(gguf_decoder_multilayer
        examples/gguf_decoder_multilayer.cpp
    )
    add_executable(gguf_first_layer_forward
        examples/gguf_first_layer_forward.cpp
    )
    target_link_libraries(gguf_first_layer_forward
        PRIVATE
            chapter01_tensor_loader
            chapter05_quantization
    )
    add_executable(gguf_first_layer_multitoken
        examples/gguf_first_layer_multitoken.cpp
    )
    add_executable(qwen15_moe_ffn_inspect
        examples/qwen15_moe_ffn_inspect.cpp
    )
    add_executable(qwen15_moe_grouped_batch_benchmark
        examples/qwen15_moe_grouped_batch_benchmark.cpp
    )
    target_link_libraries(gguf_first_layer_multitoken
        PRIVATE
            chapter01_tensor_loader
            chapter05_quantization
    )
    set_target_properties(
        gguf_decoder_multilayer
        gguf_first_layer_forward
        gguf_first_layer_multitoken
        qwen15_moe_ffn_inspect
        qwen15_moe_grouped_batch_benchmark
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    target_link_libraries(gguf_decoder_multilayer
        PRIVATE cpp_llm_engine_model
    )
    target_link_libraries(qwen15_moe_ffn_inspect
        PRIVATE cpp_llm_engine_model
    )
    target_link_libraries(qwen15_moe_grouped_batch_benchmark
        PRIVATE cpp_llm_engine_model
    )
endif()

if(TARGET gguf_decoder_multilayer AND CHAPTER01_ENABLE_LLAMA_CPP)
    target_compile_definitions(gguf_decoder_multilayer
        PRIVATE CHAPTER01_COMPARE_LLAMA_CPP=1
    )
    target_include_directories(gguf_decoder_multilayer
        PRIVATE
            ${LLAMA_CPP_ROOT}/include
            ${LLAMA_CPP_ROOT}/ggml/include
    )
    target_link_libraries(gguf_decoder_multilayer
        PRIVATE
            ${LLAMA_LIBRARY}
            ${GGML_LIBRARY}
            ${GGML_CPU_LIBRARY}
            ${GGML_BASE_LIBRARY}
            ${CMAKE_DL_LIBS}
    )
    set_target_properties(gguf_decoder_multilayer PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )
endif()

if(TARGET cpp_llm_engine_model AND CHAPTER01_ENABLE_LLAMA_CPP)
    add_executable(qwen15_moe_ffn_compare
        examples/qwen15_moe_ffn_compare.cpp
    )
    target_include_directories(qwen15_moe_ffn_compare
        PRIVATE
            ${LLAMA_CPP_ROOT}/include
            ${LLAMA_CPP_ROOT}/ggml/include
    )
    target_link_libraries(qwen15_moe_ffn_compare
        PRIVATE
            cpp_llm_engine_model
            ${LLAMA_LIBRARY}
            ${GGML_LIBRARY}
            ${GGML_CPU_LIBRARY}
            ${GGML_BASE_LIBRARY}
            ${CMAKE_DL_LIBS}
    )
    set_target_properties(qwen15_moe_ffn_compare PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )

    add_executable(deepseek_v2_lite_layer_capture
        examples/deepseek_v2_lite_layer_capture.cpp
    )
    target_include_directories(deepseek_v2_lite_layer_capture
        PRIVATE
            ${LLAMA_CPP_ROOT}/include
            ${LLAMA_CPP_ROOT}/ggml/include
    )
    target_link_libraries(deepseek_v2_lite_layer_capture
        PRIVATE
            cpp_llm_engine_model
            ${LLAMA_LIBRARY}
            ${GGML_LIBRARY}
            ${GGML_CPU_LIBRARY}
            ${GGML_BASE_LIBRARY}
            ${CMAKE_DL_LIBS}
    )
    set_target_properties(deepseek_v2_lite_layer_capture PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )
endif()
