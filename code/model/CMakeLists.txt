# 真实模型执行模块：连接 Chapter01 模型资产与 Chapter05 量化后端。
cmake_minimum_required(VERSION 3.20)
project(cpp_llm_engine_model LANGUAGES CXX)

if(NOT TARGET chapter01_tensor_loader)
    message(FATAL_ERROR "cpp_llm_engine_model requires chapter01_tensor_loader")
endif()

if(TARGET chapter05_quantization)
    add_library(cpp_llm_engine_model
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
    target_link_libraries(gguf_first_layer_multitoken
        PRIVATE
            chapter01_tensor_loader
            chapter05_quantization
    )
    set_target_properties(
        gguf_decoder_multilayer
        gguf_first_layer_forward
        gguf_first_layer_multitoken
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    target_link_libraries(gguf_decoder_multilayer
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
