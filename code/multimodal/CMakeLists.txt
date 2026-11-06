cmake_minimum_required(VERSION 3.20)
project(cpp_llm_multimodal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

option(MULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER "Enable the optional real tokenizer path" OFF)
option(MULTIMODAL_ENABLE_CBLAS "Enable the optional CBLAS vision compute backend" OFF)
option(MULTIMODAL_ENABLE_SERVICE_INTEGRATION
    "Enable the optional Chapter10 service smoke in the real projector inspector" OFF)
set(MULTIMODAL_CBLAS_LIBRARY "" CACHE FILEPATH "Optional CBLAS library override")
set(LLAMA_CPP_ROOT "" CACHE PATH "llama.cpp source checkout")
set(LLAMA_CPP_BUILD_DIR "" CACHE PATH "llama.cpp build directory containing bin/")

add_library(cpp_llm_multimodal
    src/input_embedding.cpp
    src/idefics3_preprocessor.cpp
    src/image_decoder.cpp
    src/image_preprocessor.cpp
    src/mmproj_schema.cpp
    src/request_metrics.cpp
    src/smolvlm_prompt.cpp
    src/vision_encoder.cpp
    src/vision_projector.cpp
    src/vision_weights.cpp
    ../chapter01_tensor_loader/src/gguf_reader.cpp
    ../chapter01_tensor_loader/src/gguf_tensor_loader.cpp
    ../chapter01_tensor_loader/src/model_metadata.cpp
    ../chapter01_tensor_loader/src/model_metadata_adapter.cpp
    ../chapter01_tensor_loader/src/model_schema.cpp
    ../chapter01_tensor_loader/src/real_layer_weights.cpp
    ../model/src/real_llama_cpu_decoder.cpp
    ../chapter01_tensor_loader/src/tensor.cpp
    ../chapter05_quantization/src/q4k_dequantize.cpp
    ../chapter05_quantization/src/q4k_kernel.cpp
    ../chapter05_quantization/src/q4k_linear.cpp
    ../chapter05_quantization/src/q6k_dequantize.cpp
    ../chapter05_quantization/src/q6k_kernel.cpp
    ../chapter05_quantization/src/q6k_linear.cpp
    ../chapter05_quantization/src/q8_0_dequantize.cpp
    ../chapter05_quantization/src/q8_0_linear.cpp
    ../chapter05_quantization/src/quantized_linear_policy.cpp
)
target_include_directories(cpp_llm_multimodal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../chapter01_tensor_loader/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../chapter05_quantization/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/include
)

if(MULTIMODAL_ENABLE_CBLAS)
    if(MULTIMODAL_CBLAS_LIBRARY)
        if(NOT EXISTS "${MULTIMODAL_CBLAS_LIBRARY}")
            message(FATAL_ERROR "MULTIMODAL_CBLAS_LIBRARY does not exist")
        endif()
        target_link_libraries(cpp_llm_multimodal PUBLIC "${MULTIMODAL_CBLAS_LIBRARY}")
    else()
        find_package(BLAS REQUIRED)
        target_link_libraries(cpp_llm_multimodal PUBLIC BLAS::BLAS)
    endif()
    target_compile_definitions(cpp_llm_multimodal PRIVATE CPP_LLM_MULTIMODAL_HAS_CBLAS=1)
endif()

find_package(JPEG QUIET)
if(JPEG_FOUND)
    target_sources(cpp_llm_multimodal PRIVATE src/jpeg_decoder.cpp)
    target_link_libraries(cpp_llm_multimodal PRIVATE JPEG::JPEG)
    target_compile_definitions(cpp_llm_multimodal PRIVATE CPP_LLM_HAS_LIBJPEG=1)
endif()

add_executable(multimodal_image_inspect examples/image_inspect.cpp)
target_link_libraries(multimodal_image_inspect PRIVATE cpp_llm_multimodal)

add_executable(multimodal_mmproj_inspect examples/mmproj_inspect.cpp)
target_link_libraries(multimodal_mmproj_inspect PRIVATE cpp_llm_multimodal)

add_executable(multimodal_vision_embedding_inspect examples/vision_embedding_inspect.cpp)
target_link_libraries(multimodal_vision_embedding_inspect PRIVATE cpp_llm_multimodal)

add_executable(multimodal_vision_block_inspect examples/vision_block_inspect.cpp)
target_link_libraries(multimodal_vision_block_inspect PRIVATE cpp_llm_multimodal)

add_executable(multimodal_vision_encoder_inspect examples/vision_encoder_inspect.cpp)
target_link_libraries(multimodal_vision_encoder_inspect PRIVATE cpp_llm_multimodal)

add_executable(multimodal_vision_projector_inspect examples/vision_projector_inspect.cpp)
target_link_libraries(multimodal_vision_projector_inspect PRIVATE cpp_llm_multimodal)

if(MULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER)
    if(NOT EXISTS "${LLAMA_CPP_ROOT}/include/llama.h" OR
       NOT EXISTS "${LLAMA_CPP_BUILD_DIR}/bin")
        message(FATAL_ERROR "LLAMA_CPP_ROOT and LLAMA_CPP_BUILD_DIR are required for the real tokenizer")
    endif()
    find_library(LLAMA_LIBRARY NAMES llama PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_LIBRARY NAMES ggml PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_BASE_LIBRARY NAMES ggml-base PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_CPU_LIBRARY NAMES ggml-cpu PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    target_sources(multimodal_vision_projector_inspect PRIVATE
        ../chapter01_tensor_loader/src/real_tokenizer.cpp
    )
    target_include_directories(multimodal_vision_projector_inspect PRIVATE
        ${LLAMA_CPP_ROOT}/include
        ${LLAMA_CPP_ROOT}/ggml/include
    )
    target_compile_definitions(multimodal_vision_projector_inspect PRIVATE
        CPP_LLM_MULTIMODAL_HAS_REAL_TOKENIZER=1
    )
    target_link_libraries(multimodal_vision_projector_inspect PRIVATE
        ${LLAMA_LIBRARY}
        ${GGML_LIBRARY}
        ${GGML_CPU_LIBRARY}
        ${GGML_BASE_LIBRARY}
        ${CMAKE_DL_LIBS}
    )
    set_target_properties(multimodal_vision_projector_inspect PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )
endif()

if(MULTIMODAL_ENABLE_SERVICE_INTEGRATION)
    if(NOT MULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER)
        message(FATAL_ERROR
            "MULTIMODAL_ENABLE_SERVICE_INTEGRATION requires MULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER")
    endif()
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../chapter06_generation
        ${CMAKE_CURRENT_BINARY_DIR}/chapter06_generation
    )
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../chapter10_serving
        ${CMAKE_CURRENT_BINARY_DIR}/chapter10_serving
    )
    target_sources(multimodal_vision_projector_inspect PRIVATE
        ../engine/src/embedding_request_executor.cpp
    )
    target_include_directories(multimodal_vision_projector_inspect PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../engine/include
    )
    target_link_libraries(multimodal_vision_projector_inspect PRIVATE
        chapter10_serving
    )
    target_compile_definitions(multimodal_vision_projector_inspect PRIVATE
        CPP_LLM_MULTIMODAL_HAS_SERVICE=1
    )
endif()

add_executable(multimodal_preprocessor_test tests/image_preprocessor_test.cpp)
target_link_libraries(multimodal_preprocessor_test PRIVATE cpp_llm_multimodal)
cpp_llm_prepare_test(multimodal_preprocessor_test)
target_compile_definitions(multimodal_preprocessor_test PRIVATE
    MULTIMODAL_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(multimodal_mmproj_schema_test tests/mmproj_schema_test.cpp)
target_link_libraries(multimodal_mmproj_schema_test PRIVATE cpp_llm_multimodal)
cpp_llm_prepare_test(multimodal_mmproj_schema_test)

add_executable(multimodal_vision_encoder_test tests/vision_encoder_test.cpp)
target_link_libraries(multimodal_vision_encoder_test PRIVATE cpp_llm_multimodal)
cpp_llm_prepare_test(multimodal_vision_encoder_test)

add_executable(multimodal_vision_weights_test tests/vision_weights_test.cpp)
target_link_libraries(multimodal_vision_weights_test PRIVATE cpp_llm_multimodal)
cpp_llm_prepare_test(multimodal_vision_weights_test)

add_executable(multimodal_vision_projector_test tests/vision_projector_test.cpp)
target_link_libraries(multimodal_vision_projector_test PRIVATE cpp_llm_multimodal)
cpp_llm_prepare_test(multimodal_vision_projector_test)

add_executable(multimodal_input_embedding_test tests/input_embedding_test.cpp)
target_link_libraries(multimodal_input_embedding_test PRIVATE cpp_llm_multimodal)
cpp_llm_prepare_test(multimodal_input_embedding_test)

add_executable(multimodal_smolvlm_prompt_test tests/smolvlm_prompt_test.cpp)
target_link_libraries(multimodal_smolvlm_prompt_test PRIVATE cpp_llm_multimodal)
cpp_llm_prepare_test(multimodal_smolvlm_prompt_test)

add_executable(multimodal_request_metrics_test tests/request_metrics_test.cpp)
target_link_libraries(multimodal_request_metrics_test PRIVATE cpp_llm_multimodal)
cpp_llm_prepare_test(multimodal_request_metrics_test)

include(CTest)
enable_testing()
add_test(NAME multimodal_preprocessor_test COMMAND multimodal_preprocessor_test)
add_test(NAME multimodal_mmproj_schema_test COMMAND multimodal_mmproj_schema_test)
add_test(NAME multimodal_vision_encoder_test COMMAND multimodal_vision_encoder_test)
add_test(NAME multimodal_vision_weights_test COMMAND multimodal_vision_weights_test)
add_test(NAME multimodal_vision_projector_test COMMAND multimodal_vision_projector_test)
add_test(NAME multimodal_input_embedding_test COMMAND multimodal_input_embedding_test)
add_test(NAME multimodal_smolvlm_prompt_test COMMAND multimodal_smolvlm_prompt_test)
add_test(NAME multimodal_request_metrics_test COMMAND multimodal_request_metrics_test)
cpp_llm_set_test_labels(multimodal_preprocessor_test unit multimodal preprocessing)
cpp_llm_set_test_labels(multimodal_mmproj_schema_test contract multimodal loader)
cpp_llm_set_test_labels(multimodal_vision_encoder_test component multimodal vision)
cpp_llm_set_test_labels(multimodal_vision_weights_test unit multimodal loader)
cpp_llm_set_test_labels(multimodal_vision_projector_test component multimodal projector)
cpp_llm_set_test_labels(multimodal_input_embedding_test integration multimodal embedding)
cpp_llm_set_test_labels(multimodal_smolvlm_prompt_test unit multimodal prompt)
cpp_llm_set_test_labels(multimodal_request_metrics_test unit multimodal metrics)
