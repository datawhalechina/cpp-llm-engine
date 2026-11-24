cmake_minimum_required(VERSION 3.20)
project(cpp_llm_multimodal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

option(MULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER "Enable the optional real tokenizer path" OFF)
option(MULTIMODAL_ENABLE_CBLAS "Enable the optional CBLAS vision compute backend" OFF)
option(MULTIMODAL_ENABLE_CUDA "Enable optional CUDA vision adapters" OFF)
option(MULTIMODAL_ENABLE_SERVICE_INTEGRATION
    "Enable the optional Chapter10 service smoke in the real projector inspector" OFF)
set(MULTIMODAL_CBLAS_LIBRARY "" CACHE FILEPATH "Optional CBLAS library override")
set(LLAMA_CPP_ROOT "" CACHE PATH "llama.cpp source checkout")
set(LLAMA_CPP_BUILD_DIR "" CACHE PATH "llama.cpp build directory containing bin/")
set(MULTIMODAL_CUDA_SMOLVLM_MMPROJ "" CACHE FILEPATH
    "Optional SmolVLM mmproj for the CUDA vision-to-context test")
set(MULTIMODAL_CUDA_SMOLVLM_MODEL "" CACHE FILEPATH
    "Optional SmolVLM text GGUF for the CUDA vision-to-context test")
set(MULTIMODAL_CUDA_SMOLVLM_IMAGE "" CACHE FILEPATH
    "Optional image for the CUDA vision-to-context test")

if(NOT TARGET chapter01_tensor_loader)
    set(CHAPTER01_BUILD_STANDALONE_TARGETS OFF)
    if(MULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER)
        set(CHAPTER01_ENABLE_LLAMA_CPP ON)
    endif()
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../chapter01_tensor_loader
        ${CMAKE_CURRENT_BINARY_DIR}/chapter01_tensor_loader
        EXCLUDE_FROM_ALL
    )
    unset(CHAPTER01_BUILD_STANDALONE_TARGETS)
    unset(CHAPTER01_ENABLE_LLAMA_CPP)
endif()

if(NOT TARGET chapter05_quantization)
    set(CHAPTER05_BUILD_STANDALONE_TARGETS OFF)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../chapter05_quantization
        ${CMAKE_CURRENT_BINARY_DIR}/chapter05_quantization
        EXCLUDE_FROM_ALL
    )
    unset(CHAPTER05_BUILD_STANDALONE_TARGETS)
endif()

if(MULTIMODAL_ENABLE_CUDA AND NOT TARGET cpp_llm_engine_cuda)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
        ${CMAKE_CURRENT_BINARY_DIR}/cuda
    )
endif()

if(NOT TARGET cpp_llm_engine_model)
    set(CPP_LLM_MODEL_BUILD_STANDALONE_TARGETS OFF)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../model
        ${CMAKE_CURRENT_BINARY_DIR}/model
        EXCLUDE_FROM_ALL
    )
    unset(CPP_LLM_MODEL_BUILD_STANDALONE_TARGETS)
endif()

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
)
target_include_directories(cpp_llm_multimodal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(cpp_llm_multimodal
    PUBLIC chapter01_tensor_loader
    PRIVATE chapter05_quantization
)

if(MULTIMODAL_ENABLE_CUDA)
    add_library(cpp_llm_multimodal_cuda
        src/cuda_siglip_block.cpp
        src/cuda_vision_encoder.cpp
        src/cuda_vision_embedding.cpp
        src/cuda_vision_pipeline.cpp
        src/cuda_vision_projector.cpp
    )
    target_include_directories(cpp_llm_multimodal_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(cpp_llm_multimodal_cuda PUBLIC
        cpp_llm_multimodal
        cpp_llm_engine_cuda
    )
    add_executable(multimodal_cuda_vision_embedding_test
        tests/cuda_vision_embedding_test.cpp
    )
    target_link_libraries(multimodal_cuda_vision_embedding_test PRIVATE
        cpp_llm_multimodal_cuda
    )
    cpp_llm_prepare_test(multimodal_cuda_vision_embedding_test)
    add_executable(multimodal_cuda_vision_projector_test
        tests/cuda_vision_projector_test.cpp
    )
    target_link_libraries(multimodal_cuda_vision_projector_test PRIVATE
        cpp_llm_multimodal_cuda
    )
    cpp_llm_prepare_test(multimodal_cuda_vision_projector_test)
    add_executable(multimodal_cuda_siglip_block_test
        tests/cuda_siglip_block_test.cpp
    )
    target_link_libraries(multimodal_cuda_siglip_block_test PRIVATE
        cpp_llm_multimodal_cuda
    )
    cpp_llm_prepare_test(multimodal_cuda_siglip_block_test)
    add_executable(multimodal_cuda_vision_encoder_test
        tests/cuda_vision_encoder_test.cpp
    )
    target_link_libraries(multimodal_cuda_vision_encoder_test PRIVATE
        cpp_llm_multimodal_cuda
    )
    cpp_llm_prepare_test(multimodal_cuda_vision_encoder_test)
    add_executable(multimodal_cuda_vision_block_inspect
        examples/cuda_vision_block_inspect.cpp
    )
    target_link_libraries(multimodal_cuda_vision_block_inspect PRIVATE
        cpp_llm_multimodal_cuda
    )
    add_executable(multimodal_cuda_vision_encoder_inspect
        examples/cuda_vision_encoder_inspect.cpp
    )
    target_link_libraries(multimodal_cuda_vision_encoder_inspect PRIVATE
        cpp_llm_multimodal_cuda
    )
    add_executable(multimodal_cuda_vision_pipeline_inspect
        examples/cuda_vision_pipeline_inspect.cpp
    )
    target_link_libraries(multimodal_cuda_vision_pipeline_inspect PRIVATE
        cpp_llm_multimodal_cuda
    )
    add_executable(multimodal_cuda_smolvlm_context_inspect
        examples/cuda_smolvlm_context_inspect.cpp
    )
    target_link_libraries(multimodal_cuda_smolvlm_context_inspect PRIVATE
        cpp_llm_multimodal_cuda
        cpp_llm_engine_llama_cpu_model
        cpp_llm_engine_llama_cuda_model
    )
    if(MULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER)
        if(NOT TARGET chapter06_generation)
            add_subdirectory(
                ${CMAKE_CURRENT_SOURCE_DIR}/../chapter06_generation
                ${CMAKE_CURRENT_BINARY_DIR}/chapter06_generation
            )
        endif()
        target_sources(multimodal_cuda_smolvlm_context_inspect PRIVATE
            ../engine/src/embedding_request_executor.cpp
        )
        target_include_directories(multimodal_cuda_smolvlm_context_inspect PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../engine/include
        )
        target_compile_definitions(multimodal_cuda_smolvlm_context_inspect PRIVATE
            CPP_LLM_MULTIMODAL_HAS_REAL_TOKENIZER=1
        )
        target_link_libraries(multimodal_cuda_smolvlm_context_inspect PRIVATE
            chapter01_real_tokenizer
            chapter06_generation
        )
        set_target_properties(multimodal_cuda_smolvlm_context_inspect PROPERTIES
            BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
        )
    endif()
endif()

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
target_link_libraries(multimodal_vision_projector_inspect PRIVATE
    cpp_llm_multimodal
    cpp_llm_engine_llama_cpu_model
)

if(MULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER)
    if(NOT TARGET chapter01_real_tokenizer)
        message(FATAL_ERROR
            "The real tokenizer requires the chapter01_real_tokenizer target")
    endif()
    target_compile_definitions(multimodal_vision_projector_inspect PRIVATE
        CPP_LLM_MULTIMODAL_HAS_REAL_TOKENIZER=1
    )
    target_link_libraries(multimodal_vision_projector_inspect PRIVATE
        chapter01_real_tokenizer
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
    if(NOT TARGET chapter06_generation)
        add_subdirectory(
            ${CMAKE_CURRENT_SOURCE_DIR}/../chapter06_generation
            ${CMAKE_CURRENT_BINARY_DIR}/chapter06_generation
        )
    endif()
    if(NOT TARGET chapter10_serving)
        add_subdirectory(
            ${CMAKE_CURRENT_SOURCE_DIR}/../chapter10_serving
            ${CMAKE_CURRENT_BINARY_DIR}/chapter10_serving
        )
    endif()
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
    if(MULTIMODAL_ENABLE_CUDA)
        target_link_libraries(multimodal_cuda_smolvlm_context_inspect PRIVATE
            chapter10_serving
        )
        target_compile_definitions(multimodal_cuda_smolvlm_context_inspect PRIVATE
            CPP_LLM_MULTIMODAL_HAS_SERVICE=1
        )
    endif()
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
if(MULTIMODAL_ENABLE_CUDA)
    add_test(
        NAME multimodal_cuda_vision_embedding_test
        COMMAND multimodal_cuda_vision_embedding_test)
    add_test(
        NAME multimodal_cuda_vision_projector_test
        COMMAND multimodal_cuda_vision_projector_test)
    add_test(
        NAME multimodal_cuda_siglip_block_test
        COMMAND multimodal_cuda_siglip_block_test)
    add_test(
        NAME multimodal_cuda_vision_encoder_test
        COMMAND multimodal_cuda_vision_encoder_test)
    set(_smolvlm_test_assets
        "${MULTIMODAL_CUDA_SMOLVLM_MMPROJ}"
        "${MULTIMODAL_CUDA_SMOLVLM_MODEL}"
        "${MULTIMODAL_CUDA_SMOLVLM_IMAGE}")
    set(_smolvlm_test_requested FALSE)
    foreach(_asset IN LISTS _smolvlm_test_assets)
        if(NOT "${_asset}" STREQUAL "")
            set(_smolvlm_test_requested TRUE)
        endif()
    endforeach()
    if(_smolvlm_test_requested)
        foreach(_asset IN LISTS _smolvlm_test_assets)
            if("${_asset}" STREQUAL "" OR NOT EXISTS "${_asset}")
                message(FATAL_ERROR
                    "SmolVLM CUDA context test requires three existing assets")
            endif()
        endforeach()
        add_test(
            NAME multimodal_cuda_smolvlm_context_test
            COMMAND multimodal_cuda_smolvlm_context_inspect
                "${MULTIMODAL_CUDA_SMOLVLM_MMPROJ}"
                "${MULTIMODAL_CUDA_SMOLVLM_MODEL}"
                "${MULTIMODAL_CUDA_SMOLVLM_IMAGE}")
        cpp_llm_set_test_labels(
            multimodal_cuda_smolvlm_context_test
            real_model slow cuda multimodal vision generation integration)
        cpp_llm_set_test_contract(
            multimodal_cuda_smolvlm_context_test
            OWNER 06 LEVEL real_model STATUS active)
    endif()
endif()
cpp_llm_set_test_labels(multimodal_preprocessor_test unit multimodal preprocessing)
cpp_llm_set_test_labels(multimodal_mmproj_schema_test contract multimodal loader)
cpp_llm_set_test_labels(multimodal_vision_encoder_test component multimodal vision)
cpp_llm_set_test_labels(multimodal_vision_weights_test unit multimodal loader)
cpp_llm_set_test_labels(multimodal_vision_projector_test component multimodal projector)
cpp_llm_set_test_labels(multimodal_input_embedding_test integration multimodal embedding)
cpp_llm_set_test_labels(multimodal_smolvlm_prompt_test unit multimodal prompt)
cpp_llm_set_test_labels(multimodal_request_metrics_test unit multimodal metrics)
if(MULTIMODAL_ENABLE_CUDA)
    cpp_llm_set_test_labels(
        multimodal_cuda_vision_embedding_test
        component cuda multimodal vision embedding)
    cpp_llm_set_test_labels(
        multimodal_cuda_vision_projector_test
        component cuda multimodal vision projector)
    cpp_llm_set_test_labels(
        multimodal_cuda_siglip_block_test
        component cuda multimodal vision siglip attention)
    cpp_llm_set_test_labels(
        multimodal_cuda_vision_encoder_test
        integration cuda multimodal vision siglip encoder)
endif()
