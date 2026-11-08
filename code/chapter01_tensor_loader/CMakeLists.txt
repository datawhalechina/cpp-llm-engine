cmake_minimum_required(VERSION 3.20)
project(chapter01_tensor_loader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/../testing/cmake/CppLlmTesting.cmake)

option(CHAPTER01_ENABLE_LLAMA_CPP "Build the real llama.cpp tokenizer adapter" OFF)
set(LLAMA_CPP_ROOT "" CACHE PATH "llama.cpp checkout containing include/ and build/bin/")
set(LLAMA_CPP_BUILD_DIR "" CACHE PATH "Optional llama.cpp build directory containing bin/")
set(CHAPTER01_REAL_MODEL "" CACHE FILEPATH "Optional GGUF model used by the structure CTest")

add_library(chapter01_tensor_loader
    src/tensor.cpp
    src/model_loader.cpp
    src/model_config.cpp
    src/model_metadata.cpp
    src/model_schema.cpp
    src/model_metadata_adapter.cpp
    src/gguf_reader.cpp
    src/gguf_tensor_loader.cpp
    src/model_builder.cpp
    src/tokenizer_metadata.cpp
    src/tokenizer.cpp
    src/real_layer_weights.cpp
)

target_include_directories(chapter01_tensor_loader
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_library(chapter01_test_cases STATIC
    tests/chapter01_tensor_loader_test.cpp
)
target_link_libraries(chapter01_test_cases PRIVATE chapter01_tensor_loader)
cpp_llm_prepare_test(chapter01_test_cases)

add_executable(chapter01_loader_test tests/chapter01_loader_test.cpp)
add_executable(chapter01_schema_test tests/chapter01_schema_test.cpp)
add_executable(chapter01_tokenizer_payload_test tests/chapter01_tokenizer_payload_test.cpp)

add_executable(gguf_inspect
    examples/gguf_inspect.cpp
)

add_executable(gguf_first_layer_inspect
    examples/gguf_first_layer_inspect.cpp
)

target_link_libraries(gguf_inspect
    PRIVATE chapter01_tensor_loader
)

target_link_libraries(gguf_first_layer_inspect
    PRIVATE chapter01_tensor_loader
)

target_link_libraries(chapter01_loader_test PRIVATE chapter01_test_cases)
cpp_llm_prepare_test(chapter01_loader_test)
target_link_libraries(chapter01_schema_test PRIVATE chapter01_test_cases)
cpp_llm_prepare_test(chapter01_schema_test)
target_link_libraries(chapter01_tokenizer_payload_test PRIVATE chapter01_test_cases)
cpp_llm_prepare_test(chapter01_tokenizer_payload_test)

enable_testing()
add_test(NAME chapter01_loader_test COMMAND chapter01_loader_test)
cpp_llm_set_test_labels(chapter01_loader_test "component;loader;cie1")
cpp_llm_set_test_contract(chapter01_loader_test OWNER 01 LEVEL component STATUS active)

add_test(NAME chapter01_schema_test COMMAND chapter01_schema_test)
cpp_llm_set_test_labels(chapter01_schema_test "component;loader;schema;gguf")
cpp_llm_set_test_contract(chapter01_schema_test OWNER 01 LEVEL component STATUS active)

add_test(NAME chapter01_tokenizer_payload_test
    COMMAND chapter01_tokenizer_payload_test)
cpp_llm_set_test_labels(chapter01_tokenizer_payload_test
    "component;loader;tokenizer;quantized_payload")
cpp_llm_set_test_contract(
    chapter01_tokenizer_payload_test OWNER 01 LEVEL component STATUS active)

if(CHAPTER01_REAL_MODEL)
    if(NOT EXISTS "${CHAPTER01_REAL_MODEL}")
        message(FATAL_ERROR "CHAPTER01_REAL_MODEL must point to an existing GGUF file")
    endif()
    add_test(
        NAME real_tinyllama_first_layer_structure_test
        COMMAND gguf_first_layer_inspect "${CHAPTER01_REAL_MODEL}"
    )
    cpp_llm_set_test_labels(real_tinyllama_first_layer_structure_test
        "real_model;loader;attention;structure")
    cpp_llm_set_test_contract(real_tinyllama_first_layer_structure_test
        OWNER 04 LEVEL real_model STATUS active)
endif()

if(CHAPTER01_ENABLE_LLAMA_CPP)
    if(NOT EXISTS "${LLAMA_CPP_ROOT}/include/llama.h")
        message(FATAL_ERROR "LLAMA_CPP_ROOT must point to a llama.cpp checkout")
    endif()
    if(NOT LLAMA_CPP_BUILD_DIR)
        set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_ROOT}/build")
    endif()
    if(NOT EXISTS "${LLAMA_CPP_BUILD_DIR}/bin")
        message(FATAL_ERROR "LLAMA_CPP_BUILD_DIR must contain bin/")
    endif()
    find_library(LLAMA_LIBRARY NAMES llama PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_LIBRARY NAMES ggml PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_BASE_LIBRARY NAMES ggml-base PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_CPU_LIBRARY NAMES ggml-cpu PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)

    add_library(chapter01_real_tokenizer
        src/real_tokenizer.cpp
    )
    target_include_directories(chapter01_real_tokenizer
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${LLAMA_CPP_ROOT}/include
            ${LLAMA_CPP_ROOT}/ggml/include
    )
    target_link_libraries(chapter01_real_tokenizer
        PUBLIC
            chapter01_tensor_loader
            ${LLAMA_LIBRARY}
            ${GGML_LIBRARY}
            ${GGML_CPU_LIBRARY}
            ${GGML_BASE_LIBRARY}
            ${CMAKE_DL_LIBS}
    )
    set_target_properties(chapter01_real_tokenizer PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )

    add_executable(llama_tokenizer_compare
        examples/llama_tokenizer_compare.cpp
    )
    target_link_libraries(llama_tokenizer_compare PRIVATE chapter01_real_tokenizer)
    set_target_properties(llama_tokenizer_compare PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
    )

endif()
