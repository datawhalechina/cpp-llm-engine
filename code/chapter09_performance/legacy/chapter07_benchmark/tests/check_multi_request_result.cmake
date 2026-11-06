# Chapter 07：执行多请求基准并检查机器可读结果的证据字段。
if(NOT DEFINED BENCHMARK_EXECUTABLE OR NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "BENCHMARK_EXECUTABLE and OUTPUT_FILE are required")
endif()

execute_process(
    COMMAND "${BENCHMARK_EXECUTABLE}"
        --warmup 0
        --iterations 2
        --output "${OUTPUT_FILE}"
    RESULT_VARIABLE benchmark_status
    OUTPUT_VARIABLE benchmark_output
    ERROR_VARIABLE benchmark_error
)

if(NOT benchmark_status EQUAL 0)
    message(FATAL_ERROR
        "benchmark failed (${benchmark_status})\n${benchmark_output}\n${benchmark_error}")
endif()

file(READ "${OUTPUT_FILE}" result_json)
foreach(required_text
        "\"schema_version\": 2"
        "\"experiment_id\": \"EXP-CPU-01\""
        "\"workload_id\": \"WL-CPU-TEACHING-BATCH-V1\""
        "\"environment\""
        "\"cpu_model\""
        "\"compiler\""
        "\"build_type\""
        "\"statistics\""
        "\"median\""
        "\"samples\""
        "\"execution_mode\": \"round_robin\""
        "\"execution_mode\": \"tensor_batch\""
        "\"model_invocations\""
        "\"tensor_batch_steps\""
        "\"iteration\": 2"
        "\"correctness\": true")
    string(FIND "${result_json}" "${required_text}" field_position)
    if(field_position EQUAL -1)
        message(FATAL_ERROR "missing result field: ${required_text}")
    endif()
endforeach()
