// CUDA GEMV 实验：测量 Q4_K/Q6_K 在 batch=1/2/4/8 下的融合 kernel 路径。
// 口径：固定 2048x2048 权重，计时包含 kernel 提交和末尾同步，不含分配与传输。
// 验证：每个 batch 行与同输入的单行调用逐元素比较；本程序不注册为 CTest。
#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using cpp_llm_engine::core::DType;
using cpp_llm_engine::core::WeightFormat;
using cpp_llm_engine::cuda::CudaExecutionContext;
using cpp_llm_engine::cuda::CudaQuantizedMatrix;
using cpp_llm_engine::cuda::CudaTensor;

constexpr std::size_t kRows = 2048;
constexpr std::size_t kCols = 2048;
constexpr std::size_t kElementsPerBlock = 256;
constexpr std::size_t kQ4KBlockBytes = 144;
constexpr std::size_t kQ6KBlockBytes = 210;

struct Options final {
    std::vector<std::size_t> batches{1, 2, 4, 8};
    std::string format = "all";
    std::size_t iterations = 50;
};

bool parse_positive(std::string_view text, std::size_t* value) {
    if (text.empty() || value == nullptr) return false;
    std::size_t parsed = 0;
    for (const char character : text) {
        if (character < '0' || character > '9') return false;
        const std::size_t digit = static_cast<std::size_t>(character - '0');
        if (parsed > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
            return false;
        }
        parsed = parsed * 10 + digit;
    }
    if (parsed == 0) return false;
    *value = parsed;
    return true;
}

bool parse_options(int argc, char** argv, Options* options) {
    if (options == nullptr) return false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (option == "--batch" && index + 1 < argc) {
            std::size_t batch = 0;
            if (!parse_positive(argv[++index], &batch)) return false;
            options->batches = {batch};
        } else if (option == "--format" && index + 1 < argc) {
            options->format = argv[++index];
            if (options->format != "q4" && options->format != "q6" &&
                options->format != "all") {
                return false;
            }
        } else if (option == "--iterations" && index + 1 < argc) {
            if (!parse_positive(argv[++index], &options->iterations)) return false;
        } else {
            return false;
        }
    }
    return true;
}

std::vector<std::uint8_t> make_payload(WeightFormat format) {
    const std::size_t block_bytes =
        format == WeightFormat::Q4K ? kQ4KBlockBytes : kQ6KBlockBytes;
    const std::size_t block_count = kRows * kCols / kElementsPerBlock;
    std::vector<std::uint8_t> payload(block_count * block_bytes, 0);
    for (std::size_t block = 0; block < block_count; ++block) {
        std::uint8_t* current = payload.data() + block * block_bytes;
        if (format == WeightFormat::Q4K) {
            current[0] = 0x00;
            current[1] = 0x34;  // fp16 0.25
            current[2] = 0x00;
            current[3] = 0x2c;  // fp16 0.0625
            for (std::size_t index = 0; index < 12; ++index) {
                current[4 + index] = static_cast<std::uint8_t>(1 + index * 5);
            }
            for (std::size_t index = 0; index < 128; ++index) {
                current[16 + index] =
                    static_cast<std::uint8_t>((index * 29 + block * 7) & 0xff);
            }
        } else {
            for (std::size_t index = 0; index < 192; ++index) {
                current[index] =
                    static_cast<std::uint8_t>((index * 17 + block * 11) & 0xff);
            }
            for (std::size_t index = 0; index < 16; ++index) {
                current[192 + index] = static_cast<std::uint8_t>(1 + index % 7);
            }
            current[208] = 0x00;
            current[209] = 0x34;  // fp16 0.25
        }
    }
    return payload;
}

bool check_status(const cpp_llm_engine::core::Status& status, const char* action) {
    if (status.is_ok()) return true;
    std::cerr << action << " failed: " << status.message() << '\n';
    return false;
}

bool run_format(
    WeightFormat format,
    std::string_view format_name,
    const Options& options) {
    CudaExecutionContext context;
    if (!check_status(CudaExecutionContext::create(&context), "create context")) {
        return false;
    }
    CudaQuantizedMatrix weights;
    if (!check_status(
            CudaQuantizedMatrix::create(kRows, kCols, format, &weights),
            "create weights")) {
        return false;
    }
    const auto payload = make_payload(format);
    if (!check_status(
            context.upload(&weights, payload.data(), payload.size()),
            "upload weights")) {
        return false;
    }

    double single_mean_ms = 0.0;
    for (const std::size_t batch : options.batches) {
        std::vector<float> input_host(batch * kCols);
        for (std::size_t row = 0; row < batch; ++row) {
            for (std::size_t col = 0; col < kCols; ++col) {
                input_host[row * kCols + col] =
                    static_cast<float>(static_cast<int>((row * 13 + col) % 31) - 15) /
                    16.0F;
            }
        }
        CudaTensor input;
        CudaTensor output;
        if (!check_status(
                CudaTensor::create_contiguous(
                    {batch, kCols}, DType::Float32, &input),
                "create input") ||
            !check_status(
                CudaTensor::create_contiguous(
                    {batch, kRows}, DType::Float32, &output),
                "create output") ||
            !check_status(
                context.upload(
                    &input, input_host.data(), input_host.size() * sizeof(float)),
                "upload input")) {
            return false;
        }
        for (int warmup = 0; warmup < 5; ++warmup) {
            if (!check_status(
                    context.quantized_linear_batch(weights, input, &output),
                    "warmup")) {
                return false;
            }
        }
        if (!check_status(context.synchronize(), "warmup synchronize")) return false;
        const auto start = Clock::now();
        for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
            if (!check_status(
                    context.quantized_linear_batch(weights, input, &output),
                    "measure")) {
                return false;
            }
        }
        if (!check_status(context.synchronize(), "measure synchronize")) return false;
        const double mean_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - start).count() /
            static_cast<double>(options.iterations);
        if (batch == 1) single_mean_ms = mean_ms;

        std::vector<float> batch_output(batch * kRows);
        if (!check_status(
                context.download(
                    output, batch_output.data(), batch_output.size() * sizeof(float)),
                "download batch") ||
            !check_status(context.synchronize(), "download batch synchronize")) {
            return false;
        }
        CudaTensor single_input;
        CudaTensor single_output;
        if (!check_status(
                CudaTensor::create_contiguous({kCols}, DType::Float32, &single_input),
                "create single input") ||
            !check_status(
                CudaTensor::create_contiguous({kRows}, DType::Float32, &single_output),
                "create single output")) {
            return false;
        }
        std::vector<float> reference(kRows);
        float max_error = 0.0F;
        for (std::size_t row = 0; row < batch; ++row) {
            if (!check_status(
                    context.upload(
                        &single_input,
                        input_host.data() + row * kCols,
                        kCols * sizeof(float)),
                    "upload single") ||
                !check_status(
                    context.quantized_linear(weights, single_input, &single_output),
                    "single reference") ||
                !check_status(
                    context.download(
                        single_output, reference.data(), reference.size() * sizeof(float)),
                    "download single") ||
                !check_status(context.synchronize(), "single synchronize")) {
                return false;
            }
            for (std::size_t col = 0; col < kRows; ++col) {
                max_error = std::max(
                    max_error,
                    std::fabs(batch_output[row * kRows + col] - reference[col]));
            }
        }
        std::cout << "format=" << format_name
                  << " batch=" << batch
                  << " rows=" << kRows
                  << " cols=" << kCols
                  << " iterations=" << options.iterations
                  << " mean_ms=" << mean_ms
                  << " speedup_vs_repeated_single=";
        if (single_mean_ms > 0.0) {
            std::cout << static_cast<double>(batch) * single_mean_ms / mean_ms;
        } else {
            std::cout << "not_measured";
        }
        std::cout << " max_error=" << max_error << '\n';
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        std::cerr << "usage: cpp_llm_engine_cuda_quantized_gemv_matrix "
                     "[--batch N] [--format q4|q6|all] [--iterations N]\n";
        return 2;
    }
    if ((options.format == "q4" || options.format == "all") &&
        !run_format(WeightFormat::Q4K, "Q4_K", options)) {
        return 1;
    }
    if ((options.format == "q6" || options.format == "all") &&
        !run_format(WeightFormat::Q6K, "Q6_K", options)) {
        return 1;
    }
    return 0;
}
