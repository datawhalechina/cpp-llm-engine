// CUDA 执行层：验证连续 device KV Cache 的生命周期与请求隔离。
// 职责：检查 D2D append 偏移、容量、shape、clear 复用、移动和独立缓存。
// 通过条件：读回值与输入一致，失败不推进 size，两个缓存互不影响。
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "check failed: " #condition " at line " << __LINE__ \
                      << '\n';                                                  \
            return false;                                                       \
        }                                                                       \
    } while (false)

namespace {

using cpp_llm_engine::core::DType;
using cpp_llm_engine::cuda::CudaKvCache;
using cpp_llm_engine::cuda::CudaStream;
using cpp_llm_engine::cuda::CudaTensor;

bool make_tensor(
    const std::vector<float>& values,
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dimension,
    const CudaStream& stream,
    CudaTensor* output) {
    CHECK(CudaTensor::create_contiguous(
        {tokens, heads, head_dimension}, DType::Float32, output).is_ok());
    CHECK(output->copy_from_host_async(
        values.data(), values.size() * sizeof(float), stream).is_ok());
    return true;
}

bool test_append_clear_and_move() {
    CudaStream stream;
    CudaKvCache cache;
    CHECK(CudaStream::create(&stream).is_ok());
    CHECK(CudaKvCache::create(4, 2, 3, &cache).is_ok());
    CHECK(cache.valid());
    CHECK(cache.token_width() == 6);

    const std::vector<float> key_a{
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12};
    const std::vector<float> value_a{
        101, 102, 103, 104, 105, 106,
        107, 108, 109, 110, 111, 112};
    const std::vector<float> key_b{13, 14, 15, 16, 17, 18};
    const std::vector<float> value_b{113, 114, 115, 116, 117, 118};
    CudaTensor key_tensor_a;
    CudaTensor value_tensor_a;
    CudaTensor key_tensor_b;
    CudaTensor value_tensor_b;
    CHECK(make_tensor(key_a, 2, 2, 3, stream, &key_tensor_a));
    CHECK(make_tensor(value_a, 2, 2, 3, stream, &value_tensor_a));
    CHECK(make_tensor(key_b, 1, 2, 3, stream, &key_tensor_b));
    CHECK(make_tensor(value_b, 1, 2, 3, stream, &value_tensor_b));
    CHECK(cache.append(key_tensor_a, value_tensor_a, stream).is_ok());
    CHECK(cache.append(key_tensor_b, value_tensor_b, stream).is_ok());
    CHECK(cache.size_tokens() == 3);

    std::vector<float> key_output(18);
    std::vector<float> value_output(18);
    CHECK(cache.copy_to_host_async(
        key_output.data(), value_output.data(), 3, stream).is_ok());
    CHECK(stream.synchronize().is_ok());
    for (std::size_t index = 0; index < key_a.size(); ++index) {
        CHECK(key_output[index] == key_a[index]);
        CHECK(value_output[index] == value_a[index]);
    }
    for (std::size_t index = 0; index < key_b.size(); ++index) {
        CHECK(key_output[key_a.size() + index] == key_b[index]);
        CHECK(value_output[value_a.size() + index] == value_b[index]);
    }

    CudaKvCache moved = std::move(cache);
    CHECK(moved.valid());
    CHECK(moved.size_tokens() == 3);
    CHECK(!cache.valid());
    CHECK(moved.trim(1).is_ok());
    CHECK(moved.size_tokens() == 1);
    CHECK(!moved.trim(2).is_ok());
    CHECK(moved.trim(0).is_ok());
    CHECK(moved.append(key_tensor_b, value_tensor_b, stream).is_ok());
    std::vector<float> cleared_key(6);
    std::vector<float> cleared_value(6);
    CHECK(moved.copy_to_host_async(
        cleared_key.data(), cleared_value.data(), 1, stream).is_ok());
    CHECK(stream.synchronize().is_ok());
    CHECK(cleared_key == key_b);
    CHECK(cleared_value == value_b);
    CHECK(moved.reset().is_ok());
    CHECK(!moved.valid());
    CHECK(!moved.trim(0).is_ok());
    return true;
}

bool test_failures_and_isolation() {
    CudaStream stream;
    CudaStream invalid_stream;
    CudaKvCache first;
    CudaKvCache second;
    CHECK(CudaStream::create(&stream).is_ok());
    CHECK(!CudaKvCache::create(1, 1, 1, nullptr).is_ok());
    CHECK(!CudaKvCache::create(0, 1, 1, &first).is_ok());
    CHECK(!CudaKvCache::create(
        std::numeric_limits<std::size_t>::max(), 2, 2, &first).is_ok());
    CHECK(CudaKvCache::create(2, 1, 2, &first).is_ok());
    CHECK(CudaKvCache::create(2, 1, 2, &second).is_ok());

    const std::vector<float> first_values{1, 2};
    const std::vector<float> second_values{9, 10};
    CudaTensor first_tensor;
    CudaTensor second_tensor;
    CudaTensor wrong_shape;
    CHECK(make_tensor(first_values, 1, 1, 2, stream, &first_tensor));
    CHECK(make_tensor(second_values, 1, 1, 2, stream, &second_tensor));
    CHECK(make_tensor(first_values, 1, 2, 1, stream, &wrong_shape));
    CHECK(!first.append(first_tensor, first_tensor, invalid_stream).is_ok());
    CHECK(!first.append(wrong_shape, wrong_shape, stream).is_ok());
    CHECK(first.size_tokens() == 0);
    CHECK(first.append(first_tensor, first_tensor, stream).is_ok());
    CHECK(second.append(second_tensor, second_tensor, stream).is_ok());
    first.clear();
    CHECK(first.size_tokens() == 0);
    CHECK(second.size_tokens() == 1);

    std::vector<float> key(2);
    std::vector<float> value(2);
    CHECK(second.copy_to_host_async(key.data(), value.data(), 1, stream).is_ok());
    CHECK(stream.synchronize().is_ok());
    CHECK(key == second_values);
    CHECK(value == second_values);
    CHECK(!second.copy_to_host_async(key.data(), value.data(), 2, stream).is_ok());
    CHECK(second.append(first_tensor, first_tensor, stream).is_ok());
    CHECK(!second.append(first_tensor, first_tensor, stream).is_ok());
    CHECK(second.size_tokens() == 2);
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) return 1;
    if (!test_append_clear_and_move()) return 1;
    if (!test_failures_and_isolation()) return 1;
    return 0;
}

#undef CHECK
