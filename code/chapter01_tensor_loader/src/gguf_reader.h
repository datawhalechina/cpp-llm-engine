// Chapter 01：GGUF index reader 接口
// 职责：声明 GGUF 文件索引和 metadata 的读取入口。
// 契约：read_index 只建立索引，不负责 tensor payload 解码。
// 验证：chapter01_tensor_loader_test.cpp 与真实 TinyLlama inspect。
#pragma once

#include "model_metadata_adapter.h"

#include <cstdint>
#include <string>
#include <vector>

namespace chapter01 {

struct GgufTensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;
};

struct GgufModelIndex {
    std::uint32_t version = 0;
    std::uint64_t data_offset = 0;
    MetadataMap metadata;
    std::vector<GgufTensorInfo> tensors;
};

class GgufReader {
public:
    // 只读取文件头、元数据和张量索引，不加载权重 payload。
    static bool read_index(const std::string& path,
                           GgufModelIndex* index,
                           std::string* error = nullptr);
};

}  // namespace chapter01
