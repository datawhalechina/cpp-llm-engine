// Chapter 01：教学二进制模型 loader 接口
// 职责：声明从 CIE1 文件读取 Tensor Map 的入口。
// 契约：loader 只处理教程格式；真实 GGUF 由 GgufReader/GgufTensorLoader 负责。
// 验证：chapter01_tensor_loader_test.cpp。
#pragma once

#include "tensor.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace chapter01 {

class ModelLoader {
public:
    bool load(const std::filesystem::path& path, std::string* error = nullptr);

    const std::unordered_map<std::string, Tensor>& tensors() const;
    const Tensor* find(const std::string& name) const;
    std::vector<std::string> tensor_names() const;

private:
    std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace chapter01
