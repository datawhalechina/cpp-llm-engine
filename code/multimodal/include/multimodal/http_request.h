// 多模态服务：图文 HTTP 请求数据契约
// 职责：将受限大小的 JSON/Base64 输入转换为 prompt 与 RGB 图像。
// 契约：协议解析止于多模态边界；Chapter10 Transport 不感知图像格式。
// 验证：image_preprocessor_test.cpp。
#pragma once

#include "multimodal/image.h"

#include <cstddef>
#include <string>

namespace multimodal {

struct MultimodalHttpRequest {
    std::string prompt;
    RgbImage image;
};

bool parse_multimodal_http_request(
    const std::string& body,
    MultimodalHttpRequest* output,
    std::string* error = nullptr,
    std::size_t max_encoded_image_bytes = 20U * 1024U * 1024U);

}  // namespace multimodal
