// 多模态 CPU：图像数据契约
// 职责：定义解码后的 RGB 字节图像和预处理阶段的 Float32 CHW 图像。
// 契约：数据连续存储；RGB 为 HWC，FloatImage 为 CHW；维度和元素数必须一致。
// 验证：image_preprocessor_test.cpp。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace multimodal {

struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    bool valid() const;
};

struct FloatImage {
    int channels = 0;
    int width = 0;
    int height = 0;
    std::vector<float> values;

    bool valid() const;
    float at(int channel, int y, int x) const;
};

bool load_rgb_image(const std::string& path, RgbImage* output, std::string* error = nullptr);
bool decode_rgb_image(
    const std::vector<std::uint8_t>& encoded,
    const std::string& media_type,
    RgbImage* output,
    std::string* error = nullptr);
bool jpeg_decoder_available();

}  // namespace multimodal

