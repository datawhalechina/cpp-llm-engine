// 多模态 CPU：图像预处理测试
// 职责：验证解码、resize、normalize、patch 顺序和错误路径。
// 契约：测试不依赖网络、真实模型、llama.cpp 或 GPU。
// 运行：由 CTest 执行 multimodal_preprocessor_test。
#include "multimodal/image.h"
#include "multimodal/image_preprocessor.h"
#include "multimodal/http_request.h"
#include "multimodal/idefics3_preprocessor.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#ifndef MULTIMODAL_TEST_SOURCE_DIR
#define MULTIMODAL_TEST_SOURCE_DIR "."
#endif

namespace {

int failures = 0;

#define CHECK(condition)                                                                   \
    do {                                                                                   \
        if (!(condition)) {                                                                \
            std::cerr << "CHECK failed at line " << __LINE__ << ": " #condition << '\n'; \
            ++failures;                                                                    \
        }                                                                                  \
    } while (false)

bool near(float lhs, float rhs, float tolerance = 1.0e-5f) {
    return std::fabs(lhs - rhs) <= tolerance;
}

void test_ppm_decode() {
    multimodal::RgbImage image;
    std::string error;
    const std::string path = std::string(MULTIMODAL_TEST_SOURCE_DIR) +
                             "/tests/fixtures/two_by_two.ppm";
    CHECK(multimodal::load_rgb_image(path, &image, &error));
    CHECK(error.empty());
    CHECK(image.valid());
    CHECK(image.width == 2);
    CHECK(image.height == 2);
    CHECK(image.pixels.size() == 12);
    CHECK(image.pixels[0] == 255);
    CHECK(image.pixels[4] == 255);
    CHECK(image.pixels[8] == 255);
    CHECK(image.pixels[9] == 255);
}

void test_multimodal_http_request() {
    const std::string ppm_base64 =
        "UDMKIyByZWQsIGdyZWVuLCBibHVlLCB3aGl0ZQoyIDIKMjU1CjI1NSAwIDAgICAwIDI1NSAw"
        "CjAgMCAyNTUgICAyNTUgMjU1IDI1NQo=";
    const std::string body =
        "{\"request_id\":1,\"prompt\":\"describe\\nimage\","
        "\"image_media_type\":\"image/ppm\",\"image_base64\":\"" +
        ppm_base64 + "\"}";
    multimodal::MultimodalHttpRequest request;
    std::string error;
    CHECK(multimodal::parse_multimodal_http_request(body, &request, &error));
    CHECK(error.empty());
    CHECK(request.prompt == "describe\nimage");
    CHECK(request.image.valid());
    CHECK(request.image.width == 2 && request.image.height == 2);

    const auto unchanged = request.image.pixels;
    CHECK(!multimodal::parse_multimodal_http_request(
        "{\"prompt\":\"x\",\"image_media_type\":\"image/ppm\","
        "\"image_base64\":\"%%%%\"}",
        &request,
        &error));
    CHECK(request.image.pixels == unchanged);
    CHECK(!multimodal::parse_multimodal_http_request(body, &request, &error, 4));
    CHECK(error.find("size limit") != std::string::npos);
    CHECK(!multimodal::parse_multimodal_http_request(
        "{\"prompt\":\"x\",\"image_media_type\":\"image/png\","
        "\"image_base64\":\"" + ppm_base64 + "\"}",
        &request,
        &error));
    CHECK(error.find("media type") != std::string::npos);
}

void test_resize_and_normalize() {
    const multimodal::RgbImage one_pixel{1, 1, {255, 128, 0}};
    const auto resized = multimodal::resize_bilinear(one_pixel, 2, 2);
    CHECK(resized.valid());
    CHECK(resized.pixels == std::vector<std::uint8_t>({
        255, 128, 0, 255, 128, 0, 255, 128, 0, 255, 128, 0}));

    const auto normalized = multimodal::normalize_chw(
        resized, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f});
    CHECK(normalized.valid());
    CHECK(near(normalized.at(0, 0, 0), 1.0f));
    CHECK(near(normalized.at(1, 0, 0), (128.0f / 255.0f - 0.5f) / 0.5f));
    CHECK(near(normalized.at(2, 1, 1), -1.0f));
}

void test_bilinear_center() {
    const multimodal::RgbImage input{
        2, 2,
        {0, 0, 0, 100, 100, 100, 200, 200, 200, 255, 255, 255}};
    const auto output = multimodal::resize_bilinear(input, 1, 1);
    CHECK(output.valid());
    CHECK(output.pixels == std::vector<std::uint8_t>({139, 139, 139}));
}

void test_patch_order() {
    multimodal::FloatImage image;
    image.channels = 1;
    image.width = 4;
    image.height = 2;
    image.values = {0, 1, 2, 3, 4, 5, 6, 7};
    const auto patches = multimodal::extract_patches(image, 2, 2);
    CHECK(patches.valid());
    CHECK(patches.patch_count == 2);
    CHECK(patches.patch_values == 4);
    CHECK(patches.values == std::vector<float>({0, 1, 4, 5, 2, 3, 6, 7}));
}

void test_smolvlm_shape_and_errors() {
    const multimodal::RgbImage input{1, 1, {0, 127, 255}};
    const multimodal::ImagePreprocessorConfig config{
        512, 512, 16, 16, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}};
    multimodal::PreprocessedImage output;
    std::string error;
    CHECK(multimodal::preprocess_image(input, config, &output, &error));
    CHECK(output.normalized.values.size() == 3U * 512U * 512U);
    CHECK(output.patches.patch_count == 32 * 32);
    CHECK(output.patches.patch_values == 3 * 16 * 16);
    CHECK(near(output.patches.at(0, 0), -1.0f));
    CHECK(near(output.patches.at(0, 2 * 16 * 16), 1.0f));

    auto invalid = config;
    invalid.patch_width = 15;
    CHECK(!multimodal::preprocess_image(input, invalid, &output, &error));
    CHECK(error.find("divisible") != std::string::npos);
    CHECK(!multimodal::preprocess_image(input, config, nullptr, &error));
    multimodal::RgbImage unchanged{1, 1, {1, 2, 3}};
    CHECK(!multimodal::load_rgb_image("not-an-image.txt", &unchanged, &error));
    CHECK(unchanged.pixels == std::vector<std::uint8_t>({1, 2, 3}));
}

void test_idefics3_layout() {
    const multimodal::Idefics3TilingConfig config{512, 16, 4, 256, 256};
    multimodal::Idefics3Layout layout;
    std::string error;
    CHECK(multimodal::calculate_idefics3_layout(640, 488, config, &layout, &error));
    CHECK(layout.refined_width == 1536);
    CHECK(layout.refined_height == 1024);
    CHECK(layout.content_width == 1343);
    CHECK(layout.content_height == 1024);
    CHECK(layout.offset_x == 96);
    CHECK(layout.offset_y == 0);
    CHECK(layout.grid_x == 3);
    CHECK(layout.grid_y == 2);
    CHECK(layout.visual_image_count() == 7);
}

void test_idefics3_tile_order() {
    multimodal::RgbImage input;
    input.width = 4;
    input.height = 2;
    input.pixels.reserve(4 * 2 * 3);
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 4; ++x) {
            if (x < 2) {
                input.pixels.insert(input.pixels.end(), {255, 0, 0});
            } else {
                input.pixels.insert(input.pixels.end(), {0, 0, 255});
            }
        }
    }

    const multimodal::Idefics3TilingConfig config{2, 1, 1, 8, 8};
    multimodal::Idefics3TilingResult result;
    std::string error;
    CHECK(multimodal::prepare_idefics3_tiles(input, config, &result, &error));
    CHECK(result.overview.is_overview());
    CHECK(result.tiles.size() == 2);
    CHECK(result.tiles[0].row == 0 && result.tiles[0].column == 0);
    CHECK(result.tiles[1].row == 0 && result.tiles[1].column == 1);
    CHECK(result.tiles[0].image.pixels[0] == 255);
    CHECK(result.tiles[0].image.pixels[2] == 0);
    CHECK(result.tiles[1].image.pixels[0] == 0);
    CHECK(result.tiles[1].image.pixels[2] == 255);

    auto invalid = config;
    invalid.min_image_tokens = 9;
    CHECK(!multimodal::prepare_idefics3_tiles(input, invalid, &result, &error));
    CHECK(error.find("exceed") != std::string::npos);
}

}  // namespace

int main() {
    test_ppm_decode();
    test_multimodal_http_request();
    test_resize_and_normalize();
    test_bilinear_center();
    test_patch_order();
    test_smolvlm_shape_and_errors();
    test_idefics3_layout();
    test_idefics3_tile_order();
    if (failures != 0) return 1;
    std::cout << "multimodal preprocessing checks passed\n";
    return 0;
}
