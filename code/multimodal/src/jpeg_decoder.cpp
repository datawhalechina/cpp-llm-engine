// 多模态 CPU：JPEG 解码适配
// 职责：用系统 libjpeg 将 JPEG scanline 转换为连续 RGB8。
// 契约：libjpeg fatal error 通过 setjmp 转成失败返回，不跨越 C ABI 抛异常。
// 验证：multimodal_image_inspect 对真实 JPEG 的可选检查。
#include "multimodal/image.h"

#include <csetjmp>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>

extern "C" {
#include <jpeglib.h>
}

namespace multimodal {
namespace {

struct JpegErrorManager {
    jpeg_error_mgr base;
    std::jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
    std::uint8_t* pixels;
    bool decoder_created;
};

void jpeg_error_exit(j_common_ptr common) {
    auto* manager = reinterpret_cast<JpegErrorManager*>(common->err);
    (*common->err->format_message)(common, manager->message);
    std::longjmp(manager->jump, 1);
}

}  // namespace

bool load_jpeg(const std::string& path, RgbImage* output, std::string* error) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        if (error != nullptr) *error = "unable to open JPEG: " + path;
        return false;
    }

    jpeg_decompress_struct decoder{};
    auto* manager = static_cast<JpegErrorManager*>(std::calloc(1, sizeof(JpegErrorManager)));
    if (manager == nullptr) {
        std::fclose(file);
        if (error != nullptr) *error = "unable to allocate JPEG error state";
        return false;
    }
    decoder.err = jpeg_std_error(&manager->base);
    manager->base.error_exit = jpeg_error_exit;
    if (setjmp(manager->jump) != 0) {
        if (manager->decoder_created) jpeg_destroy_decompress(&decoder);
        std::free(manager->pixels);
        std::fclose(file);
        if (error != nullptr) *error = std::string("JPEG decode failed: ") + manager->message;
        std::free(manager);
        return false;
    }

    jpeg_create_decompress(&decoder);
    manager->decoder_created = true;
    jpeg_stdio_src(&decoder, file);
    jpeg_read_header(&decoder, TRUE);
    decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&decoder);

    const auto width = static_cast<std::size_t>(decoder.output_width);
    const auto height = static_cast<std::size_t>(decoder.output_height);
    if (decoder.output_components != 3 || width == 0 || height == 0 ||
        width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        width > std::numeric_limits<std::size_t>::max() / height / 3) {
        jpeg_destroy_decompress(&decoder);
        manager->decoder_created = false;
        std::fclose(file);
        if (error != nullptr) *error = "invalid JPEG output dimensions";
        std::free(manager);
        return false;
    }

    const auto row_bytes = width * 3;
    const auto pixel_bytes = row_bytes * height;
    manager->pixels = static_cast<std::uint8_t*>(std::malloc(pixel_bytes));
    if (manager->pixels == nullptr) {
        jpeg_destroy_decompress(&decoder);
        manager->decoder_created = false;
        std::fclose(file);
        if (error != nullptr) *error = "unable to allocate JPEG output buffer";
        std::free(manager);
        return false;
    }
    while (decoder.output_scanline < decoder.output_height) {
        JSAMPROW row = manager->pixels +
                       static_cast<std::size_t>(decoder.output_scanline) * row_bytes;
        jpeg_read_scanlines(&decoder, &row, 1);
    }
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    manager->decoder_created = false;
    std::fclose(file);

    RgbImage decoded;
    decoded.width = static_cast<int>(width);
    decoded.height = static_cast<int>(height);
    decoded.pixels.assign(manager->pixels, manager->pixels + pixel_bytes);
    std::free(manager->pixels);
    manager->pixels = nullptr;
    std::free(manager);
    *output = std::move(decoded);
    return true;
}

bool decode_jpeg(
    const std::vector<std::uint8_t>& encoded,
    RgbImage* output,
    std::string* error) {
    jpeg_decompress_struct decoder{};
    auto* manager = static_cast<JpegErrorManager*>(std::calloc(1, sizeof(JpegErrorManager)));
    if (manager == nullptr) {
        if (error != nullptr) *error = "unable to allocate JPEG error state";
        return false;
    }
    decoder.err = jpeg_std_error(&manager->base);
    manager->base.error_exit = jpeg_error_exit;
    if (setjmp(manager->jump) != 0) {
        if (manager->decoder_created) jpeg_destroy_decompress(&decoder);
        std::free(manager->pixels);
        if (error != nullptr) *error = std::string("JPEG decode failed: ") + manager->message;
        std::free(manager);
        return false;
    }

    jpeg_create_decompress(&decoder);
    manager->decoder_created = true;
    jpeg_mem_src(
        &decoder,
        const_cast<unsigned char*>(encoded.data()),
        static_cast<unsigned long>(encoded.size()));
    jpeg_read_header(&decoder, TRUE);
    decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&decoder);

    const auto width = static_cast<std::size_t>(decoder.output_width);
    const auto height = static_cast<std::size_t>(decoder.output_height);
    if (decoder.output_components != 3 || width == 0 || height == 0 ||
        width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        width > std::numeric_limits<std::size_t>::max() / height / 3) {
        jpeg_destroy_decompress(&decoder);
        manager->decoder_created = false;
        if (error != nullptr) *error = "invalid JPEG output dimensions";
        std::free(manager);
        return false;
    }

    const auto row_bytes = width * 3;
    const auto pixel_bytes = row_bytes * height;
    manager->pixels = static_cast<std::uint8_t*>(std::malloc(pixel_bytes));
    if (manager->pixels == nullptr) {
        jpeg_destroy_decompress(&decoder);
        manager->decoder_created = false;
        if (error != nullptr) *error = "unable to allocate JPEG output buffer";
        std::free(manager);
        return false;
    }
    while (decoder.output_scanline < decoder.output_height) {
        JSAMPROW row = manager->pixels +
                       static_cast<std::size_t>(decoder.output_scanline) * row_bytes;
        jpeg_read_scanlines(&decoder, &row, 1);
    }
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    manager->decoder_created = false;

    RgbImage decoded;
    decoded.width = static_cast<int>(width);
    decoded.height = static_cast<int>(height);
    decoded.pixels.assign(manager->pixels, manager->pixels + pixel_bytes);
    std::free(manager->pixels);
    manager->pixels = nullptr;
    std::free(manager);
    *output = std::move(decoded);
    return true;
}

}  // namespace multimodal
