// 多模态服务：图文 JSON/Base64 请求解析
// 职责：校验字段、Base64 与图像类型，并限制请求解码后的内存规模。
// 契约：失败时不修改调用者输出；仅接受单张 JPEG 或 PPM 图像。
// 验证：image_preprocessor_test.cpp。
#include "multimodal/http_request.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace multimodal {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

bool read_json_string(
    const std::string& body,
    const std::string& key,
    std::string* value) {
    const std::string marker = "\"" + key + "\"";
    auto cursor = body.find(marker);
    if (cursor == std::string::npos) return false;
    cursor = body.find(':', cursor + marker.size());
    if (cursor == std::string::npos) return false;
    ++cursor;
    while (cursor < body.size() &&
           std::isspace(static_cast<unsigned char>(body[cursor]))) {
        ++cursor;
    }
    if (cursor >= body.size() || body[cursor] != '"') return false;
    ++cursor;

    std::string decoded;
    while (cursor < body.size()) {
        const char ch = body[cursor++];
        if (ch == '"') {
            *value = std::move(decoded);
            return true;
        }
        if (static_cast<unsigned char>(ch) < 0x20) return false;
        if (ch != '\\') {
            decoded.push_back(ch);
            continue;
        }
        if (cursor >= body.size()) return false;
        const char escaped = body[cursor++];
        switch (escaped) {
            case '"': decoded.push_back('"'); break;
            case '\\': decoded.push_back('\\'); break;
            case '/': decoded.push_back('/'); break;
            case 'b': decoded.push_back('\b'); break;
            case 'f': decoded.push_back('\f'); break;
            case 'n': decoded.push_back('\n'); break;
            case 'r': decoded.push_back('\r'); break;
            case 't': decoded.push_back('\t'); break;
            default: return false;
        }
    }
    return false;
}

int base64_value(unsigned char ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;
}

bool decode_base64(
    const std::string& encoded,
    std::size_t max_output_bytes,
    std::vector<std::uint8_t>* output) {
    if (encoded.empty() || encoded.size() % 4 != 0) return false;
    const std::size_t padding =
        (encoded.back() == '=' ? 1U : 0U) +
        (encoded.size() > 1 && encoded[encoded.size() - 2] == '=' ? 1U : 0U);
    const std::size_t groups = encoded.size() / 4;
    if (groups > std::numeric_limits<std::size_t>::max() / 3) return false;
    const std::size_t decoded_size = groups * 3 - padding;
    if (decoded_size == 0 || decoded_size > max_output_bytes) return false;

    std::vector<std::uint8_t> decoded;
    decoded.reserve(decoded_size);
    for (std::size_t offset = 0; offset < encoded.size(); offset += 4) {
        std::array<int, 4> values{};
        for (std::size_t index = 0; index < 4; ++index) {
            const char ch = encoded[offset + index];
            const bool is_final_group = offset + 4 == encoded.size();
            if (ch == '=') {
                if (!is_final_group || index < 2) return false;
                values[index] = 0;
            } else {
                if ((index > 1 && encoded[offset + index - 1] == '=') ||
                    (values[index] = base64_value(static_cast<unsigned char>(ch))) < 0) {
                    return false;
                }
            }
        }
        const std::uint32_t bits =
            (static_cast<std::uint32_t>(values[0]) << 18U) |
            (static_cast<std::uint32_t>(values[1]) << 12U) |
            (static_cast<std::uint32_t>(values[2]) << 6U) |
            static_cast<std::uint32_t>(values[3]);
        decoded.push_back(static_cast<std::uint8_t>(bits >> 16U));
        if (decoded.size() < decoded_size) {
            decoded.push_back(static_cast<std::uint8_t>(bits >> 8U));
        }
        if (decoded.size() < decoded_size) {
            decoded.push_back(static_cast<std::uint8_t>(bits));
        }
    }
    if (decoded.size() != decoded_size) return false;
    *output = std::move(decoded);
    return true;
}

}  // namespace

bool parse_multimodal_http_request(
    const std::string& body,
    MultimodalHttpRequest* output,
    std::string* error,
    std::size_t max_encoded_image_bytes) {
    if (output == nullptr) {
        set_error(error, "multimodal request output must not be null");
        return false;
    }
    std::string prompt;
    std::string media_type;
    std::string base64;
    if (!read_json_string(body, "prompt", &prompt) || prompt.empty() ||
        !read_json_string(body, "image_media_type", &media_type) ||
        !read_json_string(body, "image_base64", &base64)) {
        set_error(error, "request requires prompt, image_media_type, and image_base64 strings");
        return false;
    }

    std::vector<std::uint8_t> encoded;
    if (!decode_base64(base64, max_encoded_image_bytes, &encoded)) {
        set_error(error, "image_base64 is invalid or exceeds the configured size limit");
        return false;
    }
    RgbImage image;
    if (!decode_rgb_image(encoded, media_type, &image, error)) return false;

    MultimodalHttpRequest parsed;
    parsed.prompt = std::move(prompt);
    parsed.image = std::move(image);
    *output = std::move(parsed);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace multimodal
