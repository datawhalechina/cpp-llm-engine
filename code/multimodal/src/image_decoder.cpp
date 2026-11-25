// 多模态 CPU：图像解码入口
// 职责：提供零额外依赖的 PPM 解码，并在可用时分派到 libjpeg。
// 契约：成功时统一输出三通道 RGB8；失败时不修改调用者的输出对象。
// 验证：image_preprocessor_test.cpp 与可选真实 JPEG inspect。
#include "multimodal/image.h"

#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>

namespace multimodal {

#if defined(CPP_LLM_HAS_LIBJPEG)
bool load_jpeg(const std::string& path, RgbImage* output, std::string* error);
bool decode_jpeg(
    const std::vector<std::uint8_t>& encoded,
    RgbImage* output,
    std::string* error);
#endif

namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

bool checked_rgb_size(int width, int height, std::size_t* size) {
    if (width <= 0 || height <= 0) return false;
    const auto w = static_cast<std::size_t>(width);
    const auto h = static_cast<std::size_t>(height);
    if (w > std::numeric_limits<std::size_t>::max() / h ||
        w * h > std::numeric_limits<std::size_t>::max() / 3) {
        return false;
    }
    *size = w * h * 3;
    return true;
}

bool next_ppm_token(std::istream& input, std::string* token) {
    token->clear();
    char ch = 0;
    while (input.get(ch)) {
        if (ch == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            token->push_back(ch);
            break;
        }
    }
    while (input.get(ch)) {
        if (ch == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            break;
        }
        if (std::isspace(static_cast<unsigned char>(ch))) break;
        token->push_back(ch);
    }
    return !token->empty();
}

bool parse_int_token(std::istream& input, int* value) {
    std::string token;
    if (!next_ppm_token(input, &token)) return false;
    std::istringstream parser(token);
    parser >> *value;
    return parser && parser.eof();
}

bool decode_ppm(std::istream& input, RgbImage* output, std::string* error) {
    std::string magic;
    if (!next_ppm_token(input, &magic) || (magic != "P3" && magic != "P6")) {
        set_error(error, "unsupported PPM encoding");
        return false;
    }

    int width = 0;
    int height = 0;
    int max_value = 0;
    std::size_t size = 0;
    if (!parse_int_token(input, &width) || !parse_int_token(input, &height) ||
        !parse_int_token(input, &max_value) || max_value != 255 ||
        !checked_rgb_size(width, height, &size)) {
        set_error(error, "invalid PPM header; expected positive dimensions and max value 255");
        return false;
    }

    RgbImage decoded{width, height, std::vector<std::uint8_t>(size)};
    if (magic == "P6") {
        input.read(reinterpret_cast<char*>(decoded.pixels.data()),
                   static_cast<std::streamsize>(decoded.pixels.size()));
        if (!input) {
            set_error(error, "truncated PPM pixel payload");
            return false;
        }
    } else {
        for (auto& pixel : decoded.pixels) {
            int value = 0;
            if (!parse_int_token(input, &value) || value < 0 || value > 255) {
                set_error(error, "invalid PPM pixel value");
                return false;
            }
            pixel = static_cast<std::uint8_t>(value);
        }
    }

    *output = std::move(decoded);
    return true;
}

bool load_ppm(const std::string& path, RgbImage* output, std::string* error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        set_error(error, "unable to open image: " + path);
        return false;
    }
    return decode_ppm(input, output, error);
}

std::string lowercase_extension(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return {};
    std::string extension = path.substr(dot);
    for (auto& ch : extension) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return extension;
}

}  // namespace

bool RgbImage::valid() const {
    std::size_t expected = 0;
    return checked_rgb_size(width, height, &expected) && pixels.size() == expected;
}

bool FloatImage::valid() const {
    if (channels <= 0 || width <= 0 || height <= 0) return false;
    const auto c = static_cast<std::size_t>(channels);
    const auto w = static_cast<std::size_t>(width);
    const auto h = static_cast<std::size_t>(height);
    if (c > std::numeric_limits<std::size_t>::max() / w ||
        c * w > std::numeric_limits<std::size_t>::max() / h) {
        return false;
    }
    return values.size() == c * w * h;
}

float FloatImage::at(int channel, int y, int x) const {
    const auto plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    return values[static_cast<std::size_t>(channel) * plane +
                  static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                  static_cast<std::size_t>(x)];
}

bool load_rgb_image(const std::string& path, RgbImage* output, std::string* error) {
    if (output == nullptr) {
        set_error(error, "image output must not be null");
        return false;
    }
    const auto extension = lowercase_extension(path);
    if (extension == ".ppm" || extension == ".pnm") return load_ppm(path, output, error);
#if defined(CPP_LLM_HAS_LIBJPEG)
    if (extension == ".jpg" || extension == ".jpeg") return load_jpeg(path, output, error);
#else
    if (extension == ".jpg" || extension == ".jpeg") {
        set_error(error, "JPEG support is unavailable; install libjpeg development files and rebuild");
        return false;
    }
#endif
    set_error(error, "unsupported image extension: " + extension);
    return false;
}

bool decode_rgb_image(
    const std::vector<std::uint8_t>& encoded,
    const std::string& media_type,
    RgbImage* output,
    std::string* error) {
    if (output == nullptr) {
        set_error(error, "image output must not be null");
        return false;
    }
    if (encoded.empty()) {
        set_error(error, "encoded image must not be empty");
        return false;
    }
    if (media_type == "image/x-portable-pixmap" || media_type == "image/ppm") {
        const std::string bytes(
            reinterpret_cast<const char*>(encoded.data()), encoded.size());
        std::istringstream input(bytes, std::ios::binary);
        return decode_ppm(input, output, error);
    }
#if defined(CPP_LLM_HAS_LIBJPEG)
    if (media_type == "image/jpeg") return decode_jpeg(encoded, output, error);
#else
    if (media_type == "image/jpeg") {
        set_error(error, "JPEG support is unavailable; install libjpeg development files and rebuild");
        return false;
    }
#endif
    set_error(error, "unsupported image media type: " + media_type);
    return false;
}

bool jpeg_decoder_available() {
#if defined(CPP_LLM_HAS_LIBJPEG)
    return true;
#else
    return false;
#endif
}

}  // namespace multimodal

