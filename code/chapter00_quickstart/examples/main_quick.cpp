// Chapter 00：提供不依赖推理库的环境与模型路径 smoke check。
// 职责：验证 quickstart 可执行程序和可选 GGUF 文件是否可访问。
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* program) {
    std::cout << "Usage:\n"
              << "  " << program << " [--model <path>] [--check]\n\n"
              << "Options:\n"
              << "  --model <path>  Optional model file to verify existence and size.\n"
              << "  --check         Run the quickstart smoke check and exit.\n";
}

bool verify_model_path(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        std::cerr << "[quickstart] model file not found: " << path << "\n";
        return false;
    }

    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "[quickstart] model path is not a file: " << path << "\n";
        return false;
    }

    const auto size = std::filesystem::file_size(path);
    std::cout << "[quickstart] model file exists: " << path << "\n";
    std::cout << "[quickstart] model size: " << size << " bytes\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path model_path;
    bool check_only = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--check") {
            check_only = true;
            continue;
        }
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
            continue;
        }

        std::cerr << "[quickstart] unknown argument: " << arg << "\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "[quickstart] chapter00 quickstart smoke check\n";
    std::cout << "[quickstart] current working directory: "
              << std::filesystem::current_path() << "\n";

    if (!model_path.empty()) {
        if (!verify_model_path(model_path)) {
            return 1;
        }
    } else {
        std::cout << "[quickstart] no model path provided\n";
        std::cout << "[quickstart] this is enough to confirm the executable path works\n";
    }

    if (check_only) {
        std::cout << "[quickstart] smoke check passed\n";
        return 0;
    }

    std::cout << "[quickstart] quickstart entry point is ready\n";
    std::cout << "[quickstart] next step is to run llama.cpp with a real model\n";
    return 0;
}
