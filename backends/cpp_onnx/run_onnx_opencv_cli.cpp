#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool read_f32(const std::string &path, std::vector<float> &v) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    return static_cast<size_t>(f.gcount()) == v.size() * sizeof(float);
}

static bool write_u8(const std::string &path, const std::vector<uint8_t> &v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size()));
    return f.good();
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " MODEL.onnx INPUT_F32.bin OUTPUT_U8.bin\n";
        return 2;
    }

    constexpr int N = 1, C = 13, H = 512, W = 512, K = 4;
    std::vector<float> input(static_cast<size_t>(N) * C * H * W);
    if (!read_f32(argv[2], input)) {
        std::cerr << "Failed to read input: " << argv[2] << "\n";
        return 3;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(argv[1]);
    int dims[4] = {N, C, H, W};
    cv::Mat blob(4, dims, CV_32F, input.data());
    net.setInput(blob);
    cv::Mat logits = net.forward();

    if (logits.dims != 4 || logits.size[0] != N || logits.size[1] != K ||
        logits.size[2] != H || logits.size[3] != W) {
        std::cerr << "Unexpected output dimensions\n";
        return 4;
    }

    const float *out = reinterpret_cast<const float *>(logits.data);
    std::vector<uint8_t> pred(static_cast<size_t>(H) * W);
    for (int p = 0; p < H * W; ++p) {
        int best = 0;
        float best_v = out[p];
        for (int cls = 1; cls < K; ++cls) {
            const float v = out[cls * H * W + p];
            if (v > best_v) {
                best_v = v;
                best = cls;
            }
        }
        pred[p] = static_cast<uint8_t>(best);
    }

    if (!write_u8(argv[3], pred)) {
        std::cerr << "Failed to write output: " << argv[3] << "\n";
        return 5;
    }
    return 0;
}
