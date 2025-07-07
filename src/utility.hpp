#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "d_matrix.hpp"

void printProgressBar(int current, int total, std::chrono::steady_clock::time_point startTime, std::string processname) {
    int width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(width * progress);
    
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    int elapsedSec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% ";
    std::cout << '[' << processname << ']';
    std::cout << "(경과 시간: " << elapsedSec << " 초)\r";
    std::cout.flush();
}

using mat = d_matrix<double>;

// 읽을 idx 파일의 32비트 정수를 빅엔디안으로 파싱
inline uint32_t read_be32(std::ifstream& in) {
    uint8_t b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("Unexpected EOF while reading header");
    return (uint32_t(b[0]) << 24) |
           (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) <<  8) |
           (uint32_t(b[3]) <<  0);
}

// MNIST 이미지 로더: ubyte 파일에서 행렬 벡터로 변환
// 픽셀 값은 [0,255] -> [0.0,1.0] 구간 double로 정규화
inline std::vector<mat> load_mnist_images(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open image file: " + path);

    uint32_t magic = read_be32(in);
    if (magic != 0x00000803) throw std::runtime_error("Invalid magic for images: " + std::to_string(magic));
    uint32_t N = read_be32(in);
    uint32_t rows = read_be32(in);
    uint32_t cols = read_be32(in);

    std::vector<mat> images;
    images.reserve(N);
    for (uint32_t i = 0; i < N; ++i) {
        mat m(rows, cols);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                uint8_t pixel;
                in.read(reinterpret_cast<char*>(&pixel), 1);
                if (!in) throw std::runtime_error("Unexpected EOF while reading pixels");
                m(r, c) = double(pixel) / 255.0;
            }
        }
        images.push_back(std::move(m));
    }
    return images;
}

// MNIST 레이블 로더: ubyte 파일에서 uint8_t 벡터로 변환
inline std::vector<uint8_t> load_mnist_labels(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open label file: " + path);

    uint32_t magic = read_be32(in);
    if (magic != 0x00000801) throw std::runtime_error("Invalid magic for labels: " + std::to_string(magic));
    uint32_t N = read_be32(in);

    std::vector<uint8_t> labels(N);
    in.read(reinterpret_cast<char*>(labels.data()), N);
    if (!in) throw std::runtime_error("Unexpected EOF while reading labels");
    return labels;
}


