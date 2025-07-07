#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <iostream>

// path 에 해당하는 파일을 읽어서, 헤더(이미지:4개, 레이블:2개)의
// 각 4바이트 정수를 little↔big 엔디언으로 swap 하고,
// 나머지 데이터는 그대로 덮어씁니다.
void swapHeaderEndianInPlace(const std::string& path) {
    // 1) 원본 파일 읽기
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Error: cannot open for reading: " << path << "\n";
        return;
    }
    std::vector<char> buf = std::vector<char>(
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>()
    );
    in.close();

    // 2) 헤더에 몇 개의 32-bit 정수가 있는지 결정
    int numInts = 0;
    if (path.find("images-idx3-ubyte") != std::string::npos) {
        numInts = 4;   // magic, nImages, rows, cols
    }
    else if (path.find("labels-idx1-ubyte") != std::string::npos) {
        numInts = 2;   // magic, nLabels
    } else {
        std::cerr << "Warning: unknown file type, skipping: " << path << "\n";
        return;
    }

    // 3) 각 4바이트마다 바이트 순서 교환 (0↔3, 1↔2)
    for (int k = 0; k < numInts; ++k) {
        size_t i = size_t(k) * 4;
        if (i + 3 >= buf.size()) break;
        std::swap(buf[i + 0], buf[i + 3]);
        std::swap(buf[i + 1], buf[i + 2]);
    }

    // 4) 같은 파일에 덮어쓰기
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "Error: cannot open for writing: " << path << "\n";
        return;
    }
    out.write(buf.data(), buf.size());
    out.close();

    std::cout << "Endian-swapped header: " << path 
              << " (" << numInts << " ints)\n";
}

int main() {
    // 처리할 MNIST idx 파일 목록
    std::vector<std::string> files = {
        "../test/t10k-images-idx3-ubyte",
        "../test/train-images-idx3-ubyte",
        "../test/train-labels-idx1-ubyte",
        "../test/t10k-labels-idx1-ubyte"
    };

    for (auto& f : files) {
        swapHeaderEndianInPlace(f);
    }
    return 0;
}

