#include <ver2/GOLdatabase_2.hpp>
#include <iostream>

using namespace GOL_2;

int main() {
    std::cout << "=== Game of Life Database Generator v2 ===" << std::endl;

    try {
        // CUDA 디바이스 확인
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            std::cerr << "[FATAL] No CUDA device: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        std::cout << "CUDA devices found: " << deviceCount << std::endl;

        // 작은 테스트 데이터 생성 (10개 파일, 30% 생존률)
        std::cout << "\n=== Generating Test Dataset ===" << std::endl;
        generateGameOfLifeData(10, 0.3);

        // 생성된 데이터 로딩 테스트
        std::cout << "\n=== Loading Generated Data ===" << std::endl;
        auto dataset = LoadingData();
        std::cout << "Loaded " << dataset.size() << " samples successfully!" << std::endl;

        // 첫 번째 샘플 정보 출력
        if (!dataset.empty()) {
            auto& first_sample = dataset[0];
            std::cout << "First sample - Input size: " << first_sample.first.getRow() 
                      << "x" << first_sample.first.getCol() << std::endl;
            std::cout << "First sample - Label size: " << first_sample.second.getRow() 
                      << "x" << first_sample.second.getCol() << std::endl;
        }

        std::cout << "\n=== Test Completed Successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
