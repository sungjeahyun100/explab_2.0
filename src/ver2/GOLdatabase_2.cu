/**
 * Game of Life Database Generator - d_matrix_2 version
 * Conway's Game of Life 패턴 생성 및 라벨링을 위한 데이터베이스 생성기
 */

#include "GOLdatabase_2.hpp"

namespace GOL_2 {
    using namespace d_matrix_ver2;

    #define MAXGEN 2500
    namespace fs = std::filesystem;

    const std::string DATASET_PATH = "../dataset/";

    const int BOARDWIDTH = 100;
    const int BOARDHEIGHT = 100;
    const int WIDTH = 10;
    const int HEIGHT = 10;

    // CUDA kernel: Game of Life 다음 세대 계산
    __global__ void nextGenKernel(int* current, int* next, int width, int height) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < height && j < width) {
            int alive = 0;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if(dx == 0 && dy == 0) continue;
                    int ni = i + dx;
                    int nj = j + dy;
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        alive += current[ni * width + nj];
                    }
                }
            }

            int idx = i * width + j;
            if (current[idx] == 1) {
                next[idx] = (alive == 2 || alive == 3) ? 1 : 0;
            } else {
                next[idx] = (alive == 3) ? 1 : 0;
            }
        }
    }

    d_matrix_2<int> nextGen(const d_matrix_2<int>& current, cudaStream_t str) {
        d_matrix_2<int> next(current.getRow(), current.getCol(), str);
        int* d_curr = current.getDevPointer();
        int* d_next = next.getDevPointer();

        dim3 blockSize(32, 32);
        dim3 gridSize((current.getCol() + 31) / 32, (current.getRow() + 31) / 32);

        nextGenKernel<<<gridSize, blockSize, 0, str>>>(d_curr, d_next, current.getCol(), current.getRow());
        cudaStreamSynchronize(str);
        
        return next;
    }

    __global__ void placePatternKernel(int* board, int* pattern, int fullHeight, int fullWidth,
        int patternHeight, int patternWidth,
        int startRow, int startCol) {
        int i = blockIdx.y * blockDim.y + threadIdx.y; // pattern row
        int j = blockIdx.x * blockDim.x + threadIdx.x; // pattern col

        if (i < patternHeight && j < patternWidth) {
            int boardIdx = (startRow + i) * fullWidth + (startCol + j);
            int patternIdx = i * patternWidth + j;
            board[boardIdx] = pattern[patternIdx];
        }
    }

    d_matrix_2<int> generateFixedRatioPatternWithPadding(int fullHeight, int fullWidth, int patternHeight, int patternWidth, double aliveRatio, cudaStream_t str) {
        // 1. CPU에서 pattern 배열 셔플
        int totalPatternCells = patternHeight * patternWidth;
        int aliveCells = static_cast<int>(totalPatternCells * aliveRatio);
        std::vector<int> host_pattern(totalPatternCells, 0);
        std::fill_n(host_pattern.begin(), aliveCells, 1);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(host_pattern.begin(), host_pattern.end(), gen);

        // 2. GPU 메모리로 복사
        thrust::device_vector<int> d_pattern = host_pattern;
        d_matrix_2<int> board(fullHeight, fullWidth, str); // 전체 보드
        board.fill(0, str); // 0으로 초기화

        int startRow = (fullHeight - patternHeight) / 2;
        int startCol = (fullWidth - patternWidth) / 2;

        // 3. 커널로 중앙에 패턴 복사
        dim3 blockSize(16, 16);
        dim3 gridSize((patternWidth + 15) / 16, (patternHeight + 15) / 16);

        placePatternKernel<<<gridSize, blockSize, 0, str>>>(
            board.getDevPointer(), 
            thrust::raw_pointer_cast(d_pattern.data()), 
            fullHeight, fullWidth, 
            patternHeight, patternWidth, 
            startRow, startCol
        );

        cudaStreamSynchronize(str);
        return board;
    }

    __global__ void countAliveKernel(int* mat, int* partialSums, int totalSize) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int localSum = 0;

        for (int i = tid; i < totalSize; i += stride) {
            localSum += mat[i];
        }

        if (tid < totalSize) {
            partialSums[tid] = localSum;
        }
    }

    int countAlive(const d_matrix_2<int>& mat, cudaStream_t str) {
        int totalSize = mat.getRow() * mat.getCol();
        int threadsPerBlock = 256;
        int numBlocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;
        int totalThreads = threadsPerBlock * numBlocks;

        int* d_partialSums;
        cudaMallocAsync(&d_partialSums, sizeof(int) * totalThreads, str);

        countAliveKernel<<<numBlocks, threadsPerBlock, 0, str>>>(mat.getDevPointer(), d_partialSums, totalSize);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaStreamSynchronize(str);

        std::vector<int> partialSums(totalThreads);
        cudaMemcpyAsync(partialSums.data(), d_partialSums, sizeof(int) * totalThreads, cudaMemcpyDeviceToHost, str);
        cudaStreamSynchronize(str);

        int total = std::accumulate(partialSums.begin(), partialSums.end(), 0);
        cudaFreeAsync(d_partialSums, str);

        return total;
    }

    int simulateAndLabel(const d_matrix_2<int>& initialPattern, int fileId, cudaStream_t str) {
        d_matrix_2<int> sim = initialPattern;
        std::deque<int> history; // 최근 50개 alive 수 저장
        const int window = 50;

        int constantCount = 0;
        int prev = -1;
        bool strictlyIncreasing = true;
        int gen = 0;

        while (gen < MAXGEN) {
            int alive = countAlive(sim, str);

            // history 갱신
            if (history.size() >= window) history.pop_front();
            history.push_back(alive);

            if (prev == alive) constantCount++;
            else constantCount = 0;

            if (prev != -1 && alive <= prev) strictlyIncreasing = false;
            if (constantCount >= 100 || (strictlyIncreasing && gen >= 100)) break;

            prev = alive;
            sim = nextGen(sim, str);
            gen++;
        }

        return countAlive(sim, str);
    }

    d_matrix_2<int> simulateAndLabelingtopattern(const d_matrix_2<int>& initialPattern, int fileId, cudaStream_t str) {
        d_matrix_2<int> sim = initialPattern;
        std::deque<int> history; // 최근 50개 alive 수 저장
        const int window = 50;

        int constantCount = 0;
        int prev = -1;
        bool strictlyIncreasing = true;
        int gen = 0;

        while (gen < MAXGEN) {
            int alive = countAlive(sim, str);

            // history 갱신
            if (history.size() >= window) history.pop_front();
            history.push_back(alive);

            if (prev == alive) constantCount++;
            else constantCount = 0;

            if (prev != -1 && alive <= prev) strictlyIncreasing = false;
            if (constantCount >= 100 || (strictlyIncreasing && gen >= 100)) break;

            prev = alive;
            sim = nextGen(sim, str);
            gen++;
        }

        return sim;
    }

    void generateGameOfLifeData(int filenum, double ratio) {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            std::cerr << "[FATAL] No CUDA device: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
        cudaSetDevice(0);

        // 스트림 생성
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        fs::create_directories(DATASET_PATH);

        int totalFiles = filenum;
        double aliveratio = ratio;

        std::cout << "totalFiles:" << totalFiles << " (file direction: ../dataset)" << std::endl;
        std::cout << "aliveratio:" << aliveratio << std::endl;
        std::cout << "max generation:" << MAXGEN << std::endl;
        std::cout << "pattern size:" << HEIGHT << " * " << WIDTH << std::endl;
        std::cout << "board size:" << BOARDHEIGHT << " * " << BOARDWIDTH << std::endl;

        const char *command1 = "find ../dataset/ -type f -delete";
        std::system(command1);

        auto startTime = std::chrono::steady_clock::now();

        for (int fileId = 1; fileId <= totalFiles; ++fileId) {
            int label = -1;
            d_matrix_2<int> pattern = generateFixedRatioPatternWithPadding(BOARDHEIGHT, BOARDWIDTH, HEIGHT, WIDTH, aliveratio, stream);
            d_matrix_2<int> last_pattern = simulateAndLabelingtopattern(pattern, fileId, stream);
            label = simulateAndLabel(pattern, fileId, stream);

            std::ofstream fout(DATASET_PATH + "sample" + std::to_string(fileId) + ".txt");

            int startRow = (BOARDHEIGHT - HEIGHT) / 2;
            int startCol = (BOARDWIDTH - WIDTH) / 2;

            // 초기 패턴을 호스트로 복사
            pattern.cpyToHost();

            // 초기 패턴 저장
            for (int i = startRow; i < startRow + HEIGHT; ++i) {
                for (int j = startCol; j < startCol + WIDTH; ++j) {
                    fout << pattern(i, j);
                }
                fout << '\n';
            }

            fout << label << '\n';
            fout << '\n';

            // 최종 패턴을 호스트로 복사
            last_pattern.cpyToHost();

            // 최종 패턴 저장
            for(int i = 0; i < BOARDHEIGHT; i++){
                for(int j = 0; j < BOARDWIDTH; j++){
                    fout << last_pattern(i, j);
                }
                fout << '\n';
            }

            fout.close();
            printProgressBar(fileId, totalFiles, startTime, "");
        }
        
        std::cout << std::endl << "[Done] Dataset generation complete." << std::endl;

        auto totalElapsed = std::chrono::steady_clock::now() - startTime;
        int totalSec = std::chrono::duration_cast<std::chrono::seconds>(totalElapsed).count();
        std::cout << "총 실행 시간: " << totalSec << " 초" << std::endl;

        cudaStreamDestroy(stream);
    }

    std::vector<std::pair<d_matrix_2<double>, d_matrix_2<double>>> LoadingData() {
        std::vector<std::pair<d_matrix_2<double>, d_matrix_2<double>>> dataset;
        dataset.reserve(1000);

        // 스트림 생성
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        for (const auto& entry : fs::directory_iterator(DATASET_PATH)) {
            if (entry.path().extension() != ".txt") continue;

            std::ifstream fin(entry.path());
            if (!fin) {
                std::cerr << "파일 열기 실패: " << entry.path() << '\n';
                continue;
            }

            d_matrix_2<double> input(WIDTH*HEIGHT, 1, stream);
            std::string line;
            int row = 0;
            while (row < WIDTH && std::getline(fin, line)) {
                int len = std::min(HEIGHT, static_cast<int>(line.size()));
                for (int col = 0; col < len; ++col) {
                    input(row * HEIGHT + col, 0) = line[col] - '0';
                }
                row++;
            }

            int label_index = -1;
            if (std::getline(fin, line)) label_index = std::stoi(line);

            d_matrix_2<double> label(BIT_WIDTH, 1, stream);
            // 1) 모두 0으로 초기화
            label.fill(0.0, stream);
            // 2) 각 비트 위치에 0/1 설정 (LSB부터)
            for (int b = 0; b < BIT_WIDTH; ++b) {
                label(b, 0) = (label_index >> b) & 1;
            }

            input.cpyToDev();
            label.cpyToDev();
            dataset.emplace_back(std::move(input), std::move(label));
        }

        cudaStreamDestroy(stream);
        return dataset;
    }

} // namespace GOL_2
