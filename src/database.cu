/**나중에 printProgressBar 단계 나눠서 출력시키는 코드 추가시켜야 할듯. 약간 게임 로딩바처럼. */

#include "database.hpp"

#define MAXGEN 2500
namespace fs = std::filesystem;

const std::string DATASET_PATH = "../dataset/";

const int BOARDWIDTH = 100;
const int BOARDHEIGHT = 100;
const int WIDTH = 10;
const int HEIGHT = 10;

// CUDA kernel 그대로 활용 (width, height 파라미터 수정)
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

d_matrix<int> nextGen(const d_matrix<int>& current) {
    d_matrix<int> next(current.getRow(), current.getCol());
    int* d_curr = current.getDevPointer();
    int* d_next = next.getDevPointer();

    dim3 blockSize(32, 32);
    dim3 gridSize((current.getCol() + 31) / 32, (current.getRow() + 31) / 32);

    nextGenKernel<<<gridSize, blockSize>>>(d_curr, d_next, current.getCol(), current.getRow());
    cudaDeviceSynchronize();
    next.cpyToHost();
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

d_matrix<int> generateFixedRatioPatternWithPadding(int fullHeight, int fullWidth, int patternHeight, int patternWidth, double aliveRatio) {
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
    d_matrix<int> board(fullHeight, fullWidth); // 전체 보드
    board.fill(0); // 0으로 초기화

    int startRow = (fullHeight - patternHeight) / 2;
    int startCol = (fullWidth - patternWidth) / 2;

// 3. 커널로 중앙에 패턴 복사
    dim3 blockSize(16, 16);
    dim3 gridSize((patternWidth + 15) / 16, (patternHeight + 15) / 16);

    placePatternKernel<<<gridSize, blockSize>>>(board.getDevPointer(), thrust::raw_pointer_cast(d_pattern.data()), fullHeight, fullWidth, patternHeight, patternWidth, startRow, startCol);

    cudaDeviceSynchronize();
    board.cpyToHost(); // 필요 시 host로 복사

    return board;
}



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



int countAlive(const d_matrix<int>& mat) {
    int totalSize = mat.getRow() * mat.getCol();
    int threadsPerBlock = 256;
    int numBlocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;
    int totalThreads = threadsPerBlock * numBlocks;

    int* d_partialSums;
    cudaMalloc(&d_partialSums, sizeof(int) * totalThreads);

    countAliveKernel<<<numBlocks, threadsPerBlock>>>(mat.getDevPointer(), d_partialSums, totalSize);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    std::vector<int> partialSums(totalThreads);
    cudaMemcpy(partialSums.data(), d_partialSums, sizeof(int) * totalThreads, cudaMemcpyDeviceToHost);

    int total = std::accumulate(partialSums.begin(), partialSums.end(), 0);
    cudaFree(d_partialSums);

    return total;
}



int simulateAndLabel(const d_matrix<int>& initialPattern, int fileId) {
    d_matrix<int> sim = initialPattern;
    std::deque<int> history; // 최근 50개 alive 수 저장
    const int window = 50;

    int constantCount = 0;
    int prev = -1;
    bool strictlyIncreasing = true;
    int gen = 0;

    while (gen < MAXGEN) {
        int alive = countAlive(sim);

        // history 갱신
        if (history.size() >= window) history.pop_front();
        history.push_back(alive);

        if (prev == alive) constantCount++;
        else constantCount = 0;

        if (prev != -1 && alive <= prev) strictlyIncreasing = false;
        if (constantCount >= 100 || (strictlyIncreasing && gen >= 100)) break;

        prev = alive;
        sim.cpyToDev();
        sim = nextGen(sim);
        gen++;
    }

    return countAlive(sim);
}

d_matrix<int> simulateAndLabelingtopattern(const d_matrix<int>& initialPattern, int fileId){
    d_matrix<int> sim = initialPattern;
    std::deque<int> history; // 최근 50개 alive 수 저장
    const int window = 50;

    int constantCount = 0;
    int prev = -1;
    bool strictlyIncreasing = true;
    int gen = 0;

    while (gen < MAXGEN) {
        int alive = countAlive(sim);

        // history 갱신
        if (history.size() >= window) history.pop_front();
        history.push_back(alive);

        if (prev == alive) constantCount++;
        else constantCount = 0;

        if (prev != -1 && alive <= prev) strictlyIncreasing = false;
        if (constantCount >= 100 || (strictlyIncreasing && gen >= 100)) break;

        prev = alive;
        sim.cpyToDev();
        sim = nextGen(sim);
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


    fs::create_directories(DATASET_PATH);

    int totalFiles = filenum; // 변경 가능
    double aliveratio = ratio;

    std::cout << "totalFlies:" << totalFiles << "(flie direction: ../dataset)" << std::endl;
    std::cout << "aliveratio:" << aliveratio << std::endl;
    std::cout << "max genaration:" << MAXGEN << std::endl;
    std::cout << "pattern size:" << HEIGHT << " * " << WIDTH << std::endl;
    std::cout << "board size:" << BOARDHEIGHT << " * " << BOARDWIDTH << std::endl;

    const char *commend1 = "find ../dataset/ -type f -delete";//DATASET_PATH 부분,

    std::system(commend1);

    auto startTime = std::chrono::steady_clock::now();

    d_matrix<int> pattern(BOARDHEIGHT, BOARDWIDTH);

    for (int fileId = 1; fileId <= totalFiles; ++fileId) {
        int label = -1;
        d_matrix<int> last_pattern(BOARDHEIGHT, BOARDWIDTH);
        pattern = generateFixedRatioPatternWithPadding(BOARDHEIGHT, BOARDWIDTH, HEIGHT, WIDTH, aliveratio);
        last_pattern = simulateAndLabelingtopattern(pattern, fileId);
        label = simulateAndLabel(pattern, fileId);

        std::ofstream fout(DATASET_PATH + "sample" + std::to_string(fileId) + ".txt");

        int startRow = (BOARDHEIGHT - HEIGHT) / 2;
        int startCol = (BOARDWIDTH - WIDTH) / 2;


        for (int i = startRow; i < startRow + HEIGHT; ++i) {
            for (int j = startCol; j < startCol + WIDTH; ++j) {
                fout << pattern(i, j);
            }
            fout << '\n';
        }

        fout << label << '\n';
        fout << '\n';

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
}



std::vector<std::pair<d_matrix<double>, d_matrix<double>>> LoadingData() {
    std::vector<std::pair<d_matrix<double>, d_matrix<double>>> dataset;
    dataset.reserve(1000);

    for (const auto& entry : fs::directory_iterator(DATASET_PATH)) {
        if (entry.path().extension() != ".txt") continue;

        std::ifstream fin(entry.path());
        if (!fin) {
            std::cerr << "파일 열기 실패: " << entry.path() << '\n';
            continue;
        }

        d_matrix<double> input(WIDTH*HEIGHT, 1);
        std::string line;
        int row = 0;
        while (row < WIDTH && std::getline(fin, line)) {
            int len = std::min(HEIGHT, static_cast<int>(line.size()));
            for (int col = 0; col < len; ++col)
                input(row * HEIGHT + col, 0) = line[col] - '0';
            row++;
        }

        int label_index = -1;
        if (std::getline(fin, line)) label_index = std::stoi(line);

        d_matrix<double> label(BIT_WIDTH, 1);
        // 1) 모두 0으로 초기화
        label.fill(0.0);
        // 2) 각 비트 위치에 0/1 설정 (LSB부터)
        for (int b = 0; b < BIT_WIDTH; ++b) {
            label(b, 0) = (label_index >> b) & 1;
        }
        // --------------------------------

        input.cpyToDev();
        label.cpyToDev();
        dataset.emplace_back(input, label);
    }

    return dataset;
}


