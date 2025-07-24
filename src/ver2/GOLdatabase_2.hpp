#pragma once
#include <ver2/d_matrix_2.hpp>
#include <ver2/utility.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <random>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <deque>
#include <cstdlib>
#include <thrust/device_vector.h>

namespace GOL_2 {
    using namespace d_matrix_ver2;

    #define SAMPLE 2000

    extern const int BOARDWIDTH;
    extern const int BOARDHEIGHT;
    constexpr int BIT_WIDTH = 8;  // 예: 0~255 범위 표현용

    // Game of Life 다음 세대 계산 커널
    __global__ void nextGenKernel(int* current, int* next, int width, int height);

    // 다음 세대 계산 함수
    d_matrix_2<int> nextGen(const d_matrix_2<int>& current, cudaStream_t str = 0);

    // 살아있는 셀 개수 계산
    int countAlive(const d_matrix_2<int>& mat, cudaStream_t str = 0);

    // 고정 비율 패턴 생성 (패딩 포함)
    d_matrix_2<int> generateFixedRatioPatternWithPadding(
        int fullHeight, int fullWidth, 
        int patternHeight, int patternWidth, 
        double aliveRatio, cudaStream_t str = 0
    );

    // Game of Life 데이터 생성
    void generateGameOfLifeData(int filenum, double ratio);

    // 데이터 로딩
    std::vector<std::pair<d_matrix_2<double>, d_matrix_2<double>>> LoadingData();

    // 시뮬레이션 및 라벨링 (최종 패턴 반환)
    d_matrix_2<int> simulateAndLabelingtopattern(const d_matrix_2<int>& initialPattern, int fileId, cudaStream_t str = 0);

    // 시뮬레이션 및 라벨링 (살아있는 셀 개수 반환)
    int simulateAndLabel(const d_matrix_2<int>& initialPattern, int fileId, cudaStream_t str = 0);

    // 패턴 배치 커널
    __global__ void placePatternKernel(
        int* board, int* pattern, 
        int fullHeight, int fullWidth,
        int patternHeight, int patternWidth,
        int startRow, int startCol
    );

    // 살아있는 셀 카운트 커널
    __global__ void countAliveKernel(int* mat, int* partialSums, int totalSize);
}
