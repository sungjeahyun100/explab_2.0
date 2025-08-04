#!/bin/bash

echo "🧪 매트릭스 함수 테스트 컴파일 시작..."

# CUDA 설정
CUDA_PATH="/usr/local/cuda"
NVCC="nvcc"
CUDA_FLAGS="-arch=sm_75 -std=c++20 -O2"

# 인클루드 경로
INCLUDE_PATHS="-I./src -I${CUDA_PATH}/include"

# 라이브러리 경로
LIB_PATHS="-L${CUDA_PATH}/lib64"

# 링크할 라이브러리
LIBS="-lcublas -lcurand"

# 디렉토리 설정
SRC_DIR="./src/ver2"
TEST_DIR="./test"
BUILD_DIR="./build"

# 빌드 디렉토리 생성
mkdir -p ${BUILD_DIR}

echo "d_matrix_2.cu 컴파일..."
${NVCC} ${CUDA_FLAGS} ${INCLUDE_PATHS} -c ${SRC_DIR}/d_matrix_2.cu -o ${BUILD_DIR}/d_matrix_2.o

if [ $? -ne 0 ]; then
    echo "❌ d_matrix_2.cu 컴파일 실패!"
    exit 1
fi

echo "test_matrix_functions.cu 컴파일..."
${NVCC} ${CUDA_FLAGS} ${INCLUDE_PATHS} -c ${TEST_DIR}/test_matrix_functions.cu -o ${BUILD_DIR}/test_matrix_functions.o

if [ $? -ne 0 ]; then
    echo "❌ test_matrix_functions.cu 컴파일 실패!"
    exit 1
fi

echo "링킹..."
${NVCC} ${CUDA_FLAGS} ${LIB_PATHS} ${LIBS} \
    ${BUILD_DIR}/d_matrix_2.o \
    ${BUILD_DIR}/test_matrix_functions.o \
    -o ${BUILD_DIR}/test_matrix_functions

if [ $? -eq 0 ]; then
    echo "✅ 컴파일 성공!"
    echo "실행 파일: ${BUILD_DIR}/test_matrix_functions"
    echo ""
    echo "바로 실행:"
    ${BUILD_DIR}/test_matrix_functions
else
    echo "❌ 링킹 실패!"
    exit 1
fi
