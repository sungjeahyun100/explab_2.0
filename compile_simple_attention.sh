#!/bin/bash

# 간단한 Attention 테스트 컴파일 스크립트

echo "간단한 Attention 테스트 컴파일 시작..."

# CUDA 설정
CUDA_PATH="/usr/local/cuda"
NVCC="nvcc"
CUDA_FLAGS="-arch=sm_75 -std=c++20 -O2 --extended-lambda"

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
    echo "d_matrix_2.cu 컴파일 실패!"
    exit 1
fi

echo "simple_attention_test.cu 컴파일..."
${NVCC} ${CUDA_FLAGS} ${INCLUDE_PATHS} -c ${TEST_DIR}/simple_attention_test.cu -o ${BUILD_DIR}/simple_attention_test.o

if [ $? -ne 0 ]; then
    echo "simple_attention_test.cu 컴파일 실패!"
    exit 1
fi

echo "링킹..."
${NVCC} ${CUDA_FLAGS} ${LIB_PATHS} ${LIBS} \
    ${BUILD_DIR}/d_matrix_2.o \
    ${BUILD_DIR}/simple_attention_test.o \
    -o ${BUILD_DIR}/simple_attention_test

if [ $? -eq 0 ]; then
    echo "✅ 컴파일 성공!"
    echo "실행 파일: ${BUILD_DIR}/simple_attention_test"
    echo ""
    echo "실행 명령어:"
    echo "cd ${BUILD_DIR} && ./simple_attention_test"
    echo ""
    echo "바로 실행하려면:"
    echo "${BUILD_DIR}/simple_attention_test"
else
    echo "❌ 링킹 실패!"
    exit 1
fi
