#!/bin/bash

# Attention 메커니즘 컴파일 스크립트

echo "Attention 메커니즘 컴파일 시작..."

# CUDA 및 cuDNN 설정
CUDA_PATH="/usr/local/cuda"
CUDNN_PATH="/usr/local/cuda"

# 컴파일러 설정
NVCC="nvcc"
CXX_FLAGS="-std=c++20 -O3"
CUDA_FLAGS="-arch=sm_75 -std=c++20 -O3 -rdc=true --extended-lambda"

# 인클루드 경로
INCLUDE_PATHS="-I./src -I${CUDA_PATH}/include -I${CUDNN_PATH}/include"

# 라이브러리 경로  
LIB_PATHS="-L${CUDA_PATH}/lib64 -L${CUDNN_PATH}/lib64"

# 링크할 라이브러리
LIBS="-lcudnn -lcublas -lcurand -lcusolver"

# 소스 파일들
SRC_DIR="./src/ver2"
EXAMPLE_DIR="./example_code"
BUILD_DIR="./build"

# 빌드 디렉토리 생성
mkdir -p ${BUILD_DIR}

echo "d_matrix_2.cu 컴파일 중..."
${NVCC} ${CUDA_FLAGS} ${INCLUDE_PATHS} -c ${SRC_DIR}/d_matrix_2.cu -o ${BUILD_DIR}/d_matrix_2.o

echo "attention.cu 컴파일 중..."
${NVCC} ${CUDA_FLAGS} ${INCLUDE_PATHS} -c ${SRC_DIR}/attention.cu -o ${BUILD_DIR}/attention.o

echo "attention_example.cu 컴파일 중..."
${NVCC} ${CUDA_FLAGS} ${INCLUDE_PATHS} -c ${EXAMPLE_DIR}/attention_example.cu -o ${BUILD_DIR}/attention_example.o

echo "최종 실행 파일 링킹..."
${NVCC} ${CUDA_FLAGS} ${LIB_PATHS} ${LIBS} \
    ${BUILD_DIR}/d_matrix_2.o \
    ${BUILD_DIR}/attention.o \
    ${BUILD_DIR}/attention_example.o \
    -o ${BUILD_DIR}/attention_test

if [ $? -eq 0 ]; then
    echo "컴파일 성공! 실행 파일: ${BUILD_DIR}/attention_test"
    echo ""
    echo "실행 명령어:"
    echo "cd ${BUILD_DIR} && ./attention_test"
else
    echo "컴파일 실패!"
    exit 1
fi
