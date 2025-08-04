#!/bin/bash

# GOL Attention 모델 컴파일 스크립트

echo "🎯 GOL Attention 모델 컴파일 시작..."

# CUDA 설정
CUDA_PATH="/usr/local/cuda"
NVCC="nvcc"
CUDA_FLAGS="-arch=sm_75 -std=c++20 -O2 --extended-lambda"

# 인클루드 경로
INCLUDE_PATHS="-I./src -I${CUDA_PATH}/include"

# 라이브러리 경로
LIB_PATHS="-L${CUDA_PATH}/lib64"

# 링크할 라이브러리
LIBS="-lcublas -lcurand -lcudnn"

# 디렉토리 설정
SRC_DIR="./src/ver2"
EXAMPLE_DIR="./example_code"
BUILD_DIR="./build"

# 빌드 디렉토리 생성
mkdir -p ${BUILD_DIR}

echo "📦 의존성 파일들 컴파일..."

echo "  - d_matrix_2.cu 컴파일..."
${NVCC} ${CUDA_FLAGS} ${INCLUDE_PATHS} -c ${SRC_DIR}/d_matrix_2.cu -o ${BUILD_DIR}/d_matrix_2.o

if [ $? -ne 0 ]; then
    echo "❌ d_matrix_2.cu 컴파일 실패!"
    exit 1
fi

echo "  - GOLdatabase_2.cu 컴파일..."
${NVCC} ${CUDA_FLAGS} ${INCLUDE_PATHS} -c ${SRC_DIR}/GOLdatabase_2.cu -o ${BUILD_DIR}/GOLdatabase_2.o

if [ $? -ne 0 ]; then
    echo "❌ GOLdatabase_2.cu 컴파일 실패!"
    exit 1
fi

echo "🧠 GOL Attention 모델 컴파일..."
${NVCC} ${CUDA_FLAGS} ${INCLUDE_PATHS} -c ${EXAMPLE_DIR}/gol_attention_model.cu -o ${BUILD_DIR}/gol_attention_model.o

if [ $? -ne 0 ]; then
    echo "❌ gol_attention_model.cu 컴파일 실패!"
    exit 1
fi

echo "🔗 링킹..."
${NVCC} ${CUDA_FLAGS} ${LIB_PATHS} ${LIBS} \
    ${BUILD_DIR}/d_matrix_2.o \
    ${BUILD_DIR}/GOLdatabase_2.o \
    ${BUILD_DIR}/gol_attention_model.o \
    -o ${BUILD_DIR}/gol_attention_model

if [ $? -eq 0 ]; then
    echo "✅ GOL Attention 모델 컴파일 성공!"
    echo "실행 파일: ${BUILD_DIR}/gol_attention_model"
    echo ""
    echo "실행 명령어:"
    echo "cd ${BUILD_DIR} && ./gol_attention_model"
    echo ""
    echo "바로 실행하려면:"
    echo "${BUILD_DIR}/gol_attention_model"
else
    echo "❌ 링킹 실패!"
    exit 1
fi
