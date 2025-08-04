#!/bin/bash

echo "🔨 간단한 d_matrix_2 분리 테스트 컴파일 중..."

nvcc -std=c++20 --extended-lambda -rdc=true \
     -I./src -I./include \
     -lcudnn -lcurand -lcublas -lcusolver \
     -O3 -Xcompiler -fPIC \
     -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     ./test/simple_test.cu \
     ./src/ver2/d_matrix_2.cu \
     -o ./build/simple_test

if [ $? -eq 0 ]; then
    echo "✅ 컴파일 성공!"
    echo "🚀 테스트 실행 중..."
    ./build/simple_test
else
    echo "❌ 컴파일 실패!"
    exit 1
fi
