#!/bin/bash

echo "🔨 수정된 d_matrix_2 라이브러리 테스트 컴파일 중..."

nvcc -std=c++20 --extended-lambda -rdc=true \
     -I./src -I./include \
     -lcudnn -lcurand -lcublas -lcusolver \
     -O3 -Xcompiler -fPIC \
     -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     ./src/ver2/d_matrix_2.cu \
     ./test/test_matrix_functions.cu \
     -o ./build/test_matrix_functions

if [ $? -eq 0 ]; then
    echo "✅ 컴파일 성공! 실행 파일: ./build/test_matrix_functions"
else
    echo "❌ 컴파일 실패!"
    exit 1
fi
