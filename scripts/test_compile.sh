set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"


#nvcc -std=c++20 \
#    -G \
#    -g \
#    -expt-extended-lambda \
#    -I "$PROJECT_ROOT/src" \
#    "$PROJECT_ROOT/test/test_ver2.cu" \
#    -o "$BUILD_DIR/test_per" \
#    -lcurl \
#    -lcurand \
#    -lcudnn \
#    -Xcompiler="-pthread"
#echo "✅ build/test_per 빌드 완료"

nvcc -std=c++20 \
    -G \
    -g \
    -expt-extended-lambda \
    -I "$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/src/ver2/d_matrix_2.cu" \
    "$PROJECT_ROOT/test/test_conv_2.cu" \
    -o "$BUILD_DIR/test_con" \
    -lcurl \
    -lcurand \
    -lcudnn \
    -Xcompiler="-pthread"
echo "✅ build/test_con 빌드 완료"

nvcc -std=c++20 \
    -G \
    -g \
    -expt-extended-lambda \
    -I "$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/src/ver2/d_matrix_2.cu" \
    "$PROJECT_ROOT/src/ver2/GOLdatabase_2.cu" \
    "$PROJECT_ROOT/test/test_GOL_2.cu" \
    -o "$BUILD_DIR/test_gol" \
    -lcurand \
    -Xcompiler="-pthread"
echo "✅ build/test_gol 빌드 완료"


