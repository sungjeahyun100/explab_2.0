set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"

nvcc -std=c++20 \
    -I "$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/test/testConvolute.cu" \
    "$PROJECT_ROOT/src/d_matrix.cu" \
    "$PROJECT_ROOT/src/perceptron.cu" \
    -o "$BUILD_DIR/testCon" \
    -lcurl \
    -lcurand \
    -Xcompiler="-pthread"
echo "✅ build/testCon 빌드 완료"
