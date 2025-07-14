set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"


nvcc -std=c++20 \
    -G \
    -g \
    -I "$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/test/test_ver2.cu" \
    -o "$BUILD_DIR/testCon" \
    -lcurl \
    -lcurand \
    -lcudnn \
    -Xcompiler="-pthread"
echo "✅ build/testCon 빌드 완료"



