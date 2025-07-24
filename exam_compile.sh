set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"

nvcc -std=c++20 \
    -G \
    -g \
    -expt-extended-lambda \
    -I "$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/example_code/exp_model1.cu" \
    -o "$BUILD_DIR/GOLsolver" \
    -lcurl \
    -lcurand \
    -lcudnn \
    -Xcompiler="-pthread"
echo "✅ build/GOLsolver 빌드 완료"



