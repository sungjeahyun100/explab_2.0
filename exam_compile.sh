set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"

nvcc -std=c++20 \
    -G \
    -g \
    -expt-extended-lambda \
    -I "$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/src/ver2/d_matrix_2.cu" \
    "$PROJECT_ROOT/src/ver2/GOLdatabase_2.cu" \
    "$PROJECT_ROOT/example_code/exp_model1.cu" \
    -o "$BUILD_DIR/GOLsolver" \
    -lcurl \
    -lcurand \
    -lcudnn \
    -Xcompiler="-pthread"
echo "✅ build/GOLsolver 빌드 완료"

nvcc -std=c++20 \
    -G \
    -g \
    -expt-extended-lambda \
    -I "$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/src/ver2/d_matrix_2.cu" \
    "$PROJECT_ROOT/src/ver2/GOLdatabase_2.cu" \
    "$PROJECT_ROOT/example_code/genGOL.cu" \
    -o "$BUILD_DIR/genGOL" \
    -lcurl \
    -lcurand \
    -lcudnn \
    -Xcompiler="-pthread"
echo "✅ build/genGOL 빌드 완료"

