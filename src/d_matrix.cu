#include "d_matrix.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& os, const d_matrix<T>& matrix) {
    os << matrix.getRow() << "x" << matrix.getCol() << "\n"; // 행렬 크기 출력
    for (int i = 0; i < matrix.getRow(); ++i) {
        for (int j = 0; j < matrix.getCol(); ++j) {
            os << matrix(i, j) << " "; // 행렬 요소 출력
        }
        os << "\n"; // 행 끝에서 줄바꿈
    }
    return os;
}

template <typename T>
std::istream& operator>>(std::istream& is, d_matrix<T>& matrix) {
    int rows, cols;
    is >> rows >> cols; // 행렬 크기 읽기
    matrix.resize(rows, cols); // 행렬 크기 재설정

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            is >> matrix(i, j); // 행렬 요소 읽기
        }
    }
    matrix.cpyToDev(); // 호스트 메모리에서 장치 메모리로 복사
    return is;
}

template<typename T>
__global__ void TransInKernel(T* d_A, T* d_C, int row, int col) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // row index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // col index

    if (x < row && y < col) {
        d_C[x * col + y] = d_A[y * row + x]; // 올바른 인덱싱
    }
}

template<typename T>
__global__ void HPinKernel(T* d_A, T* d_B, T* d_C, int row, int col) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < row && y < col) {
        d_C[x * col + y] = d_A[x * col + y] * d_B[x * col + y];
    }
}

template<typename T>
d_matrix<T> HadamardProduct(const d_matrix<T>& d_A, const d_matrix<T>& d_B) {
    int row = d_A.getRow();
    int col = d_A.getCol();

    d_matrix<T> C(row, col);

    dim3 blockSize(32, 32);
    dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);

    HPinKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), row, col);
    cudaDeviceSynchronize();
    C.cpyToHost();

    return C;
}

template<typename T>
__global__ void ScalaKernel(T* d_A, T scalar, T* d_C, int row, int col) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < row && y < col) {
        d_C[x * col + y] = d_A[x * col + y] * scalar;
    }
}

template<typename T>
d_matrix<T> ScalaProduct(const d_matrix<T>& d_A, T scalar) {
    int row = d_A.getRow();
    int col = d_A.getCol();

    d_matrix<T> C(row, col);

    dim3 blockSize(32, 32);
    dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);

    ScalaKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), scalar, C.getDevPointer(), row, col);
    cudaDeviceSynchronize();

    C.cpyToHost();
    return C;
}

template<typename T>
__global__ void MPinKernel(T* d_A, T* d_B, T* d_C, int row, int col, int eq) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // row index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // col index

    if (x < row && y < col) {
        T sum = 0;
        for (int i = 0; i < eq; i++) {
            sum += d_A[x * eq + i] * d_B[i * col + y]; // 올바른 인덱싱
        }
        d_C[x * col + y] = sum;
    }
}

template<typename T>
d_matrix<T> matrixMP(const d_matrix<T>& d_A, const d_matrix<T>& d_B) {
    int row = d_A.getRow();
    int col = d_B.getCol();
    int eq = d_A.getCol(); // 두 행렬의 공통 차원 (A의 col == B의 row)

    d_matrix<T> C(row, col);

    dim3 blockSize(32, 32);
    dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);

    MPinKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), row, col, eq);
    cudaDeviceSynchronize();
    C.cpyToHost();
    
    return C;
}

template<typename T>
__global__ void PlusinKernel(T* d_A, T* d_B, T* d_C, int row, int col) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < row && y < col) {
        d_C[x * col + y] = d_A[x * col + y] + d_B[x * col + y]; 
    }
}

template<typename T>
d_matrix<T> matrixPlus(const d_matrix<T>& d_A, const d_matrix<T>& d_B){//마이너스는 이렇게: d_matrix<double> C = matrixPlus(A, ScalaProduct(B, -1));
    int row = d_A.getRow();
    int col = d_A.getCol();

    d_matrix<T> C(row, col);

    dim3 blockSize(32, 32);
    dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);

    PlusinKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), row, col);
    cudaDeviceSynchronize();
    C.cpyToHost();

    return C;
}

template<typename T>
__device__ T relu(T x) {
    return x > 0 ? x : 0;
}
 
template<typename T>
__device__ T d_relu(T x){
    return x > 0 ? 1 : 0;
}

template<typename T>
__device__ T lrelu(T x){
    return x > 0 ? x : 0.01*x;
}

template<typename T>
__device__ T d_lrelu(T x){
    return x > 0 ? 1 : 0.01;
}

const double alpha = 1.6732632423543772848170429916717l;
const double lamda = 1.0507009873554804934193349852946l;

template<typename T>
__device__ T ELU(T x){
    return x > 0 ? x : alpha*(exp(x) - 1);
}

template<typename T>
__device__ T d_ELU(T x){
    return x > 0 ? 1 : alpha*exp(x);
}

template<typename T>
__device__ T SELU(T x){
    return x > 0 ? lamda*x : alpha*lamda*(exp(x) - 1);
}

template<typename T>
__device__ T d_SELU(T x){
    return x > 0 ? lamda : alpha*lamda*exp(x);
}

template <typename T>
__device__ T Softplus(T x)
{
    if (x > T(0)) {
        return x + log1p(exp(-x));
    } else {
        return log1p(exp(x));
    }
}

template <typename T>
__device__ T Softsign(T x)
{
    return x/(T(1)+fabs(x));
}

template <typename T>
__device__ T d_Softsign(T x)
{
    T n = T(1)+fabs(x);
    return T(1)/(n*n);
}

template <typename T>
__device__ T Swish(T x)
{
    return x / (T(1) + exp(-x));
}

template <typename T>
__device__ T d_Swish(T x)
{
    T ex   = exp(x);
    T denom = ex + T(1);
    return ex * (x + ex + 1.0) / (denom * denom); 
}

template<typename T>
__device__ T Identity(T x){
    return x;
}

template<typename T>
__device__ T sigmoid(T x) {
    return T(1) / (T(1) + exp(-x));
}

template<typename T>
__device__ T d_sigmoid(T x) {
    T s = T(1) / (T(1) + exp(-x));
    return s * (1.0 - s);
}

template<typename T>
__device__ T d_I(T x) {
    return T(1);
}

template<typename T>
__device__ T Tanh(T x){
    return tanh(x);
}

template<typename T>
__device__ T d_tanh(T x){
    return (1-tanh(x))*(1+tanh(x));
}

template<typename T>
__device__ T sqr(T x){
    return sqrt(x);
}

template<typename T>
__device__ T devide(T x){
    return 1/x;
}

template<typename T>
__device__ T Log(T x){
    return log(x);
}

template<typename T, T (*ActivateFunc)(T)>
__global__ void ActivateInKernel(T* d_A, T* d_C, int row, int col){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < row && y < col) {
        d_C[x * col + y] = ActivateFunc(d_A[x * col + y]);
    }
}
 

template<typename T, T (*ActivateFunc)(T)>
d_matrix<T> MatrixActivate(const d_matrix<T>& d_A){
    int row = d_A.getRow();
    int col = d_A.getCol();

    d_matrix<T> C(row, col);

    dim3 blockSize(32, 32);
    dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);

    ActivateInKernel<T, ActivateFunc><<<gridSize, blockSize>>>(d_A.getDevPointer(), C.getDevPointer(), row, col);
    cudaDeviceSynchronize();
    C.cpyToHost();

    return C;
}


template<typename T>
T plusAllElements(const d_matrix<T>& d_A){
    T result = 0.0;
    int row = d_A.getRow();
    int col = d_A.getCol();
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            result += d_A(i, j);
        }
    }
    return result;
}

template<typename T>
__global__ void plusScalaToMatrix(T* d_C,int row, int col, T scala){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx_x < row && idx_y < col){
        d_C[idx_x * col + idx_y] += scala;
    }
}

template<typename T>
d_matrix<T> ScalaPlus(const d_matrix<T>& d_A, T scala){
    int row = d_A.getRow();
    int col = d_A.getCol();

    d_matrix<T> C(row, col);
    C = d_A;

    dim3 blockSize(32, 32);
    dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);

    plusScalaToMatrix<<<gridSize, blockSize>>>(C.getDevPointer(), row, col, scala);
    cudaDeviceSynchronize();
    C.cpyToHost();

    return C;
}


template <typename T>
__global__ void castKernel(const T* src, double* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = static_cast<double>(src[idx]);
    }
}

template <typename T>
d_matrix<double> castToDoubleGPU(const d_matrix<T>& input) {
    d_matrix<double> output(input.getRow(), input.getCol());

    T* d_src = input.getDevPointer();
    double* d_dst = output.getDevPointer();

    int size = input.getRow()*input.getCol();
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    castKernel<<<numBlocks, blockSize>>>(d_src, d_dst, size);
    cudaDeviceSynchronize();
    output.cpyToHost();

    return output;
}

template<typename T>
__global__ void softmaxKernel(T* in, T* out, int row, int col) {
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (colIdx >= col) return;

    // 1. max
    double max_val = in[colIdx];
    for (int i = 1; i < row; ++i) {
        double val = in[i * col + colIdx];
        if (val > max_val) max_val = val;
    }

    // 2. 분자와 분모
    double sum = 0.0;
    for (int i = 0; i < row; ++i) {
        sum += exp(in[i * col + colIdx] - max_val);
    }

    // 3. 결과 저장
    for (int i = 0; i < row; ++i) {
        out[i * col + colIdx] = exp(in[i * col + colIdx] - max_val) / sum;
    }
}

template<typename T>
d_matrix<T> softmax(const d_matrix<T>& input) {
    int row = input.getRow();
    int col = input.getCol();

    d_matrix<T> output(row, col);

    int threads = 32;
    int blocks = (col + threads - 1) / threads;

    softmaxKernel<<<blocks, threads>>>(
        input.getDevPointer(),
        output.getDevPointer(),
        row, col
    );

    cudaDeviceSynchronize();
    output.cpyToHost();
    return output;
}

template<typename T>
__global__ void convertInKernel(T* d_X, int row, int col){
    int idx_x = blockDim.x*blockIdx.x+threadIdx.x;
    int idx_y = blockDim.y*blockIdx.y+threadIdx.y;

    if(idx_y < row && idx_x < col){
        if(d_X[idx_y*col+idx_x] < epsilon){
            d_X[idx_y*col+idx_x] = epsilon;
        }
    }
}

template<typename T>
d_matrix<T> convertZeroToEpsilon(d_matrix<T> x){
    int row = x.getRow();
    int col = x.getCol();

    dim3 blockSize(32, 32);
    dim3 gridSize((col + blockSize.x - 1) / blockSize.x, (row + blockSize.y - 1) / blockSize.y);

    convertInKernel<<<gridSize, blockSize>>>(x.getDevPointer(), row, col);
    cudaDeviceSynchronize();
    x.cpyToHost();
    return x;
}

template<typename T>
__global__ void convoluteInKernel(T* d_A, T* d_B, T* d_C, int inputRow, int inputCol, int filterRow, int filterCol, int outputRow, int outputCol) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x; // Output row index
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y; // Output col index

    if (idx_x < outputRow && idx_y < outputCol) {
        T sum = 0;
        for (int i = 0; i < filterRow; i++) {
            for (int j = 0; j < filterCol; j++) {
                int input_x = idx_x + i;
                int input_y = idx_y + j;
                if (input_x < inputRow && input_y < inputCol) {
                    sum += d_A[input_x * inputCol + input_y] * d_B[i * filterCol + j];
                }
            }
        }
        d_C[idx_x * outputCol + idx_y] = sum;
    }
}

template<typename T>
d_matrix<T> convolute(const d_matrix<T>& d_A, const d_matrix<T>& d_B) {
    int inputRow = d_A.getRow();
    int inputCol = d_A.getCol();
    int filterRow = d_B.getRow();
    int filterCol = d_B.getCol();

    // Calculate output dimensions
    int outputRow = inputRow - filterRow + 1;
    int outputCol = inputCol - filterCol + 1;

    // Initialize output matrix
    d_matrix<T> C(outputRow, outputCol);

    // Define CUDA grid and block sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((outputRow + blockSize.x - 1) / blockSize.x, (outputCol + blockSize.y - 1) / blockSize.y);

    // Launch CUDA kernel
    convoluteInKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), inputRow, inputCol, filterRow, filterCol, outputRow, outputCol);
    cudaDeviceSynchronize();
    C.cpyToHost();

    return C;
}

// 1) cuRAND state initialization kernel
__global__ void initCurandStates(curandState *states,
                                 unsigned long long seed,
                                 int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// 2) Weight initialization kernel using cuRAND
template<typename T>
__global__ void InitWeightInKernel(
    T*               d_weight,
    curandState*     states,
    int              row,
    int              col,
    InitType         type
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= row || y >= col) return;

    int idx = x * col + y;
    curandState localState = states[idx];

    T val;
    switch (type) {
        case InitType::He: {
            val = curand_normal(&localState) * sqrtf(2.0f / row);
            break;
        }
        case InitType::Xavier: {
            val = curand_normal(&localState) * sqrtf(1.0f / row);
            break;
        }
        case InitType::Uniform: {
            val = (curand_uniform(&localState) * 2.0f) - 1.0f;
            break;
        }
    }
    d_weight[idx] = val;
    states[idx] = localState;
}

// 3) Host function to allocate states, run kernels, and return initialized matrix
template<typename T>
d_matrix<T> InitWeight(int row, int col, InitType type) {
    d_matrix<T> weight(row, col);

    // Allocate and initialize cuRAND states
    int total = row * col;
    curandState *d_states;
    cudaMalloc(&d_states, sizeof(curandState) * total);

    // Generate seed on CPU
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<unsigned long long> dist;
    unsigned long long seed = dist(mt);

    int threadsInit = 256;
    int blocksInit = (total + threadsInit - 1) / threadsInit;
    initCurandStates<<<blocksInit, threadsInit>>>(d_states, seed, total);
    cudaDeviceSynchronize();

    // Launch weight init kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((row + blockSize.x - 1) / blockSize.x,
                  (col + blockSize.y - 1) / blockSize.y);
    InitWeightInKernel<<<gridSize, blockSize>>>(
        weight.getDevPointer(),
        d_states,
        row,
        col,
        type
    );
    cudaDeviceSynchronize();

    // Copy back to host
    weight.cpyToHost();

    // Free states
    cudaFree(d_states);
    return weight;
}

// Explicit template instantiations for double

template std::ostream& operator<< <double>(std::ostream&, const d_matrix<double>&);
template std::istream& operator>> <double>(std::istream&, d_matrix<double>&);
template d_matrix<double> HadamardProduct(const d_matrix<double>&, const d_matrix<double>&);
template d_matrix<double> ScalaProduct(const d_matrix<double>&, double);
template d_matrix<double> matrixMP(const d_matrix<double>&, const d_matrix<double>&);
template d_matrix<double> matrixPlus(const d_matrix<double>&, const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, relu<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, d_relu<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, lrelu<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, d_lrelu<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, Identity<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, sigmoid<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, d_sigmoid<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, d_I<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, Tanh<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, d_tanh<double>>(const d_matrix<double>&);

template d_matrix<double> MatrixActivate<double, ELU<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, d_ELU<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, SELU<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, d_SELU<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, Swish<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, d_Swish<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, Softsign<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, d_Softsign<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, Softplus<double>>(const d_matrix<double>&);

template d_matrix<double> MatrixActivate<double, sqr<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, devide<double>>(const d_matrix<double>&);
template d_matrix<double> MatrixActivate<double, Log<double>>(const d_matrix<double>&);
template double plusAllElements(const d_matrix<double>&);
template d_matrix<double> ScalaPlus(const d_matrix<double>&, double);
template d_matrix<double> castToDoubleGPU(const d_matrix<double>&);
template d_matrix<double> softmax(const d_matrix<double>&);
template d_matrix<double> convertZeroToEpsilon(d_matrix<double>);
template d_matrix<double> convolute(const d_matrix<double>&, const d_matrix<double>&);
template d_matrix<double> InitWeight(int, int, InitType);
template __global__ void TransInKernel<double>(double*, double*, int, int);
template __global__ void HPinKernel<double>(double*, double*, double*, int, int);
template __global__ void ScalaKernel<double>(double*, double, double*, int, int);
template __global__ void MPinKernel<double>(double*, double*, double*, int, int, int);
template __global__ void PlusinKernel<double>(double*, double*, double*, int, int);
template __global__ void ActivateInKernel<double, relu<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, d_relu<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, lrelu<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, d_lrelu<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, Identity<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, sigmoid<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, d_sigmoid<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, d_I<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, Tanh<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, d_tanh<double>>(double*, double*, int, int);

template __global__ void ActivateInKernel<double, ELU<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, d_ELU<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, SELU<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, d_SELU<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, Swish<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, d_Swish<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, Softsign<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, d_Softsign<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, Softplus<double>>(double*, double*, int, int);

template __global__ void ActivateInKernel<double, sqr<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, devide<double>>(double*, double*, int, int);
template __global__ void ActivateInKernel<double, Log<double>>(double*, double*, int, int);
template __global__ void plusScalaToMatrix<double>(double*, int, int, double);
template __global__ void castKernel<double>(const double*, double*, int);
template __global__ void softmaxKernel<double>(double*, double*, int, int);
template __global__ void convertInKernel<double>(double*, int, int);
template __global__ void convoluteInKernel<double>(double*, double*, double*, int, int, int, int, int, int);
template __global__ void InitWeightInKernel<double>(double*, curandState*, int, int, InitType);



