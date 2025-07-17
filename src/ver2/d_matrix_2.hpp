
#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <ostream>
#include <stdexcept>

#define CHECK_CUDA(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                throw std::runtime_error(cudaGetErrorString(err)); \
            } \
        } while (0)
    

namespace d_matrix_ver2{

    enum class InitType
    {
        He,
        Xavier,
        Uniform
    };
    // GPU 전용 행렬 클래스: 호스트-디바이스 자동 동기화 비활성화
    // 필요할 때 수동으로 cpytoHost()/cpytoDev() 호출

    
    
    template<typename T>
    __global__ void TransInKernel(T* d_A, T* d_C, int row, int col) {
        int x = blockIdx.x * blockDim.x + threadIdx.x; 
        int y = blockIdx.y * blockDim.y + threadIdx.y; 
    
        if (x < row && y < col) {
            d_C[y * row + x] = d_A[x*col+y];
        }
    }
    
    template<typename T>
    __global__ void rotateInKernel(T *d_A, T *d_C, int row, int col)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x; 
        int y = blockIdx.y * blockDim.y + threadIdx.y; 
    
        if (x < row && y < col) {
            int idx_C = x * col + y;
            int src_i = row - 1 - x;
            int src_j = col - 1 - y;
            int idx_A = src_i * col + src_j;
            d_C[idx_C] = d_A[idx_A];
        }
    }

    template<typename T>
    __global__ void getMatrix_Row(T* d_A, T* d_C, int row, int col, int where_r){//가로열 반환
        int x = blockIdx.x * blockDim.x + threadIdx.x; 
        int y = blockIdx.y * blockDim.y + threadIdx.y; 

        if(x < row && y < col) {
            int idx = x*col+y;
            if(x == where_r){
                d_C[y] = d_A[idx];
            }
        }
    }

    template<typename T>
    __global__ void getMatrix_Col(T* d_A, T* d_C, int row, int col, int where_c){//세로열 반환
        int x = blockIdx.x * blockDim.x + threadIdx.x; 
        int y = blockIdx.y * blockDim.y + threadIdx.y; 

        if(x < row && y < col) {
            int idx = x*col+y;
            if(y == where_c){
                d_C[x] = d_A[idx];
            }
        }
    }
    
    template<typename T>
    __global__ void reduceRows(const T* d_A, T* d_C, int rows, int cols) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < cols) {
            T sum = T(0);
            for (int i = 0; i < rows; ++i) {
                sum += d_A[i * cols + x];
            }
            d_C[x] = sum;
        }
    }
    template<typename T>
    class d_matrix_2 {
    private:
        T* d_data;
        std::vector<T> h_data;
        int row;
        int col;
        bool host_valid = true;
        bool dev_valid = true;
    public:
        d_matrix_2(int r, int c) : row(r), col(c) {
            if (row <= 0 || col <= 0) {
                std::cerr << "[ERROR] Invalid d_matrix_2 dimensions: " << row << "x" << col << std::endl;
                exit(1);
            }
            h_data.resize(r*c);
            CHECK_CUDA(cudaMalloc(&d_data, r*c*sizeof(T)));
            CHECK_CUDA(cudaMemset(d_data, 0, r*c*sizeof(T)));
        }
    
        d_matrix_2() : row(1), col(1){
            h_data.resize(row*col);
            CHECK_CUDA(cudaMalloc(&d_data, row*col*sizeof(T)));
            CHECK_CUDA(cudaMemset(d_data, 0, row*col*sizeof(T)));
        }
    
        //중괄호 생성자
        d_matrix_2(std::initializer_list<std::initializer_list<T>> list2d){
            row = static_cast<int>(list2d.size());
            col = row > 0 ? static_cast<int>(list2d.begin()->size()) : 0;
            if (row <= 0 || col <= 0) {
              std::cerr << "[ERROR] Invalid list2d dimensions: "
                        << row << "x" << col << std::endl;
              exit(1);
            }
        
            // 3) 메모리 할당
            int N = row * col;
            h_data.resize(N);
            size_t bytes = N * sizeof(T);
            CHECK_CUDA(cudaMalloc(&d_data, bytes));
        
            // 4) 값 복사
            int idx = 0;
            for (auto& Row : list2d) {
              if (static_cast<int>(Row.size()) != col) {
                std::cerr << "[ERROR] Inconsistent column size in list2d\n";
                exit(1);
              }
              for (auto& v : Row) {
                h_data[idx++] = v;
              }
            }
            // 5) 호스트 → 디바이스 복사
            CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
        }
    
        d_matrix_2(const d_matrix_2<T>& other) : row(other.row), col(other.col) {
            CHECK_CUDA(cudaMalloc(&d_data, row * col * sizeof(T)));
            h_data = other.h_data;
            CHECK_CUDA(cudaMemcpy(d_data, other.d_data, row * col * sizeof(T), cudaMemcpyDeviceToDevice));
        }
    
        d_matrix_2<T>& operator=(const d_matrix_2<T>& other) {
            if (this != &other) {
                cudaFree(d_data);
    
                row = other.row;
                col = other.col;
    
                CHECK_CUDA(cudaMalloc(&d_data, row * col * sizeof(T)));
                h_data = other.h_data;
                CHECK_CUDA(cudaMemcpy(d_data, other.d_data, row * col * sizeof(T), cudaMemcpyDeviceToDevice));
            }
            return *this;
        }

        d_matrix_2(d_matrix_2&& other) noexcept
          : row(other.row),
            col(other.col),
            d_data(other.d_data),
            h_data(std::move(other.h_data))
        {
            other.d_data = nullptr;
            other.row = other.col = 0;
        }
    
        // 이동 대입자
        d_matrix_2& operator=(d_matrix_2&& other) noexcept {
            if (this != &other) {
                if (d_data) cudaFree(d_data);
                row    = other.row;
                col    = other.col;
                d_data = other.d_data;
                h_data = std::move(other.h_data);
                other.d_data = nullptr;
                other.row = other.col = 0;
            }
            return *this;
        }
    
        bool operator==(const d_matrix_2<T>& other) const {
            if(other.col != col || other.row != row) return false;
            for(int i = 0; i < row*col; i++){
                if(this->h_data[i] != other.h_data[i]) return false;
            }
            return true;
        }
    
        bool operator!=(const d_matrix_2<T>& other) const {
            return !(*this == other);
        }
    
        T& operator()(int r, int c) {
            //if (!host_valid) throw std::runtime_error("Host data invalid; call toHost() first");
            return h_data[r * col + c];
        }
    
        // const 객체용
        const T& operator()(int r, int c) const {
            //if (!host_valid) throw std::runtime_error("Host data invalid; call toHost() first");
            return h_data[r * col + c];
        }

        /* 기존 소멸자
        ~d_matrix_2() noexcept {
            cudaError_t err = cudaFree(d_data);
            if (err != cudaSuccess) {
                std::cerr << "[CUDA ERROR in destructor] " << cudaGetErrorString(err) 
                          << " (\"~d_matrix_2()\")" << std::endl;
            }
        }
        */
        ~d_matrix_2() noexcept {
            if (d_data) {
                // cudaFree 가 에러를 반환해도 무시만 합니다.
                cudaError_t err = cudaFree(d_data);
                (void)err;
            }
        }
    
        T* getDevPointer() const { return d_data; }
        T* getHostPointer() {
            //if (!host_valid) throw std::runtime_error("Host data invalid; call toHost() first");
            return h_data.data();
        }
        const T* getHostPointer() const {
            //if (!host_valid) throw std::runtime_error("Host data invalid; call toHost() first");
            return h_data.data();
        }
        int getRow() const { return row; }
        int getCol() const { return col; }
        int size() const { return row*col; }
    
        void resize(int newRow, int newCol){
            // 1) 업데이트 순서: 먼저 멤버에 저장
            row = newRow;
            col = newCol;
            // 2) 메모리 재할당
            h_data.clear();
            CHECK_CUDA(cudaFree(d_data));
            h_data.resize(row * col);
            CHECK_CUDA(cudaMalloc(&d_data, row * col * sizeof(T)));
            // 호스트 동기화 플래그 초기화
            host_valid = false;
            dev_valid  = true;
        }
        void fill(T value) {
            for (int i = 0; i < row * col; ++i) {
                h_data[i] = value;
            }
            cpyToDev();
            host_valid = true;
        }
    
        void cpyToDev(){
            CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), row*col*sizeof(T), cudaMemcpyHostToDevice));
            dev_valid = true;
        }
        
        void cpyToHost(){
            CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, row*col*sizeof(T), cudaMemcpyDeviceToHost));
            host_valid = true;
        }
    
        void printMatrix() const {
            std::vector<T> host_data(row * col);
            CHECK_CUDA(cudaMemcpy(host_data.data(), d_data, row * col * sizeof(T), cudaMemcpyDeviceToHost));
    
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    std::cout << std::setw(4) << host_data[i * col + j] << " ";
                }
                std::cout << std::endl;
            }
        }
    
        //iterator
        T*       begin()       noexcept { return h_data.data(); }
        T*       end()         noexcept { return h_data.data() + row*col; }
        T const* begin() const noexcept { return h_data.data(); }
        T const* end()   const noexcept { return h_data.data() + row*col; }
    
        d_matrix_2<T> flatten(){
            d_matrix_2<T> V(*this);
            V.row = row*col;
            V.col = 1;
            return V;
        }
    
        d_matrix_2<T> reshape(int newR, int newC) const {
            if (static_cast<size_t>(newR) * newC != size())
                throw std::invalid_argument("reshape: size mismatch");
            d_matrix_2<T> V = *this;
            V.row = newR;
            V.col = newC;
            return V;
        }
    
        d_matrix_2<T> transpose() const {
            d_matrix_2<T> transposed(col, row);
            dim3 blockSize(32, 32);
            dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
            TransInKernel<<<gridSize, blockSize>>>(d_data, transposed.getDevPointer(), row, col);
            CHECK_CUDA(cudaDeviceSynchronize());
            return transposed;
        }
    
        d_matrix_2<T> rotated180() const {
            d_matrix_2<T> rotated(row, col);
            dim3 blockSize(32, 32);
            dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
            rotateInKernel<<<gridSize, blockSize>>>(d_data, rotated.getDevPointer(), row, col);
            CHECK_CUDA(cudaDeviceSynchronize());
            return rotated;
        }
    };
    
    
    template <typename T>
    std::ostream &operator<<(std::ostream &os, const d_matrix_2<T> &matrix)
    {
        os << matrix.getRow() << "x" << matrix.getCol() << "\n"; // 행렬 크기 출력
        for (int i = 0; i < matrix.getRow(); ++i) {
            for (int j = 0; j < matrix.getCol(); ++j) {
                os << matrix(i, j) << " "; // 행렬 요소 출력
            }
            os << "\n"; // 행 끝에서 줄바꿈
        }
        return os;
    }
    
    template<typename T>
    std::istream& operator>>(std::istream& is, d_matrix_2<T>& matrix) {
        int r, c; char sep;
        is >> r >> sep >> c;        // “2x2” 패턴 대응
        if (!is || sep!='x') {
            is.setstate(std::ios::failbit);
            return is;
        }
        matrix.resize(r, c);        // 멤버도 함께 갱신
        for (int i = 0; i < r; ++i) {
          for (int j = 0; j < c; ++j) {
            is >> matrix(i, j);
          }
        }
        matrix.cpyToDev();          // 안전하게 복사
        return is;
    }
    
    
    template<typename T>
    __global__ void zeroPad(T *d_A, T *d_C, int row, int col, int c_row, int c_col)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (x >= c_row || y >= c_col) return;
    
        // Calculate padding offsets
        int pad_row = (c_row - row) / 2;
        int pad_col = (c_col - col) / 2;
    
        // Flattened index for output
        int idx_C = x * c_col + y;
    
        // Check if inside the original matrix region
        if (x < pad_row || x >= pad_row + row ||
            y < pad_col || y >= pad_col + col) {
            // Outside input -> pad with zero
            d_C[idx_C] = T(0);
        } else {
            // Map to input index
            int src_i = x - pad_row;
            int src_j = y - pad_col;
            int idx_A = src_i * col + src_j;
            d_C[idx_C] = d_A[idx_A];
        }
    }
    
    template <typename T>
    d_matrix_2<T> zeroPedding(const d_matrix_2<T> &d_A, int size)
    {
        d_matrix_2<double> C(d_A.getRow()+(size*2), d_A.getCol()+(size*2));
        dim3 blockSize(32, 32);
        dim3 gridSize((C.getRow() + blockSize.x - 1) / blockSize.x, (C.getCol() + blockSize.y - 1) / blockSize.y);
        zeroPad<<<gridSize, blockSize>>>(d_A.getDevPointer(), C.getDevPointer(), d_A.getRow(), d_A.getCol(), C.getRow(), C.getCol());
        cudaDeviceSynchronize();
        return C;
    }
    
    template<typename T, int TILE>
    __global__ void matmul_tiled(const T* __restrict__ A,
                                 const T* __restrict__ B,
                                       T* __restrict__ C,
                                 int M, int N, int K) {
        // MxK * KxN = MxN
        // 블록 내 스레드 좌표
        int tx = threadIdx.x;
        int ty = threadIdx.y;
    
        // 전역 행·열 인덱스
        int row = blockIdx.y * TILE + ty;
        int col = blockIdx.x * TILE + tx;
    
        // Shared memory 공간 선언
        __shared__ T sA[TILE][TILE];
        __shared__ T sB[TILE][TILE];
    
        // 누적값 레지스터
        T sum = 0;
    
        // K 방향으로 타일 순회
        for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
            int A_col = t * TILE + tx;
            int B_row = t * TILE + ty;
    
            // 경계 검사 후 Shared 메모리에 로드
            sA[ty][tx] = (row < M && A_col < K)
                        ? A[row * K + A_col]
                        : T(0);
            sB[ty][tx] = (B_row < K && col < N)
                        ? B[B_row * N + col]
                        : T(0);
    
            __syncthreads();
    
            // 타일 내부 곱-누산
            #pragma unroll
            for (int k = 0; k < TILE; ++k) {
                sum += sA[ty][k] * sB[k][tx];
            }
            __syncthreads();
        }
    
        // 결과 쓰기
        if (row < M && col < N) {
            C[row * N + col] = sum;
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
    d_matrix_2<T> HadamardProduct(const d_matrix_2<T>& d_A, const d_matrix_2<T>& d_B) {
        int row = d_A.getRow();
        int col = d_A.getCol();
    
        d_matrix_2<T> C(row, col);
    
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
    
        HPinKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), row, col);
        cudaDeviceSynchronize();
    
        return C;
    }
    
    template<typename T>
    __global__ void ScalainKernel(T* d_A, T scalar, T* d_C, int row, int col) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (x < row && y < col) {
            d_C[x * col + y] = d_A[x * col + y] * scalar;
        }
    }
    
    template<typename T>
    d_matrix_2<T> ScalaProduct(const d_matrix_2<T>& d_A, T scalar) {
        int row = d_A.getRow();
        int col = d_A.getCol();
    
        d_matrix_2<T> C(row, col);
    
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
    
        ScalainKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), scalar, C.getDevPointer(), row, col);
        cudaDeviceSynchronize();
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
    
    constexpr int TILE = 32;
    
    template<typename T>
    d_matrix_2<T> matrixMP(const d_matrix_2<T>& A, const d_matrix_2<T>& B) {
        int M = A.getRow();
        int N = B.getCol();
        int K = A.getCol();  // A.cols == B.rows
    
        // 결과 행렬
        d_matrix_2<T> C(M, N);
    
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    
        // Tiled matmul 커널 호출
        matmul_tiled<T, TILE><<<grid, block>>>(A.getDevPointer(), B.getDevPointer(), C.getDevPointer(), M, N, K);
        cudaDeviceSynchronize();
    
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
    d_matrix_2<T> matrixPlus(const d_matrix_2<T>& d_A, const d_matrix_2<T>& d_B){//마이너스는 이렇게: d_matrix_2<double> C = matrixPlus(A, ScalaProduct(B, -1));
        int row = d_A.getRow();
        int col = d_A.getCol();
    
        d_matrix_2<T> C(row, col);
    
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
    
        PlusinKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), row, col);
        cudaDeviceSynchronize();
    
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
    d_matrix_2<T> MatrixActivate(const d_matrix_2<T>& d_A){
        int row = d_A.getRow();
        int col = d_A.getCol();
    
        d_matrix_2<T> C(row, col);
    
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
    
        ActivateInKernel<T, ActivateFunc><<<gridSize, blockSize>>>(d_A.getDevPointer(), C.getDevPointer(), row, col);
        cudaDeviceSynchronize();
    
        return C;
    }
    
    
    template<typename T>
    T plusAllElements(const d_matrix_2<T>& d_A){
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
    d_matrix_2<T> ScalaPlus(const d_matrix_2<T>& d_A, T scala){
        int row = d_A.getRow();
        int col = d_A.getCol();
    
        d_matrix_2<T> C(row, col);
        C = d_A;
    
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
    
        plusScalaToMatrix<<<gridSize, blockSize>>>(C.getDevPointer(), row, col, scala);
        cudaDeviceSynchronize();
    
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
    d_matrix_2<double> castToDoubleGPU(const d_matrix_2<T>& input) {
        d_matrix_2<double> output(input.getRow(), input.getCol());
    
        T* d_src = input.getDevPointer();
        double* d_dst = output.getDevPointer();
    
        int size = input.getRow()*input.getCol();
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
    
        castKernel<<<numBlocks, blockSize>>>(d_src, d_dst, size);
        cudaDeviceSynchronize();
    
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
    d_matrix_2<T> softmax(const d_matrix_2<T>& input) {
        int row = input.getRow();
        int col = input.getCol();
    
        d_matrix_2<T> output(row, col);
    
        int threads = 32;
        int blocks = (col + threads - 1) / threads;
    
        softmaxKernel<<<blocks, threads>>>(
            input.getDevPointer(),
            output.getDevPointer(),
            row, col
        );
    
        cudaDeviceSynchronize();
        return output;
    }
    
    template<typename T>
    __global__ void convertInKernel(T* d_X, int row, int col){
        double epsilon = 1e-10;
        int idx_x = blockDim.x*blockIdx.x+threadIdx.x;
        int idx_y = blockDim.y*blockIdx.y+threadIdx.y;
    
        if(idx_y < row && idx_x < col){
            if(d_X[idx_y*col+idx_x] < epsilon){
                d_X[idx_y*col+idx_x] = epsilon;
            }
        }
    }
    
    template<typename T>
    d_matrix_2<T> convertZeroToEpsilon(d_matrix_2<T> x){
        int row = x.getRow();
        int col = x.getCol();
    
        dim3 blockSize(32, 32);
        dim3 gridSize((col + blockSize.x - 1) / blockSize.x, (row + blockSize.y - 1) / blockSize.y);
    
        convertInKernel<<<gridSize, blockSize>>>(x.getDevPointer(), row, col);
        cudaDeviceSynchronize();
        return x;
    }
    
    template<typename T>
    __global__ void convoluteInKernel(T* __restrict__ d_A, T* __restrict__ d_B, T* __restrict__ d_C, int inputRow, int inputCol, int filterRow, int filterCol, int outputRow, int outputCol, int stride) {
        int idx_x = blockDim.x * blockIdx.x + threadIdx.x; // Output row index
        int idx_y = blockDim.y * blockIdx.y + threadIdx.y; // Output col index
    
        if (idx_x < outputRow && idx_y < outputCol) {
            T sum = 0;
            for (int i = 0; i < filterRow; i++) {
                for (int j = 0; j < filterCol; j++) {
                    int input_x = idx_x*stride + i;
                    int input_y = idx_y*stride + j;
                    if (input_x < inputRow && input_y < inputCol) {
                        sum += d_A[input_x * inputCol + input_y] * d_B[i * filterCol + j];
                    }
                }
            }
            d_C[idx_x * outputCol + idx_y] = sum;
        }
    }
    
    template<typename T>
    d_matrix_2<T> convolute(const d_matrix_2<T>& d_A, const d_matrix_2<T>& d_B, int stride) {
        int inputRow = d_A.getRow();
        int inputCol = d_A.getCol();
        int filterRow = d_B.getRow();
        int filterCol = d_B.getCol();
    
        // Calculate output dimensions
        int outputRow = ((inputRow - filterRow)/stride) + 1;
        int outputCol = ((inputCol - filterCol)/stride) + 1;
    
        // Initialize output matrix
        d_matrix_2<T> C(outputRow, outputCol);
    
        // Define CUDA grid and block sizes
        dim3 blockSize(32, 32);
        dim3 gridSize((outputRow + blockSize.x - 1) / blockSize.x, (outputCol + blockSize.y - 1) / blockSize.y);
    
        // Launch CUDA kernel
        convoluteInKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), inputRow, inputCol, filterRow, filterCol, outputRow, outputCol, stride);
        cudaDeviceSynchronize();
    
        return C;
    }
    
    // 1) cuRAND state initialization kernel
    __global__ void initCurandStates(curandState *states, unsigned long long seed, int total) {
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
    d_matrix_2<T> InitWeight(int row, int col, InitType type) {
        d_matrix_2<T> weight(row, col);
    
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
    
        // Free states
        cudaFree(d_states);
        return weight;
    }
    

}//namespace d_matrix_ver2

#include <functional>
namespace std {
    template<typename T>
    struct hash<d_matrix_ver2::d_matrix_2<T>> {
        size_t operator()(d_matrix_ver2::d_matrix_2<T> const& m) const noexcept {
            size_t h = m.getRow() ^ (m.getCol() << 1);
            for (auto& val : m) {
                h ^= std::hash<T>{}(val) + 0x9e3779b97f4a7c15 + (h<<6) + (h>>2);
            }
            return h;
        }
    };
}


