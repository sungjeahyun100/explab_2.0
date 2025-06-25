/**TODO: 
 * make TrancMatrix func / 2025/03/15 제작 완료
 * make matrixMultiplus(제작중) / 2025/03/15 제작 완료
 * make matrixplus(include minus) / 2025/03/16 제작 완료
 * make HP(그 이름이 뭐였더라 하여튼 그 같은 원소끼리 곱하는 그거) / 2025/03/15 제작 완료
 * make printMatrix / 2025/03/15 제작 완료
 */

/*
d_matrix<double> A(n, m);

원래 행렬 (Row-Major Order - 행 우선 저장)
--------------------------------------------
1  2
3  4
5  6
(n x m) 크기의 행렬 (3 x 2)

- 메모리 내부 저장 방식 (Row-Major Order)
[1, 2, 3, 4, 5, 6]  // 연속된 메모리에서 row-major로 저장됨

-T-> (전치 연산 후)

전치된 행렬 B (m x n)
--------------------------------------------
1  3  5
2  4  6
(2 x 3) 크기의 행렬

- 메모리 내부 저장 방식 (Row-Major Order)
[1, 3, 5, 2, 4, 6]  // 여전히 row-major로 저장되지만, 인덱싱 방식이 바뀜
*/


#pragma once
#include<cuda_runtime.h>
#include <curand_kernel.h>
#include<vector>
#include<string>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <random>

enum class InitType{
    He,
    Xavier,
    Uniform
};

const double epsilon = 1e-10;

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// English: CUDA kernel for transposing a matrix on the GPU.
// 한글: GPU에서 행렬을 전치하기 위한 CUDA 커널입니다.
template<typename T>
__global__ void TransInKernel(T* d_A, T* d_C, int row, int col);


template<typename T>
class d_matrix {
private:
    T* h_data;      // 호스트(메인 메모리) 데이터 포인터
    T* d_data;      // 디바이스(GPU) 데이터 포인터
    int rowSize;    // 행의 개수
    int colSize;    // 열의 개수
public:
    // 생성자: 행과 열 크기를 받고, 호스트와 디바이스용 메모리를 할당합니다.
    d_matrix(int row, int col) : rowSize(row), colSize(col) {
        if (rowSize <= 0 || colSize <= 0) {
            std::cerr << "[ERROR] Invalid d_matrix dimensions: " << rowSize << "x" << colSize << std::endl;
            exit(1);
        }
        // 호스트 메모리 할당
        h_data = new T[rowSize * colSize];
        size_t bytes = rowSize * colSize * sizeof(T);
        // 디바이스 메모리 할당
        CHECK_CUDA_ERROR(cudaMalloc(&d_data, bytes));
    }

    // 복사 생성자: 다른 d_matrix를 깊은 복사합니다.
    d_matrix(const d_matrix<T>& other) : rowSize(other.rowSize), colSize(other.colSize) {
        h_data = new T[rowSize * colSize];
        CHECK_CUDA_ERROR(cudaMalloc(&d_data, rowSize * colSize * sizeof(T)));
        std::copy(other.h_data, other.h_data + rowSize * colSize, h_data);
        CHECK_CUDA_ERROR(cudaMemcpy(d_data, other.d_data, rowSize * colSize * sizeof(T), cudaMemcpyDeviceToDevice));
    }

        d_matrix(std::initializer_list<std::initializer_list<T>> list2d){
        rowSize = static_cast<int>(list2d.size());
        colSize = rowSize > 0 ? static_cast<int>(list2d.begin()->size()) : 0;
        if (rowSize <= 0 || colSize <= 0) {
          std::cerr << "[ERROR] Invalid init2d dimensions: "
                    << rowSize << "x" << colSize << std::endl;
          exit(1);
        }
    
        // 3) 메모리 할당
        int N = rowSize * colSize;
        h_data = new T[N];
        size_t bytes = N * sizeof(T);
        CHECK_CUDA_ERROR(cudaMalloc(&d_data, bytes));
    
        // 4) 값 복사
        int idx = 0;
        for (auto& row : init2d) {
          if (static_cast<int>(row.size()) != colSize) {
            std::cerr << "[ERROR] Inconsistent column size in init2d\n";
            exit(1);
          }
          for (auto& v : row) {
            h_data[idx++] = v;
          }
        }
        // 5) 호스트 → 디바이스 복사
        CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    }
    
    d_matrix(std::initializer_list<T> list) : rowSize(1), colSize(static_cast<int>(list.size())){
        int N = rowSize * colSize;
        // 호스트 메모리 할당
        h_data = new T[N];
        // 디바이스 메모리 할당
        size_t bytes = N * sizeof(T);
        CHECK_CUDA_ERROR(cudaMalloc(&d_data, bytes));
    
        // 값 복사
        int i = 0;
        for (auto& v : list) {
          h_data[i++] = v;
        }
        // 호스트 → 디바이스
        CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    }

    // 복사 대입 연산자: 기존 리소스를 해제하고, 다른 행렬을 깊은 복사합니다.
    d_matrix<T>& operator=(const d_matrix<T>& other) {
        if (this != &other) {
            delete[] h_data;
            cudaFree(d_data);

            rowSize = other.rowSize;
            colSize = other.colSize;

            h_data = new T[rowSize * colSize];
            CHECK_CUDA_ERROR(cudaMalloc(&d_data, rowSize * colSize * sizeof(T)));
            std::copy(other.h_data, other.h_data + rowSize * colSize, h_data);
            CHECK_CUDA_ERROR(cudaMemcpy(d_data, other.d_data, rowSize * colSize * sizeof(T), cudaMemcpyDeviceToDevice));
        }
        return *this;
    }

    // 소멸자: 할당된 메모리를 해제합니다.
    ~d_matrix() {
        delete[] h_data;
        cudaFree(d_data);
    }

    // resize: 행렬 크기를 재설정하며 기존 메모리는 해제 후 새로 할당합니다.
    void resize(int col, int row){
        delete[] h_data;
        cudaFree(d_data);

        h_data = new T[row * col];
        CHECK_CUDA_ERROR(cudaMalloc(&d_data, row * col * sizeof(T)));
    }

    // cpyToDev: 호스트 데이터를 디바이스 메모리로 복사합니다.
    void cpyToDev() {
        CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, rowSize * colSize * sizeof(T), cudaMemcpyHostToDevice));
    }

    // cpyToHost: 디바이스 데이터를 호스트 메모리로 복사합니다.
    void cpyToHost() {
        CHECK_CUDA_ERROR(cudaMemcpy(h_data, d_data, rowSize * colSize * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // printMatrix: 디바이스 데이터를 호스트로 복사한 후 행렬을 출력합니다.
    void printMatrix() {
        std::vector<T> host_data(rowSize * colSize);
        CHECK_CUDA_ERROR(cudaMemcpy(host_data.data(), d_data, rowSize * colSize * sizeof(T), cudaMemcpyDeviceToHost));

        std::cout << "행렬 출력:" << std::endl;
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                std::cout << std::setw(4) << host_data[i * colSize + j];
            }
            std::cout << std::endl;
        }
    }

    // operator() 오버로드: 2차원 인덱스로 호스트 데이터를 접근합니다.
    T& operator()(int i, int j) const {
        return h_data[i * colSize + j];
    }

    // getDevPointer: 디바이스 메모리 포인터 반환
    T* getDevPointer() const { return d_data; }

    // getHostPointer: 호스트 메모리 포인터 반환
    T* getHostPointer() const { return h_data; }

    // getRow: 행 개수를 반환
    int getRow() const { return rowSize; }

    // getCol: 열 개수를 반환
    int getCol() const { return colSize; }

    // size: 총 원소 개수를 반환
    int size() const { return rowSize*colSize; }

    // fill: 행렬의 모든 요소를 특정 값으로 채우고, 디바이스 메모리로 복사합니다.
    void fill(T value) {
        for (int i = 0; i < rowSize * colSize; ++i) {
            h_data[i] = value;
        }
        cpyToDev();
    }
    // Transpose the matrix and return a new transposed matrix
    d_matrix<T> transpose() const {
        d_matrix<T> transposed(colSize, rowSize);
        dim3 blockSize(32, 32);
        dim3 gridSize((rowSize + blockSize.x - 1) / blockSize.x, (colSize + blockSize.y - 1) / blockSize.y);
        TransInKernel<<<gridSize, blockSize>>>(d_data, transposed.getDevPointer(), rowSize, colSize);
        cudaDeviceSynchronize();
        transposed.cpyToHost();
        return transposed;
    }
};


// English: Overloads the stream insertion operator for printing a d_matrix object.
// 한글: d_matrix 객체를 출력하기 위한 스트림 삽입 연산자를 오버로드합니다.
template <typename T>
std::ostream& operator<<(std::ostream& os, const d_matrix<T>& matrix);

// English: Overloads the stream extraction operator for reading a d_matrix object.
// 한글: d_matrix 객체를 읽기 위한 스트림 추출 연산자를 오버로드합니다.
template <typename T>
std::istream& operator>>(std::istream& is, d_matrix<T>& matrix);

// English: CUDA kernel for performing the Hadamard product (element-wise multiplication) on the GPU.
// 한글: GPU에서 아다마르 곱(원소별 곱셈)을 수행하기 위한 CUDA 커널입니다.
template<typename T>
__global__ void HPinKernel(T* d_A, T* d_B, T* d_C, int row, int col);

// English: Computes the Hadamard product of two matrices using GPU computation.
// 한글: GPU 연산을 사용하여 두 행렬의 아다마르 곱을 계산합니다.
template<typename T>
d_matrix<T> HadamardProduct(const d_matrix<T>& d_A, const d_matrix<T>& d_B);

// English: CUDA kernel for multiplying a matrix by a scalar on the GPU.
// 한글: GPU에서 행렬에 스칼라를 곱하기 위한 CUDA 커널입니다.
template<typename T>
__global__ void ScalaKernel(T* d_A, T scalar, T* d_C, int row, int col);

// English: Multiplies a matrix by a scalar using GPU computation.
// 한글: GPU 연산을 사용하여 행렬에 스칼라를 곱합니다.
template<typename T>
d_matrix<T> ScalaProduct(const d_matrix<T>& d_A, T scalar);

// English: CUDA kernel for matrix multiplication on the GPU.
// 한글: GPU에서 행렬 곱셈을 수행하기 위한 CUDA 커널입니다.
template<typename T>
__global__ void MPinKernel(T* d_A, T* d_B, T* d_C, int row, int col, int eq);

// English: Multiplies two matrices using GPU computation.
// 한글: GPU 연산을 사용하여 두 행렬을 곱합니다.
template<typename T>
d_matrix<T> matrixMP(const d_matrix<T>& d_A, const d_matrix<T>& d_B);

// English: CUDA kernel for adding two matrices on the GPU.
// 한글: GPU에서 두 행렬을 더하기 위한 CUDA 커널입니다.
template<typename T>
__global__ void PlusinKernel(T* d_A, T* d_B, T* d_C, int row, int col);

// English: Adds two matrices using GPU computation.
// 한글: GPU 연산을 사용하여 두 행렬을 더합니다.
template<typename T>
d_matrix<T> matrixPlus(const d_matrix<T>& d_A, const d_matrix<T>& d_B);

// English: Device function for applying the ReLU activation function.
// 한글: ReLU 활성화 함수를 적용하기 위한 디바이스 함수입니다.
template<typename T> __device__ T relu(T x);

// English: Device function for applying the derivative of the ReLU activation function.
// 한글: ReLU 활성화 함수의 도함수를 적용하기 위한 디바이스 함수입니다.
template<typename T> __device__ T d_relu(T x);

// English: Device function for applying the Leaky ReLU activation function.
// 한글: Leaky ReLU 활성화 함수를 적용하기 위한 디바이스 함수입니다.
template<typename T> __device__ T lrelu(T x);

// English: Device function for applying the derivative of the Leaky ReLU activation function.
// 한글: Leaky ReLU 활성화 함수의 도함수를 적용하기 위한 디바이스 함수입니다.
template<typename T> __device__ T d_lrelu(T x);

template<typename T> __device__ T ELU(T x);

template<typename T> __device__ T d_ELU(T x);

template<typename T> __device__ T SELU(T x);

template<typename T> __device__ T d_SELU(T x);

template<typename T> __device__ T Softplus(T x);

template<typename T> __device__ T Softsign(T x);

template<typename T> __device__ T d_Softsign(T x);

template<typename T> __device__ T Swish(T x);

template<typename T> __device__ T d_Swish(T x);

// English: Device function for applying the identity function.
// 한글: 항등 함수를 적용하기 위한 디바이스 함수입니다.
template<typename T> __device__ T Identity(T x);

// English: Device function for applying the sigmoid activation function.
// 한글: 시그모이드 활성화 함수를 적용하기 위한 디바이스 함수입니다.
template<typename T> __device__ T sigmoid(T x);

// English: Device function for applying the derivative of the sigmoid activation function.
// 한글: 시그모이드 활성화 함수의 도함수를 적용하기 위한 디바이스 함수입니다.
template<typename T> __device__ T d_sigmoid(T x);

// English: Device function for applying the derivative of the identity function.
// 한글: 항등 함수의 도함수를 적용하기 위한 디바이스 함수입니다.
template<typename T> __device__ T d_I(T x);

// English: Device function for applying the hyperbolic tangent activation function.
// 한글: 하이퍼볼릭 탄젠트 활성화 함수를 적용하기 위한 디바이스 함수입니다.
template<typename T> __device__ T Tanh(T x);

// English: Device function for applying the derivative of the hyperbolic tangent activation function.
// 한글: 하이퍼볼릭 탄젠트 활성화 함수의 도함수를 적용하기 위한 디바이스 함수입니다.
template<typename T> __device__ T d_tanh(T x);

// English: Device function for squaring a value.
// 한글: 값을 제곱하기 위한 디바이스 함수입니다.
template<typename T> __device__ T sqr(T x);

// English: Device function for dividing a value.
// 한글: 값을 나누기 위한 디바이스 함수입니다.
template<typename T> __device__ T devide(T x);

// English: Device function for applying the natural logarithm.
// 한글: 자연 로그를 적용하기 위한 디바이스 함수입니다.
template<typename T> __device__ T Log(T x);

// English: CUDA kernel for applying an activation function to a matrix on the GPU.
// 한글: GPU에서 행렬에 활성화 함수를 적용하기 위한 CUDA 커널입니다.
template<typename T, T (*ActivateFunc)(T)> __global__ void ActivateInKernel(T* d_A, T* d_C, int row, int col);

// English: Applies an activation function to a matrix using GPU computation.
// 한글: GPU 연산을 사용하여 행렬에 활성화 함수를 적용합니다.
template<typename T, T (*ActivateFunc)(T)> d_matrix<T> MatrixActivate(const d_matrix<T>& d_A);

// English: Sums all elements in a matrix.
// 한글: 행렬의 모든 요소를 합산합니다.
template<typename T> T plusAllElements(const d_matrix<T>& d_A);

// English: CUDA kernel for adding a scalar to all elements of a matrix on the GPU.
// 한글: GPU에서 행렬의 모든 요소에 스칼라를 더하기 위한 CUDA 커널입니다.
template<typename T>
__global__ void plusScalaToMatrix(T* d_C, int row, int col, T scala);

// English: Adds a scalar to all elements of a matrix using GPU computation.
// 한글: GPU 연산을 사용하여 행렬의 모든 요소에 스칼라를 더합니다.
template<typename T>
d_matrix<T> ScalaPlus(const d_matrix<T>& d_A, T scala);

// English: CUDA kernel for casting a matrix to double precision on the GPU.
// 한글: GPU에서 행렬을 더블 정밀도로 변환하기 위한 CUDA 커널입니다.
template <typename T>
__global__ void castKernel(const T* src, double* dst, int N);

// English: Casts a matrix to double precision using GPU computation.
// 한글: GPU 연산을 사용하여 행렬을 더블 정밀도로 변환합니다.
template <typename T>
d_matrix<double> castToDoubleGPU(const d_matrix<T>& input);

// English: CUDA kernel for applying the softmax function to a matrix on the GPU.
// 한글: GPU에서 행렬에 소프트맥스 함수를 적용하기 위한 CUDA 커널입니다.
template<typename T>
__global__ void softmaxKernel(T* in, T* out, int row, int col);

// English: Applies the softmax function to a matrix using GPU computation.
// 한글: GPU 연산을 사용하여 행렬에 소프트맥스 함수를 적용합니다.
template<typename T>
d_matrix<T> softmax(const d_matrix<T>& input);

// English: CUDA kernel for converting zero values in a matrix to a small epsilon value on the GPU.
// 한글: GPU에서 행렬의 0 값을 작은 엡실론 값으로 변환하기 위한 CUDA 커널입니다. x/0 방지용
template<typename T>
__global__ void convertInKernel(T* d_X, int row, int col);

// English: Converts zero values in a matrix to a small epsilon value using GPU computation.
// 한글: GPU 연산을 사용하여 행렬의 0 값을 작은 엡실론 값으로 변환합니다. x/0 방지용
template<typename T>
d_matrix<T> convertZeroToEpsilon(d_matrix<T> x);

// English: CUDA kernel for performing convolution on the GPU.
// 한글: GPU에서 합성곱 연산을 수행하기 위한 CUDA 커널입니다.
template<typename T>
__global__ void convoluteInKernel(T* d_A, T* d_B, T* d_C, int inputRow, int inputCol, int filterRow, int filterCol, int outputRow, int outputCol);

// English: Performs convolution on a matrix using GPU computation.
// 한글: GPU 연산을 사용하여 행렬에 합성곱 연산을 수행합니다.
template<typename T>
d_matrix<T> convolute(const d_matrix<T>& d_A, const d_matrix<T>& d_B);


// English: CUDA kernel for initializing CURAND states on the GPU.
// 한글: GPU에서 CURAND 상태를 초기화하기 위한 CUDA 커널입니다.
__global__ void initCurandStates(curandState *states, unsigned long long seed, int total);

// English: CUDA kernel for initializing weights on the GPU.
// 한글: GPU에서 가중치를 초기화하기 위한 CUDA 커널입니다.
template<typename T> __global__ void InitWeightInKernel(T* d_weight, curandState* states, int row, int col, InitType type);

// English: Initializes weights for a matrix using GPU computation.
// 한글: GPU 연산을 사용하여 행렬의 가중치를 초기화합니다. xevier, he uniform이 있음.
template<typename T> d_matrix<T> InitWeight(int row, int col, InitType type);