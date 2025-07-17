#include <ver2/d_matrix_2.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <cudnn.h>
#include <memory>

#define CHK_CUDNN(call) if((call)!=CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(call));

namespace d2 = d_matrix_ver2;

namespace perceptron_2 {
    enum class ActType {
        ReLU,
        LReLU,
        ELU,
        SELU,
        Tanh,
        Identity,
        Sigmoid,
        Softplus,
        Softsign,
        Swish
    };
    
    enum class LossType {
        MSE,
        CrossEntropy
    };

    class optimizer{
        public:
            virtual ~optimizer() = default;
            virtual void update(d2::d_matrix_2<double>& W, d2::d_matrix_2<double>& B, const d2::d_matrix_2<double>& gW, const d2::d_matrix_2<double>& gB) = 0;
    };

    class SGD : public optimizer {
        double lr;
    public:
        explicit SGD(double lr_) : lr(lr_) {}
        void update(d2::d_matrix_2<double>& W, d2::d_matrix_2<double>& B, const d2::d_matrix_2<double>& gW, const d2::d_matrix_2<double>& gB) override {
            W = d2::matrixPlus(W, d2::ScalaProduct(gW, -lr));
            B = d2::matrixPlus(B, d2::ScalaProduct(gB, -lr));
        }
    };

    __global__ void CalcAdam(
        const double* __restrict__ d_g,
        double*       __restrict__ d_W,
        double*       __restrict__ d_M,
        double*       __restrict__ d_V,
        double        beta1,
        double        beta2,
        double        lr,
        double        eps,
        double        bc1,
        double        bc2,
        int           N      // 총 원소 수 = row*col
    ) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx >= N) return;
    
        // 1) 모멘텀·분산 업데이트
        double gi = d_g[idx];
        double m  = d_M[idx] = d_M[idx]*beta1 + gi*(1.0-beta1);
        double v  = d_V[idx] = d_V[idx]*beta2 + gi*gi*(1.0-beta2);
    
        // 2) 편향 보정된 모멘텀·분산
        double m_hat = m / bc1;
        double v_hat = v / bc2;
    
        // 3) 파라미터 업데이트
        double inv = sqrt(v_hat) / (1.0 + eps*sqrt(v_hat)); // 1/(sqrt(v_hat)+eps)
        d_W[idx]   -= lr * m_hat * inv;
    }
    
    class Adam : public optimizer {
        double lr, beta1, beta2, eps;
        int t;
        // weight: row×col, bias: 1×col
        d2::d_matrix_2<double> mW, vW;  // same shape as W
        d2::d_matrix_2<double> mB, vB;  // 1×col
        cudaStream_t st1;
    public:
        Adam(int row, int col, double lr_, double b1=0.9, double b2=0.999, double e=1e-8)
          : lr(lr_), beta1(b1), beta2(b2), eps(e), t(0),
            mW(row, col), vW(row, col),
            mB(1,     col), vB(1,     col)  // bias buffers
        {
            mW.fill(0.0);  vW.fill(0.0);
            mB.fill(0.0);  vB.fill(0.0);
            cudaStreamCreate(&st1);
        }
    
        void update(
            d2::d_matrix_2<double>& W,
            d2::d_matrix_2<double>& B,
            const d2::d_matrix_2<double>& gW,
            const d2::d_matrix_2<double>& gB
        ) override {
            ++t;
            double bc1 = 1.0 - std::pow(beta1, t);
            double bc2 = 1.0 - std::pow(beta2, t);
    
            int rows = W.getRow(), cols = W.getCol();
            int Nw   = rows * cols;  // total weight elements
            int Nb   = cols;         // total bias elements (1×col)
    
            const int blockSize = 256;
            int gridW = (Nw + blockSize - 1) / blockSize;
            int gridB = (Nb + blockSize - 1) / blockSize;
    
            // 1) weight 업데이트
            CalcAdam<<<gridW, blockSize, 0, st1>>>(
                gW.getDevPointer(),
                W.getDevPointer(),
                mW.getDevPointer(),
                vW.getDevPointer(),
                beta1, beta2, lr, eps,
                bc1, bc2,
                Nw
            );
    
            // 2) bias 업데이트 (이제 B는 1×col 이므로 N_B = cols)
            CalcAdam<<<gridB, blockSize, 0, st1>>>(
                gB.getDevPointer(),
                B.getDevPointer(),
                mB.getDevPointer(),
                vB.getDevPointer(),
                beta1, beta2, lr, eps,
                bc1, bc2,
                Nb
            );
    
            cudaStreamSynchronize(st1);
        }
    
        ~Adam() noexcept {
            cudaStreamDestroy(st1);
        }
    };

    constexpr int TILE = 32;

    class PerceptronLayer {
    protected:
        int inputSize;
        int outputSize;
        d2::d_matrix_2<double> input;
        d2::d_matrix_2<double> weight;
        d2::d_matrix_2<double> bias;
        d2::d_matrix_2<double> output;
        d2::d_matrix_2<double> delta;
        d2::d_matrix_2<double> gradW;
        d2::d_matrix_2<double> gradB;
        optimizer* opt;
        int deviceId;
        cudaDeviceProp props;
        size_t threadsPerBlock;
        size_t numberOfBlocks;
        cudaStream_t str;
        d2::d_matrix_2<double> w_t;
        d2::d_matrix_2<double> i_t;
        int sample_num;
    public:
        inline dim3 grid2d(int rows, int cols) {
          return dim3(
            (cols + TILE-1)/TILE,   // x-direction = #tiles across columns
            (rows + TILE-1)/TILE    // y-direction = #tiles across rows
          );
        }

        inline dim3 block2d() { return dim3(TILE, TILE); }

        PerceptronLayer(int n, int i, int o, optimizer* optimizer, d2::InitType init)
            : inputSize(i), outputSize(o), sample_num(n),
              input(n, i), weight(i, o), bias(1, o),
              output(n, o), delta(n, o), gradW(i, o), gradB(1, o), opt(optimizer){
            cudaGetDevice(&deviceId);
            cudaGetDeviceProperties(&props, deviceId);
            threadsPerBlock = props.maxThreadsPerBlock;
            numberOfBlocks = props.multiProcessorCount;
            weight = d2::InitWeight<double>(i, o,init);
            bias.fill(0.01);
            i_t.resize(inputSize, n);
            cudaStreamCreate(&str);
        }

        const d2::d_matrix_2<double>& getWeight() const { return weight; }
        const d2::d_matrix_2<double>& getDelta()  const { return delta; }
    
        void feedforward(const d2::d_matrix_2<double>& in) {
            input = in;
            d2::matmul_tiled<double, TILE><<<grid2d(outputSize, inputSize), block2d(), 2*TILE*TILE*sizeof(double), str>>>(input.getDevPointer(), weight.getDevPointer(), output.getDevPointer(), sample_num, outputSize, inputSize);//2^5, 2^5, 2개
            d2::PlusinKernel<double><<<grid2d(sample_num, outputSize), block2d(), 0, str>>>(output.getDevPointer(), bias.getDevPointer(), output.getDevPointer(), sample_num, outputSize);
            cudaStreamSynchronize(str);
            
        }
    
        void backprop(PerceptronLayer* next, const d2::d_matrix_2<double>& ext_delta, const d2::d_matrix_2<double>& act_deriv) {
            d2::d_matrix_2<double> grad_input = ext_delta;
            if(next != nullptr) {
                int tR = next->weight.getCol();
                int tC = next->weight.getRow();
                w_t.resize(tC, tR);
                d2::TransInKernel<double><<<grid2d(tC, tR), block2d(), 0, str>>>(next->weight.getDevPointer(), w_t.getDevPointer(), next->weight.getRow(), next->weight.getCol());
                d2::matmul_tiled<double, 32><<<grid2d(inputSize, outputSize), block2d(), 2*TILE*TILE*sizeof(double), str>>>(next->delta.getDevPointer(), w_t.getDevPointer(), grad_input.getDevPointer(), sample_num, inputSize, outputSize);
            }

            {
                d2::HPinKernel<double><<<grid2d(inputSize, 1), block2d(), 0, str>>>(grad_input.getDevPointer(), act_deriv.getDevPointer(), delta.getDevPointer(), inputSize, sample_num);
                d2::TransInKernel<double><<<grid2d(inputSize, sample_num), block2d(), 0, str>>>(input.getDevPointer(), i_t.getDevPointer(), sample_num, inputSize);
                d2::matmul_tiled<double, TILE><<<grid2d(outputSize, inputSize), block2d(), 2*TILE*TILE*sizeof(double), str>>>(i_t.getDevPointer(), delta.getDevPointer(), gradW.getDevPointer(), inputSize, outputSize, sample_num);
                d2::reduceRows<double><<<numberOfBlocks, threadsPerBlock, 0, str>>>(delta.getDevPointer(), gradB.getDevPointer(), sample_num, outputSize);
            }
            opt->update(weight, bias, gradW, gradB);
            cudaStreamSynchronize(str);
        }
    
        d2::d_matrix_2<double>& getOutput(){ return output; }

        ~PerceptronLayer() noexcept {
            cudaStreamDestroy(str);
        }
    };

    // ActivateLayer----------------------------------------------------------------------------------------------------------------------------

    // 활성화 계층
    // 사용법: pushInput()으로 입력, Active()로 활성화 적용, getOutput()으로 결과 반환
    // 지원: ReLU, LReLU, Identity, Sigmoid
    // d_Active: 미분값 반환
    class ActivateLayer{
        private:
            ActType act;
            d2::d_matrix_2<double> input;
            d2::d_matrix_2<double> output;
        public:
            // 생성자: 행, 열, 활성화 종류 지정
            ActivateLayer(int row, int col, ActType a) : input(row, col), output(row, col), act(a){}
            // 입력 설정
            void pushInput(const d2::d_matrix_2<double>& in);
            // 활성화 적용 (output = f(input))
            void Active();
            // 활성화 미분값 반환 (f'(z))
            d2::d_matrix_2<double> d_Active(const d2::d_matrix_2<double>& z);
            // 결과 반환
            const d2::d_matrix_2<double>& getOutput() const ;
    };
    
    // LossLayer--------------------------------------------------------------------------------------------------------------------------------
    
    // 손실 계층
    // 사용법: pushTarget, pushOutput으로 데이터 입력 후 getLoss(), getGrad() 호출
    // 지원: MSE(평균제곱오차), CrossEntropy(크로스엔트로피)
    // getLoss: loss 반환, getGrad: dL/dz 반환
    class LossLayer{
        private:
            d2::d_matrix_2<double> target;
            d2::d_matrix_2<double> output;
            LossType Loss;
        public:
            // 생성자: 행, 열, 손실 종류 지정
            LossLayer(int row, int col, LossType L) : target(row, col), output(row, col), Loss(L){}
            // 타겟/출력 입력
            void pushTarget(const d2::d_matrix_2<double>& Target);
            void pushOutput(const d2::d_matrix_2<double>& Output);
            // 손실값 반환
            double getLoss();
            // 손실 미분 반환
            d2::d_matrix_2<double> getGrad();
            inline dim3 grid2d(int rows, int cols) {
                return dim3(
                  (cols + TILE-1)/TILE,   // x-direction = #tiles across columns
                  (rows + TILE-1)/TILE    // y-direction = #tiles across rows
                );
            }
    
            inline dim3 block2d() { return dim3(TILE, TILE); }
    };
    
    void ActivateLayer::pushInput(const d2::d_matrix_2<double>& in){
        input = in;
    }
    
    // 활성화 적용 (output = f(input))
    // 지원: ReLU, LReLU, Identity, Sigmoid
    void ActivateLayer::Active(){
        switch (act) {
            case ActType::ReLU:
                output = d2::MatrixActivate<double, d2::relu>(input); break;
            case ActType::LReLU:
                output = d2::MatrixActivate<double, d2::lrelu>(input); break;
            case ActType::Identity:
                output = d2::MatrixActivate<double, d2::Identity>(input); break;
            case ActType::Sigmoid:
                output = d2::MatrixActivate<double, d2::sigmoid>(input); break;
            case ActType::Tanh:
                output = d2::MatrixActivate<double, d2::Tanh>(input); break;
            case ActType::ELU:
                output = d2::MatrixActivate<double, d2::ELU>(input); break;
            case ActType::SELU:
                output = d2::MatrixActivate<double, d2::SELU>(input); break;
            case ActType::Softplus:
                output = d2::MatrixActivate<double, d2::Softplus>(input); break;
            case ActType::Softsign:
                output = d2::MatrixActivate<double, d2::Softsign>(input); break;
            case ActType::Swish:
                output = d2::MatrixActivate<double, d2::Swish>(input); break;
            default:
                throw std::runtime_error("Unsupported ActivationType in perceptronLayer");
        }
    }
    
    // 활성화 함수 미분값 반환 (f'(z))
    // ReLU: 1(x>0), 0(x<=0)
    // LReLU: 1(x>0), 0.01(x<=0)
    // Identity: 1
    // Sigmoid: σ'(x) = σ(x)(1-σ(x))
    d2::d_matrix_2<double> ActivateLayer::d_Active(const d2::d_matrix_2<double>& z) {
        switch (act) {
            case ActType::ReLU:
                return d2::MatrixActivate<double, d2::d_relu>(z); 
            case ActType::LReLU:
                return d2::MatrixActivate<double, d2::d_lrelu>(z); 
            case ActType::Identity:
                return d2::MatrixActivate<double, d2::d_I>(z); 
            case ActType::Sigmoid:
                return d2::MatrixActivate<double, d2::d_sigmoid>(z); 
            case ActType::Tanh:
                return d2::MatrixActivate<double, d2::d_tanh>(z); 
            case ActType::ELU:
                return d2::MatrixActivate<double, d2::d_ELU>(z); 
            case ActType::SELU:
                return d2::MatrixActivate<double, d2::d_SELU>(z); 
            case ActType::Softplus:
                return d2::MatrixActivate<double, d2::sigmoid>(z);
            case ActType::Softsign:
                return d2::MatrixActivate<double, d2::d_Softsign>(z);
            case ActType::Swish:
                return d2::MatrixActivate<double, d2::d_Swish>(z);
            default:
                throw std::runtime_error("Unsupported ActivationType in d_Active");
        }
    }
    
    // 활성화 결과 반환
    const d2::d_matrix_2<double>& ActivateLayer::getOutput() const {
        return output; 
    }
    
    // 타겟 입력
    void LossLayer::pushTarget(const d2::d_matrix_2<double>& Target){
        target = Target;
    }
    
    // 출력 입력
    void LossLayer::pushOutput(const d2::d_matrix_2<double>& Output){
        output = Output;
    }
    
    // 손실값 반환
    // MSE: L = 1/n Σ(y-p)^2
    // CrossEntropy: L = -Σ y log(softmax(p))
    double LossLayer::getLoss() {
        switch (Loss) {
            case LossType::MSE: {
                int N = output.getRow();
    
                auto idx_begin = thrust::make_counting_iterator(0);
                auto idx_end   = thrust::make_counting_iterator(N);
    
                thrust::device_ptr<double> o_dev(output.getDevPointer());
                thrust::device_ptr<double> t_dev(target.getDevPointer());
                double* o_ptr = thrust::raw_pointer_cast(o_dev);
                double* t_ptr = thrust::raw_pointer_cast(t_dev);
    
                double sum = thrust::transform_reduce(
                    thrust::device,
                    idx_begin, idx_end,
    
                    [o_ptr, t_ptr] __device__ (int i) {
                        double d = o_ptr[i] - t_ptr[i];
                        return d * d;
                    },
    
                    0.0,
                    thrust::plus<double>()
                );
                return sum / (static_cast<double>(N)*output.getCol());
            }
    
            case LossType::CrossEntropy: {
                int N = output.getRow();
                // 먼저 소프트맥스
                d2::d_matrix_2<double> p = softmax(output);
    
                auto idx_begin = thrust::make_counting_iterator(0);
                auto idx_end   = thrust::make_counting_iterator(N);
    
                thrust::device_ptr<double> p_dev(p.getDevPointer());
                thrust::device_ptr<double> t_dev(target.getDevPointer());
                double* p_ptr = thrust::raw_pointer_cast(p_dev);
                double* t_ptr = thrust::raw_pointer_cast(t_dev);
    
                double sum = thrust::transform_reduce(
                    thrust::device,
                    idx_begin, idx_end,
    
                    [p_ptr, t_ptr] __device__ (int i) {
                        return -t_ptr[i] * ::log(p_ptr[i]);
                    },
    
                    0.0,
                    thrust::plus<double>()
                );
                return sum / (static_cast<double>(N)*output.getCol());
            }
    
            default:
                throw std::runtime_error("Unsupported LossType in getLoss");
        }
    }
    // 손실 미분 반환
    // MSE: dL/dz = 2(y-p)
    // CrossEntropy: dL/dz = softmax(p) - y
    d2::d_matrix_2<double> LossLayer::getGrad() {
        // 1) 디바이스→호스트 복사
        output.cpyToHost();
        target.cpyToHost();
    
        switch (Loss) {
            case LossType::MSE: {
                // L = (1/N) Σ (o - t)^2  이므로  dL/dz = 2*(o - t)/N
                int N = output.getRow();
                // diff = output - target
                d2::d_matrix_2<double> diff = matrixPlus(output, ScalaProduct(target, -1.0));
                return ScalaProduct(diff, 2.0 / static_cast<double>(N));
            }
    
            case LossType::CrossEntropy: {
                int N = output.getRow();
                // 2) 소프트맥스 확률 계산
                d2::d_matrix_2<double> p = softmax(output);
    
                // 3) gradient = (p - y) / N
                d2::d_matrix_2<double> grad = matrixPlus(p, ScalaProduct(target, -1.0));
                return ScalaProduct(grad, 1.0 / static_cast<double>(N));
            }
    
            default:
                throw std::runtime_error("Unsupported LossType in getGrad");
        }
    }

    class convLayer {
        // cuDNN 오브젝트
        cudnnHandle_t                _handle;
        cudnnTensorDescriptor_t      _xDesc, _yDesc, _biasDesc;
        cudnnFilterDescriptor_t      _wDesc;
        cudnnConvolutionDescriptor_t _convDesc;
        cudnnConvolutionFwdAlgo_t    _bestAlgo;
        cudnnConvolutionBwdDataAlgo_t _bestBwdDAlgo;
        cudnnConvolutionBwdFilterAlgo_t _bestBwdFAlgo;
        size_t                       _workspaceBytes;
        void*                        _workspace = nullptr;
    
        // 호스트/디바이스 저장용
        int N,C,H,W,K,R,S,pad_h,pad_w,stride_h,stride_w;
        d2::d_matrix_2<double> input, kernel, bias;
        d2::d_matrix_2<double> output, delta, gradW, gradB;
    
        optimizer* opt;
    
      public:
        convLayer(int N_, int C_, int H_, int W_,
                  int K_, int R_, int S_,
                  int pad_h_, int pad_w_,
                  int stride_h_, int stride_w_,
                  optimizer* o)
          : N(N_), C(C_), H(H_), W(W_)
          , K(K_), R(R_), S(S_)
          , pad_h(pad_h_), pad_w(pad_w_)
          , stride_h(stride_h_), stride_w(stride_w_)
          , input(N_, C_*H_*W_)
          , kernel(K_, C_*R_*S_)
          , bias(1, K_)
          , output(N_, K_*((H_+2*pad_h_-R_)/stride_h_+1)*((W_+2*pad_w_-S_)/stride_w_+1))
          , delta(output.getRow(), output.getCol())
          , gradW(kernel.getRow(), kernel.getCol())
          , gradB(bias.getRow(), bias.getCol())
          , opt(o)
        {
            // — cuDNN 생략 없이 에러 체크까지 —
            CHK_CUDNN(cudnnCreate(&_handle));
            CHK_CUDNN(cudnnCreateTensorDescriptor(&_xDesc));
            CHK_CUDNN(cudnnCreateTensorDescriptor(&_yDesc));
            CHK_CUDNN(cudnnCreateTensorDescriptor(&_biasDesc));
            CHK_CUDNN(cudnnCreateFilterDescriptor(&_wDesc));
            CHK_CUDNN(cudnnCreateConvolutionDescriptor(&_convDesc));
      
            CHK_CUDNN(cudnnSetTensor4dDescriptor(_xDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N,C,H,W));
            CHK_CUDNN(cudnnSetFilter4dDescriptor(_wDesc,
                CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K,C,R,S));
            CHK_CUDNN(cudnnSetConvolution2dDescriptor(_convDesc,
                pad_h, pad_w, stride_h, stride_w, 1,1,
                CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));
      
            int outN,outC,outH,outW;
            CHK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
                _convDesc, _xDesc, _wDesc, &outN,&outC,&outH,&outW));
            CHK_CUDNN(cudnnSetTensor4dDescriptor(_yDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, outN,outC,outH,outW));
            CHK_CUDNN(cudnnSetTensor4dDescriptor(_biasDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1,outC,1,1));
      
            // fastest algo 한 개만 뽑기 (수정:알고리즘 여려개 뽐기로 수정함)
            int algoCount = 0;
            cudnnConvolutionFwdAlgoPerf_t perf[ CUDNN_CONVOLUTION_FWD_ALGO_COUNT ];
            CHK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(_handle, _xDesc, _wDesc, _convDesc, _yDesc, CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &algoCount, perf));
            _bestAlgo      = perf[0].algo;
            _workspaceBytes= perf[0].memory;

            cudnnConvolutionBwdFilterAlgoPerf_t perfF[ CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT ];
            int returnedF = 0;
            CHK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(_handle, _xDesc, _yDesc, _convDesc, _wDesc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &returnedF, perfF));
            _workspaceBytes  = max(_workspaceBytes, perfF[0].memory);
  
            cudnnConvolutionBwdDataAlgoPerf_t perfD[ CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT ];
            int returnedD = 0;
            CHK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(_handle, _wDesc, _yDesc, _convDesc, _xDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &returnedD, perfD));
            _workspaceBytes  = max(_workspaceBytes, perfD[0].memory);

            _bestBwdFAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;  // 기본값
            for(int i = 0; i < returnedF; ++i) {
                if (perfF[i].status == CUDNN_STATUS_SUCCESS && perfF[i].memory <= _workspaceBytes) {
                    _bestBwdFAlgo = perfF[i].algo;
                    _workspaceBytes = std::max(_workspaceBytes, perfF[i].memory);
                    break;
                }
            }
            _bestBwdDAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;  // 기본값
            for(int i = 0; i < returnedD; ++i) {
                if (perfD[i].status == CUDNN_STATUS_SUCCESS && perfD[i].memory <= _workspaceBytes) {
                    _bestBwdDAlgo = perfD[i].algo;
                    _workspaceBytes = std::max(_workspaceBytes, perfD[i].memory);
                    break;
                }
            }
  
            if (_workspaceBytes > 0) {
              cudaError_t err = cudaMalloc(&_workspace, _workspaceBytes);
              if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc failed for workspace: ") + cudaGetErrorString(err));
              }
            }
  
            std::cout << "[conv] fwdAlgo=" << _bestAlgo
            << " bwdF=" << _bestBwdFAlgo
            << " bwdD=" << _bestBwdDAlgo
            << " workspace=" << (_workspaceBytes/1024/1024) << "MB\n";
        }

        inline dim3 grid2d(int rows, int cols) {
          return dim3(
            (cols + TILE-1)/TILE,   // x-direction = #tiles across columns
            (rows + TILE-1)/TILE    // y-direction = #tiles across rows
          );
        }

        inline dim3 block2d() { return dim3(TILE, TILE); }
    
        // 순전파 (호스트에서 호출)
        // x: (N × C×H×W)  kernel: (K × C×R×S)  bias: (1×K×1×1)
        d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& x_dev) {
          // 1) 입력 복사
          input = x_dev;
          // 2) convolution
          const double alpha=1.0, beta=0.0;
          CHK_CUDNN(cudnnConvolutionForward(
              _handle, &alpha,
              _xDesc, input.getDevPointer(),
              _wDesc, kernel.getDevPointer(),
              _convDesc, _bestAlgo,
              _workspace, _workspaceBytes,
              &beta,    _yDesc, output.getDevPointer()));
          // 3) bias 추가
          CHK_CUDNN(cudnnAddTensor(
              _handle, &alpha,
              _biasDesc, bias.getDevPointer(),
              &alpha,
              _yDesc, output.getDevPointer()));
          return output;  // (N × K×Ho×Wo)
        }
    
        // 역전파 (호스트에서 호출)
        // next==nullptr 이면 맨 앞 레이어
        // dY: Loss 에서 넘어온 dL/dZ  act_deriv: 활성화 함수 도함수(z)
        d2::d_matrix_2<double> backward(PerceptronLayer* next, const d2::d_matrix_2<double>& dY_dev, const d2::d_matrix_2<double>& act_deriv_dev)
        {
          // 1) 외부 델타 or 다음 레이어 전파 델타 준비
          if(next){
            // 다음 레이어의 convLayer::delta 를 가져와서
            delta = d2::matrixMP(next->getDelta(), next->getWeight().transpose());  
          } else {
            delta = dY_dev;
          }
          // 2) 활성화 미분 곱하기
          int Rr=delta.getRow(), Cc=delta.getCol();
          d2::HPinKernel<<<grid2d(Rr,Cc), block2d()>>>(delta.getDevPointer(), act_deriv_dev.getDevPointer(), delta.getDevPointer(), Rr, Cc);
          cudaDeviceSynchronize();
          cudaError_t err = cudaGetLastError();
          if (err != cudaSuccess) {
              std::cerr << "[ERROR] HPinKernel failed at convLayer::backward(): "
                        << cudaGetErrorString(err) << std::endl;
              std::abort();
          }
    
          // 3) gradW 계산
          const double alpha=1.0, beta=0.0;
          CHK_CUDNN(cudnnConvolutionBackwardFilter(
              _handle, &alpha,
              _xDesc, input.getDevPointer(),
              _yDesc, delta.getDevPointer(),
              _convDesc, _bestBwdFAlgo,
              _workspace, _workspaceBytes,
              &beta,
              _wDesc, gradW.getDevPointer()));
    
          // 4) gradB 계산
          CHK_CUDNN(cudnnConvolutionBackwardBias(
              _handle, &alpha,
              _yDesc, delta.getDevPointer(),
              &beta,
              _biasDesc, gradB.getDevPointer()));
    
          // 5) dX 계산 (이전 레이어로 전파할 델타)
          CHK_CUDNN(cudnnConvolutionBackwardData(
              _handle, &alpha,
              _wDesc, kernel.getDevPointer(),
              _yDesc, delta.getDevPointer(),
              _convDesc, _bestBwdDAlgo,
              _workspace, _workspaceBytes,
              &beta,
              _xDesc, input.getDevPointer()));
    
          // 6) 파라미터 업데이트
          opt->update(kernel, bias, gradW, gradB);
    
          // 최종적으로 이전 레이어로 보낼 델타 반환
          // (already device 에 있으므로 Host 로 안 옮겨도 됩니다)
          return delta;
        }
    
        ~convLayer(){
          if(_workspace)  cudaFree(_workspace);
          cudnnDestroyConvolutionDescriptor(_convDesc);
          cudnnDestroyFilterDescriptor(_wDesc);
          cudnnDestroyTensorDescriptor(_xDesc);
          cudnnDestroyTensorDescriptor(_yDesc);
          cudnnDestroyTensorDescriptor(_biasDesc);
          cudnnDestroy(_handle);
        }
    
        // 필요하다면 getter들도 추가
        const d2::d_matrix_2<double>& getOutput() const { return output; }
        const d2::d_matrix_2<double>& getDelta () const { return delta;  }
        const d2::d_matrix_2<double>& getGradW () const { return gradW;  }
        const d2::d_matrix_2<double>& getGradB () const { return gradB;  }
    };


}//namespace perceptron_2

