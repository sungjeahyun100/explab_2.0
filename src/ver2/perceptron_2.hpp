#include <ver2/d_matrix_2.hpp>
#include <memory>

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
        d2::d_matrix_2<double> mW, vW, mB, vB;
        cudaStream_t st1;
    public:
        Adam(int row, int col, double lr_, double b1=0.9, double b2=0.999, double e=1e-8)
            : lr(lr_), beta1(b1), beta2(b2), eps(e), t(0),
              mW(row, col), vW(row, col), mB(row,1), vB(row,1) {
            mW.fill(0.0); vW.fill(0.0); mB.fill(0.0); vB.fill(0.0);
            cudaStreamCreate(&st1);
        }
        void update(d2::d_matrix_2<double>& W, d2::d_matrix_2<double>& B, const d2::d_matrix_2<double>& gW, const d2::d_matrix_2<double>& gB) override {
            t++;

            double bc1 = 1.0 - std::pow(beta1, t);
            double bc2 = 1.0 - std::pow(beta2, t);

            int rows = W.getRow(), cols = W.getCol();
            int N = rows * cols;
            
            const int blockSize = 256;
            int gridSize = (N + blockSize - 1) / blockSize;
            CalcAdam<<<gridSize, blockSize, 0, st1>>>(gW.getDevPointer(), W.getDevPointer(), mW.getDevPointer(), vW.getDevPointer(), beta1, beta2, lr, eps, bc1, bc2, N);
            CalcAdam<<<gridSize, blockSize, 0, st1>>>(gB.getDevPointer(), B.getDevPointer(), mB.getDevPointer(), vB.getDevPointer(), beta1, beta2, lr, eps, bc1, bc2, rows);
            cudaStreamSynchronize(st1);
        }
        ~Adam() noexcept {
            cudaError_t err = cudaStreamDestroy(st1);
            if (err != cudaSuccess) {
                std::cerr << "[CUDA ERROR in Adam::~Adam] " << cudaGetErrorString(err) << std::endl;
            }
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
        std::unique_ptr<optimizer> opt;
        int deviceId;
        cudaDeviceProp props;
        size_t threadsPerBlock;
        size_t numberOfBlocks;
        cudaStream_t str;
        d2::d_matrix_2<double> w_t;
        d2::d_matrix_2<double> i_t;
    public:
        inline dim3 grid2d(int rows, int cols) {
          return dim3(
            (cols + TILE-1)/TILE,   // x-direction = #tiles across columns
            (rows + TILE-1)/TILE    // y-direction = #tiles across rows
          );
        }

        inline dim3 block2d() { return dim3(TILE, TILE); }

        PerceptronLayer(int i, int o, optimizer* optimizer, d2::InitType init)
            : inputSize(i), outputSize(o),
              input(i,1), weight(o,i), bias(o,1),
              output(o,1), delta(o,1), gradW(o,i), gradB(o,1), opt(optimizer){
            cudaGetDevice(&deviceId);
            cudaGetDeviceProperties(&props, deviceId);
            threadsPerBlock = props.maxThreadsPerBlock;
            numberOfBlocks = props.multiProcessorCount;
            weight = d2::InitWeight<double>(o,i,init);
            bias.fill(0.01);
            i_t.resize(1, inputSize);
            cudaStreamCreate(&str);
        }
    
        void feedforward(const d2::d_matrix_2<double>& in) {
            input = in;
            d2::matmul_tiled<double, 32><<<grid2d(outputSize, inputSize), block2d(), 2*TILE*TILE*sizeof(double), str>>>(weight.getDevPointer(), input.getDevPointer(), output.getDevPointer(), outputSize, 1, inputSize);//2^5, 2^5, 2개
            d2::PlusinKernel<double><<<grid2d(outputSize, 1), block2d(), 0, str>>>(output.getDevPointer(), bias.getDevPointer(), output.getDevPointer(), outputSize, 1);
            cudaStreamSynchronize(str);
            
        }
    
        void backprop(PerceptronLayer* next, const d2::d_matrix_2<double>& ext_delta, const d2::d_matrix_2<double>& act_deriv) {
            d2::d_matrix_2<double> grad_input = ext_delta;
            if(next != nullptr) {
                int tR = next->weight.getCol();
                int tC = next->weight.getRow();
                w_t.resize(tC, tR);
                d2::TransInKernel<double><<<grid2d(tC, tR), block2d(), 0, str>>>(next->weight.getDevPointer(), w_t.getDevPointer(), next->weight.getRow(), next->weight.getCol());
                d2::matmul_tiled<double, 32><<<grid2d(inputSize, outputSize), block2d(), 2*TILE*TILE*sizeof(double), str>>>(w_t.getDevPointer(), next->delta.getDevPointer(), grad_input.getDevPointer(), inputSize, 1, outputSize);
            }

            {
                d2::HPinKernel<double><<<grid2d(inputSize, 1), block2d(), 0, str>>>(grad_input.getDevPointer(), act_deriv.getDevPointer(), delta.getDevPointer(), inputSize, 1);
                d2::TransInKernel<double><<<grid2d(1, inputSize), block2d(), 0, str>>>(input.getDevPointer(), i_t.getDevPointer(), input.getRow(), 1);
                d2::matmul_tiled<double, 32><<<grid2d(outputSize, inputSize), block2d(), 2*TILE*TILE*sizeof(double), str>>>(delta.getDevPointer(), i_t.getDevPointer(), gradW.getDevPointer(), outputSize, inputSize, 1);
                gradB = delta;
            }
            opt->update(weight, bias, gradW, gradB);
            cudaStreamSynchronize(str);
        }
    
        d2::d_matrix_2<double>& getOutput(){ return output; }

        ~PerceptronLayer() noexcept {
            cudaStreamDestroy(str);
        }
    };

    // ActivateLayer-------------------------------------------------------------------------------------------------------------------

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
    };
    
    void ActivateLayer::pushInput(const d2::d_matrix_2<double>& in){
        input = in;
        input.cpyToDev();
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
    double LossLayer::getLoss(){
        // 1) 디바이스→호스트 복사
        output.cpyToHost();
        target.cpyToHost();
    
        switch (Loss)
        {
            case LossType::MSE: {
                // MSE: L = 1/N Σ (output − target)², 전부 호스트 계산
                int N = output.getRow();
                double sum = 0.0;
                for (int i = 0; i < N; ++i) {
                    double diff = output(i, 0) - target(i, 0);
                    sum += diff * diff;
                }
                return sum / static_cast<double>(N);
            }
    
            case LossType::CrossEntropy: {
                int N = output.getRow();
                // 2) 소프트맥스 확률 계산
                d2::d_matrix_2<double> p = softmax(output);
    
                // 3) 크로스엔트로피 손실: L = -1/N Σ y_i * log(p_i)
                double loss = 0.0;
                for (int i = 0; i < N; ++i) {
                    double yi = target(i, 0);
                    double pi = std::min(std::max(p(i, 0), 1e-12), 1.0);  // 클리핑
                    loss -= yi * std::log(pi);
                }
                return loss / static_cast<double>(N);
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

    class convLayer{
        private:
            
    };

}//namespace perceptron_2

