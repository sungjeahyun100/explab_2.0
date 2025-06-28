#include "perceptronVer2.hpp"

void ActivateLayer::pushInput(const d_matrix<double>& in){
    input = in;
    input.cpyToDev();
}

// 활성화 적용 (output = f(input))
// 지원: ReLU, LReLU, Identity, Sigmoid
void ActivateLayer::Active(){
    switch (act) {
        case ActivationType::ReLU:
            output = MatrixActivate<double, relu>(input); break;
        case ActivationType::LReLU:
            output = MatrixActivate<double, lrelu>(input); break;
        case ActivationType::Identity:
            output = MatrixActivate<double, Identity>(input); break;
        case ActivationType::Sigmoid:
            output = MatrixActivate<double, sigmoid>(input); break;
        case ActivationType::Tanh:
            output = MatrixActivate<double, Tanh>(input); break;
        case ActivationType::ELU:
            output = MatrixActivate<double, ELU>(input); break;
        case ActivationType::SELU:
            output = MatrixActivate<double, SELU>(input); break;
        case ActivationType::Softplus:
            output = MatrixActivate<double, Softplus>(input); break;
        case ActivationType::Softsign:
            output = MatrixActivate<double, Softsign>(input); break;
        case ActivationType::Swish:
            output = MatrixActivate<double, Swish>(input); break;
        default:
            throw std::runtime_error("Unsupported ActivationType in perceptronLayer");
    }
}

// 활성화 함수 미분값 반환 (f'(z))
// ReLU: 1(x>0), 0(x<=0)
// LReLU: 1(x>0), 0.01(x<=0)
// Identity: 1
// Sigmoid: σ'(x) = σ(x)(1-σ(x))
d_matrix<double> ActivateLayer::d_Active(const d_matrix<double>& z) {
    switch (act) {
        case ActivationType::ReLU:
            return MatrixActivate<double, d_relu>(z); 
        case ActivationType::LReLU:
            return MatrixActivate<double, d_lrelu>(z); 
        case ActivationType::Identity:
            return MatrixActivate<double, d_I>(z); 
        case ActivationType::Sigmoid:
            return MatrixActivate<double, d_sigmoid>(z); 
        case ActivationType::Tanh:
            return MatrixActivate<double, d_tanh>(z); 
        case ActivationType::ELU:
            return MatrixActivate<double, d_ELU>(z); 
        case ActivationType::SELU:
            return MatrixActivate<double, d_SELU>(z); 
        case ActivationType::Softplus:
            return MatrixActivate<double, sigmoid>(z);
        case ActivationType::Softsign:
            return MatrixActivate<double, d_Softsign>(z);
        case ActivationType::Swish:
            return MatrixActivate<double, d_Swish>(z);
        default:
            throw std::runtime_error("Unsupported ActivationType in d_Active");
    }
}

// 활성화 결과 반환
const d_matrix<double>& ActivateLayer::getOutput() const {
    return output; 
}

// 타겟 입력
void LossLayer::pushTarget(const d_matrix<double>& Target){
    target = Target;
}

// 출력 입력
void LossLayer::pushOutput(const d_matrix<double>& Output){
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
            auto diff = matrixPlus(output, ScalaProduct<double>(target, -1));
            auto S = HadamardProduct(diff, diff);
            sum = plusAllElements(S);
            return sum / static_cast<double>(N);
        }

        case LossType::CrossEntropy: {
            int N = output.getRow();
            // 2) 소프트맥스 확률 계산
            d_matrix<double> p = softmax(output);

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
d_matrix<double> LossLayer::getGrad() {
    // 1) 디바이스→호스트 복사
    output.cpyToHost();
    target.cpyToHost();

    switch (Loss) {
        case LossType::MSE: {
            // L = (1/N) Σ (o - t)^2  이므로  dL/dz = 2*(o - t)/N
            int N = output.getRow();
            // diff = output - target
            d_matrix<double> diff = matrixPlus(output, ScalaProduct(target, -1.0));
            return ScalaProduct(diff, 2.0 / static_cast<double>(N));
        }

        case LossType::CrossEntropy: {
            int N = output.getRow();
            // 2) 소프트맥스 확률 계산
            d_matrix<double> p = softmax(output);

            // 3) gradient = (p - y) / N
            d_matrix<double> grad = matrixPlus(p, ScalaProduct(target, -1.0));
            return ScalaProduct(grad, 1.0 / static_cast<double>(N));
        }

        default:
            throw std::runtime_error("Unsupported LossType in getGrad");
    }
}



