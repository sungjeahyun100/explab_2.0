#ifndef PERCEPTRON_VER2_HPP
#define PERCEPTRON_VER2_HPP

#include <cmath>
#include <string>
#include <chrono>
#include "d_matrix.hpp"

// 활성화 타입과 손실 타입은 기존과 동일하게 사용
enum class ActivationType {
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


//-------------------------------------------------------------
// Optimizer base class and two implementations
//-------------------------------------------------------------
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(d_matrix<double>& W, d_matrix<double>& B, const d_matrix<double>& gW, const d_matrix<double>& gB) = 0;
    virtual double getLR() = 0;
};

class SGD : public Optimizer {
    double lr;
public:
    SGD(double lr_) : lr(lr_) {}
    void update(d_matrix<double>& W, d_matrix<double>& B, const d_matrix<double>& gW, const d_matrix<double>& gB) override {
        W = matrixPlus(W, ScalaProduct(gW, -lr));
        B = matrixPlus(B, ScalaProduct(gB, -lr));
    }
    double getLR() override { return lr; }
};

class Adam : public Optimizer {
    double lr, beta1, beta2, eps;
    int t;
    d_matrix<double> mW, vW, mB, vB;
public:
    Adam(double lr_, double b1=0.9, double b2=0.999, double e=1e-8)
      : lr(lr_), beta1(b1), beta2(b2), eps(e), t(0), mW(1, 1), mB(1, 1), vW(1, 1), vB(1, 1) {}

    // 실제 상태(state) 할당은 init() 호출 시에
    void init(int row, int col) {
        mW.resize(row, col);  vW.resize(row, col);
        mB.resize(row, 1);    vB.resize(row, 1);
        mW.fill(0.0);  vW.fill(0.0);
        mB.fill(0.0);  vB.fill(0.0);
    }
    void update(d_matrix<double>& W, d_matrix<double>& B, const d_matrix<double>& gW, const d_matrix<double>& gB) override {

        if (t==0) init(W.getRow(), W.getCol());

        t++;
        mW = matrixPlus(ScalaProduct(mW,beta1), ScalaProduct(gW,1.0-beta1));
        vW = matrixPlus(ScalaProduct(vW,beta2), ScalaProduct(HadamardProduct(gW,gW),1.0-beta2));
        mB = matrixPlus(ScalaProduct(mB,beta1), ScalaProduct(gB,1.0-beta1));
        vB = matrixPlus(ScalaProduct(vB,beta2), ScalaProduct(HadamardProduct(gB,gB),1.0-beta2));

        double bc1 = 1.0 - std::pow(beta1, t);
        double bc2 = 1.0 - std::pow(beta2, t);
        d_matrix<double> mW_hat = ScalaProduct(mW, 1.0/bc1);
        d_matrix<double> vW_hat = ScalaProduct(vW, 1.0/bc2);
        d_matrix<double> mB_hat = ScalaProduct(mB, 1.0/bc1);
        d_matrix<double> vB_hat = ScalaProduct(vB, 1.0/bc2);

        auto denomW = ScalaPlus(MatrixActivate<double, sqr>(vW_hat), eps);
        auto denomB = ScalaPlus(MatrixActivate<double, sqr>(vB_hat), eps);
        auto invW   = MatrixActivate<double, devide>(denomW);
        auto invB   = MatrixActivate<double, devide>(denomB);

        W = matrixPlus(W, ScalaProduct(HadamardProduct(mW_hat, invW), -lr));
        B = matrixPlus(B, ScalaProduct(HadamardProduct(mB_hat, invB), -lr));

        cudaDeviceSynchronize();
    }
    double getLR() override { return lr; }
};

//-------------------------------------------------------------
// PerceptronLayer using external Optimizer
//-------------------------------------------------------------
class perceptronLayer {
protected:
    int inputSize;
    int outputSize;
    d_matrix<double> input;
    d_matrix<double> weight;
    d_matrix<double> bias;
    d_matrix<double> output;
    d_matrix<double> delta;
    d_matrix<double> Gt_I;
    d_matrix<double> gradW;
    d_matrix<double> gradB;
    Optimizer* opt;
public:
    perceptronLayer(int i, int o, Optimizer* optimizer, InitType init)
        : inputSize(i), outputSize(o),
          input(i,1), weight(o,i), bias(o,1),
          output(o,1), delta(o,1), gradW(o,i), gradB(o,1), opt(optimizer), Gt_I(i, 1) {
        weight = InitWeight<double>(o,i,init);
        bias.fill(0.01);

        weight.cpyToDev();
        bias.cpyToDev();
    }

    void feedforward(const d_matrix<double>& in) {
        input = in;
        output = matrixPlus(matrixMP(weight, input), bias);
        cudaDeviceSynchronize();
    }

    void calcGrad(perceptronLayer* next, const d_matrix<double>& ext_delta, const d_matrix<double>& act_deriv) {
        Gt_I = ext_delta;
        if(next != nullptr) {
            auto wd = matrixMP(next->weight.transpose(), next->delta);
            Gt_I = wd;
        }
        delta = HadamardProduct(Gt_I, act_deriv);
        gradW = matrixMP(delta, input.transpose());
        gradB = delta;
        cudaDeviceSynchronize();
    }

    d_matrix<double> backprop(perceptronLayer* next, const d_matrix<double>& ext_delta, const d_matrix<double>& act_deriv) {
        calcGrad(next, ext_delta, act_deriv);
        opt->update(weight, bias, gradW, gradB);
        d_matrix<double> gradInput = matrixMP(weight.transpose(), delta);
        return gradInput;
    }

    d_matrix<double>& getOutput(){ return output; }
};

//-------------------------------------------------------------
// ConvolutionLayer
//-------------------------------------------------------------
class convolutionLayer {
    int inRow, inCol;           // 입력 크기
    int fRow, fCol;             // 필터 크기
    int stride;
    d_matrix<double> input;     // 입력 저장
    d_matrix<double> filter;    
    d_matrix<double> bias;
    d_matrix<double> output;    
    d_matrix<double> delta;     // 다음 레이어로부터 받은 δ
    d_matrix<double> gFilter;   // ∂L/∂W
    d_matrix<double> gBias;     // ∂L/∂b
    Optimizer* opt;

public:
    convolutionLayer(int iR, int iC,
                     int fR, int fC, int st,
                     Optimizer* optimizer,
                     InitType init)
      : inRow(iR), inCol(iC),
        fRow(fR), fCol(fC),
        stride(st),
        input(iR, iC),
        filter(fR, fC),
        bias(1, 1),
        output((iR - fR) / stride + 1,
               (iC - fC) / stride + 1),
        delta(output.getRow(), output.getCol()),
        gFilter(fR, fC),
        gBias(1, 1),
        opt(optimizer)
    {
        filter = InitWeight<double>(fR, fC, init);
        bias.fill(0.01);
    }

    void feedforward(const d_matrix<double>& in) {
        input = in;
        output = convolute<double>(input, filter, stride);
        output = ScalaPlus(output, bias(0, 0));
        cudaDeviceSynchronize();
    }

    d_matrix<double> backprop(const d_matrix<double>& ext_delta) {
        delta = ext_delta;
        int outR = output.getRow();
        int outC = output.getCol();
    
        // 1) 필터 그래디언트
        gFilter.fill(0.0);
        gFilter = convolute<double>(input, delta, stride);
    
        // 2) 바이어스 그래디언트
        gBias(0,0) = plusAllElements(delta);
    
        // 3) W, B 동시 업데이트
        opt->update(filter, bias, gFilter, gBias);
        filter.cpyToDev();
        bias.cpyToDev();

        int dR = outR + (outR-1)*(stride-1);
        int dC = outC + (outC-1)*(stride-1);
        d_matrix<double> delta_dilated(dR, dC);
        delta_dilated.fill(0.0);
        for(int i=0; i<outR; ++i)
          for(int j=0; j<outC; ++j)
            delta_dilated(i*stride, j*stride) = delta(i,j);

        delta_dilated.cpyToDev();
        // 4) 입력 그래디언트 δ_prev 계산
        auto filter_rot = filter.rotated180();
        auto delta_full  = zeroPedding<double>(delta_dilated, fRow - 1);
        d_matrix<double> delta_prev = convolute<double>(delta_full, filter_rot, 1);
    
        return delta_prev;
    }

    d_matrix<double>& getOutput(){ return output; }
};

class MultiConvLayer {
private:
    int inH, inW;           // 입력 높이·너비
    int fH, fW;             // 필터 높이·너비
    int stride;
    int outCh;              // 필터(출력 채널) 수

    d_matrix<double> input;      // 입력 feature map

    std::vector<d_matrix<double>> filters;  // 각 필터
    std::vector<d_matrix<double>> biases;   // 각 채널 바이어스
    std::vector<d_matrix<double>> outputs;  // 순전파 출력(feature maps)
    std::vector<d_matrix<double>> deltas;   // 역전파용 델타
    std::vector<d_matrix<double>> gFilters; // 필터 그래디언트
    std::vector<d_matrix<double>> gBiases;  // 바이어스 그래디언트

    Optimizer* opt;

public:
    MultiConvLayer(int inH_, int inW_,
                   int fH_, int fW_,
                   int stride_,
                   int outChannels,
                   Optimizer* optimizer,
                   InitType init)
      : inH(inH_), inW(inW_), fH(fH_), fW(fW_),
        stride(stride_), outCh(outChannels),
        input(inH_, inW_),
        filters(outCh, d_matrix<double>(fH_, fW_)),
        biases(outCh, d_matrix<double>(1, 1)),
        outputs(outCh, d_matrix<double>((inH_-fH_)/stride_+1, (inW_-fW_)/stride_+1)),
        deltas(outputs),
        gFilters(outCh, d_matrix<double>(fH_, fW_)),
        gBiases(outCh, d_matrix<double>(1, 1)),
        opt(optimizer)
    {
        for(int k = 0; k < outCh; ++k) {
            filters[k] = InitWeight<double>(fH, fW, init);
            biases[k].fill(0.01);
            gBiases[k].fill(0.0);
        }
    }

    // 순전파
    void feedforward(const d_matrix<double>& in) {
        input = in;
        input.cpyToDev();

        for(int k = 0; k < outCh; ++k) {
            // 합성곱
            outputs[k] = convolute<double>(input, filters[k], stride);
            // 바이어스 브로드캐스트
            outputs[k] = ScalaPlus<double>(outputs[k], biases[k].operator()(0, 0));
        }
    }

    // 역전파: 외부 델타 전달
    d_matrix<double> backprop(const std::vector<d_matrix<double>>& ext_delta) {
        // 델타 설정
        deltas = ext_delta;

        // 그래디언트 계산
        for(int k = 0; k < outCh; ++k) {
            const auto& deltaK = deltas[k];
            int outH = deltaK.getRow(), outW = deltaK.getCol();

            // 필터 그래디언트
            gFilters[k].fill(0.0);
            gFilters[k] = convolute<double>(input, deltaK, stride);
            // 바이어스 그래디언트
            gBiases[k].operator()(0, 0) = plusAllElements(deltaK);
        }

        // 파라미터 업데이트
        for(int k = 0; k < outCh; ++k) {
            // biases는 스칼라라 직접 업데이트
            opt->update(filters[k],biases[k] ,gFilters[k], gBiases[k]);
            biases[k].operator()(0, 0) -= opt->getLR() * gBiases[k].operator()(0, 0);
        }

        // 이전 레이어 델타 계산 (입력 채널이 1개라 가정)
        d_matrix<double> delta_prev(inH, inW);
        delta_prev.fill(0.0);
        for(int k = 0; k < outCh; ++k) {
            auto rot = filters[k].rotated180();
            auto padded = zeroPedding<double>(deltas[k], fH-1);
            auto dp = convolute<double>(padded, rot, 1);
            delta_prev = matrixPlus<double>(delta_prev, dp);
        }
        delta_prev.cpyToHost();
        return delta_prev;
    }

    // 출력 맵 반환
    const std::vector<d_matrix<double>>& getOutputs() const {
        return outputs;
    }
};

//-------------------------------------------------------------
// Simple Batch Normalization Layer
//-------------------------------------------------------------
class batchNormLayer {
private:
    d_matrix<double> gamma;
    d_matrix<double> beta;
    d_matrix<double> input;
    d_matrix<double> norm;
    d_matrix<double> output;
    d_matrix<double> delta;
    d_matrix<double> gGamma;
    d_matrix<double> gBeta;
    double eps;
    Optimizer* opt;
    double invStd;
public:
    batchNormLayer(int size, Optimizer* optimizer, double eps_=1e-5)
        : gamma(size,1), beta(size,1), input(size,1), norm(size,1),
          output(size,1), delta(size,1), gGamma(size,1), gBeta(size,1),
          eps(eps_), opt(optimizer), invStd(1.0) {
        gamma.fill(1.0);
        beta.fill(0.0);
    }

    void feedforward(const d_matrix<double>& in) {
        input = in;
        int N = input.getRow();
        double mean=0.0;
        for(int i=0;i<N;i++) mean += input(i,0);
        mean /= N;
        double var=0.0;
        for(int i=0;i<N;i++) {
            double d = input(i,0)-mean;
            var += d*d;
        }
        var /= N;
        invStd = 1.0 / std::sqrt(var + eps);
        for(int i=0;i<N;i++) {
            norm(i,0) = (input(i,0)-mean)*invStd;
            output(i,0) = gamma(i,0)*norm(i,0) + beta(i,0);
        }
    }

    void backprop(const d_matrix<double>& ext_delta) {
        delta = ext_delta;
        int N = delta.getRow();
        for(int i=0;i<N;i++) {
            gGamma(i,0) = delta(i,0) * norm(i,0);
            gBeta(i,0)  = delta(i,0);
        }
        opt->update(gamma, beta, gGamma, gBeta);
        gamma.cpyToDev();
        beta.cpyToDev();
    }

    d_matrix<double>& getOutput(){ return output; }
};

// ActivateLayer-------------------------------------------------------------------------------------------------------------------

// 활성화 계층
// 사용법: pushInput()으로 입력, Active()로 활성화 적용, getOutput()으로 결과 반환
// 지원: ReLU, LReLU, Identity, Sigmoid
// d_Active: 미분값 반환
class ActivateLayer{
    private:
        ActivationType act;
        d_matrix<double> input;
        d_matrix<double> output;
    public:
        // 생성자: 행, 열, 활성화 종류 지정
        ActivateLayer(int row, int col, ActivationType a) : input(row, col), output(row, col), act(a){}
        // 입력 설정
        void pushInput(const d_matrix<double>& in);
        // 활성화 적용 (output = f(input))
        void Active();
        // 활성화 미분값 반환 (f'(z))
        d_matrix<double> d_Active(const d_matrix<double>& z);
        // 결과 반환
        const d_matrix<double>& getOutput() const ;
};

// LossLayer--------------------------------------------------------------------------------------------------------------------------------

// 손실 계층
// 사용법: pushTarget, pushOutput으로 데이터 입력 후 getLoss(), getGrad() 호출
// 지원: MSE(평균제곱오차), CrossEntropy(크로스엔트로피)
// getLoss: loss 반환, getGrad: dL/dz 반환
class LossLayer{
    private:
        d_matrix<double> target;
        d_matrix<double> output;
        LossType Loss;
    public:
        // 생성자: 행, 열, 손실 종류 지정
        LossLayer(int row, int col, LossType L) : target(row, col), output(row, col), Loss(L){}
        // 타겟/출력 입력
        void pushTarget(const d_matrix<double>& Target);
        void pushOutput(const d_matrix<double>& Output);
        // 손실값 반환
        double getLoss();
        // 손실 미분 반환
        d_matrix<double> getGrad();
};

#endif