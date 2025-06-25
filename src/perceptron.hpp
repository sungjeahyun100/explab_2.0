#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <sstream>
#include "d_matrix.hpp"

const std::string WEIGHT_DATAPATH = "../test_subject/";

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

enum class LossType{
    MSE,
    CrossEntropy
};

 
inline std::string getCurrentTimestamp();

class convolutionLayer{
    private:
        int inRow, inCol;
        int kRow, kCol;
        int numFilter;
        int outRow, outCol;
        double learning_rate;
        d_matrix<double> input;
        std::vector<d_matrix<double>> kernels;
        d_matrix<double> bias; // numFilter x 1
        std::vector<d_matrix<double>> outputs;
        d_matrix<double> flatOutput;
    public:
        convolutionLayer(int iRow, int iCol, int fRow, int fCol,
                          int nFilter, double lr, InitType init);
        void feedforward(const d_matrix<double>& raw_input);
        void backprop(const d_matrix<double>& delta_flat);
        d_matrix<double>& getOutput();
};

class perceptronLayer {
    protected:
        int inputSize;
        int outputSize;
        double learning_rate;
        d_matrix<double> input;
        d_matrix<double> weight;
        d_matrix<double> bias;
        d_matrix<double> z;
        d_matrix<double> delta;
        d_matrix<double> output;
        d_matrix<double> Gt_W, Gt_B;
    public:
        // 생성자: 입력/출력 크기, 학습률, 가중치 초기화 방식 지정
        perceptronLayer(int i, int o, double l, InitType init)
        : inputSize(i), outputSize(o),
          input(i, 1), weight(o, i), bias(o, 1),
          z(o, 1), delta(o, 1), output(o, 1),
          Gt_W(o, i), Gt_B(o, 1), learning_rate(l)
        {
            weight = InitWeight<double>(o, i, init);
            bias.fill(0.01);
        }
    
        // feedforward: z = W x + b, output = z
        void feedforward(const d_matrix<double>& raw_input);
        // backprop: 가상함수, 파생 클래스에서 구현
        virtual void backprop(perceptronLayer* next, const d_matrix<double>& external_delta, const d_matrix<double>& act_deriv) = 0;
        // calculateGrad: 델타, 그래디언트 계산
        void calculateGrad(perceptronLayer* next, const d_matrix<double>& external_delta, const d_matrix<double>& act_deriv);
        // updateWeightInDev: 가중치/바이어스 GPU로 복사
        void updateWeightInDev();
        // getOutput: 계층 출력 반환
        d_matrix<double>& getOutput();
        // saveWeight/loadWeight: 가중치 저장/불러오기
        void saveWeight();
        void loadWeight(const std::string& path);
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


// Adam과 SGD----------------------------------------------------------------------------------------------------------------------------------------------------

class Adam : public perceptronLayer
{
private:
    d_matrix<double> m_W, v_W, m_B, v_B;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int t = 0;

public:
    Adam(int i, int o, double lr, InitType Init) : perceptronLayer(i, o, lr, Init), m_W(o, i), v_W(o, i), m_B(o, 1), v_B(o, 1) {
        m_W.fill(0.00l);
        v_W.fill(0.00l);
        m_B.fill(0.00l);
        v_B.fill(0.00l);
    }

    virtual ~Adam();

    void backprop(perceptronLayer* next, const d_matrix<double>& external_delta, const d_matrix<double>& act_deriv);   
};


class SGD : public perceptronLayer
{
public:
    SGD(int i, int o, double lr, InitType Init) : perceptronLayer(i, o, lr, Init) {}

    virtual ~SGD();

    void backprop(perceptronLayer* next, const d_matrix<double>& external_delta, const d_matrix<double>& act_deriv);
};


#endif

