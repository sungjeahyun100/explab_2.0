#ifndef PERCEPTRON_VER2_HPP
#define PERCEPTRON_VER2_HPP

#include <cmath>
#include <string>
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
    virtual void update(d_matrix<double>& W,
                        d_matrix<double>& B,
                        const d_matrix<double>& gW,
                        const d_matrix<double>& gB) = 0;
};

class SGDOptimizer : public Optimizer {
    double lr;
public:
    explicit SGDOptimizer(double lr_) : lr(lr_) {}
    void update(d_matrix<double>& W, d_matrix<double>& B,
                const d_matrix<double>& gW, const d_matrix<double>& gB) override {
        W = matrixPlus(W, ScalaProduct(gW, -lr));
        B = matrixPlus(B, ScalaProduct(gB, -lr));
    }
};

class AdamOptimizer : public Optimizer {
    double lr, beta1, beta2, eps;
    int t;
    d_matrix<double> mW, vW, mB, vB;
public:
    AdamOptimizer(int row, int col, double lr_,
                  double b1=0.9, double b2=0.999, double e=1e-8)
        : lr(lr_), beta1(b1), beta2(b2), eps(e), t(0),
          mW(row, col), vW(row, col), mB(row,1), vB(row,1) {
        mW.fill(0.0); vW.fill(0.0); mB.fill(0.0); vB.fill(0.0);
    }
    void update(d_matrix<double>& W, d_matrix<double>& B,
                const d_matrix<double>& gW, const d_matrix<double>& gB) override {
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
    }
};

//-------------------------------------------------------------
// PerceptronLayer using external Optimizer
//-------------------------------------------------------------
class PerceptronLayer {
protected:
    int inputSize;
    int outputSize;
    d_matrix<double> input;
    d_matrix<double> weight;
    d_matrix<double> bias;
    d_matrix<double> output;
    d_matrix<double> delta;
    d_matrix<double> gradW;
    d_matrix<double> gradB;
    Optimizer* opt;
public:
    PerceptronLayer(int i, int o, Optimizer* optimizer, InitType init)
        : inputSize(i), outputSize(o),
          input(i,1), weight(o,i), bias(o,1),
          output(o,1), delta(o,1), gradW(o,i), gradB(o,1), opt(optimizer) {
        weight = InitWeight<double>(o,i,init);
        bias.fill(0.01);
    }

    void feedforward(const d_matrix<double>& in) {
        input = in;
        output = matrixPlus(matrixMP(weight, input), bias);
    }

    void calcGrad(PerceptronLayer* next,
                  const d_matrix<double>& ext_delta,
                  const d_matrix<double>& act_deriv) {
        d_matrix<double> grad_input = ext_delta;
        if(next != nullptr) {
            auto wd = matrixMP(next->weight.transpose(), next->delta);
            wd.cpyToDev();
            grad_input = wd;
        }
        delta = HadamardProduct(grad_input, act_deriv);
        gradW = matrixMP(delta, input.transpose());
        gradB = delta;
    }

    void backprop(PerceptronLayer* next,
                  const d_matrix<double>& ext_delta,
                  const d_matrix<double>& act_deriv) {
        calcGrad(next, ext_delta, act_deriv);
        opt->update(weight, bias, gradW, gradB);
        weight.cpyToDev();
        bias.cpyToDev();
    }

    d_matrix<double>& getOutput(){ return output; }
};

//-------------------------------------------------------------
// ConvolutionLayer with backprop
//-------------------------------------------------------------
class ConvolutionLayer {
    int inRow, inCol;
    int fRow, fCol;
    int stride;
    d_matrix<double> input;
    d_matrix<double> filter;
    d_matrix<double> bias;
    d_matrix<double> output;
    d_matrix<double> delta;
    d_matrix<double> gFilter;
    d_matrix<double> gBias;
    Optimizer* opt;
public:
    ConvolutionLayer(int iRow, int iCol,
                     int fr, int fc, int st,
                     Optimizer* optimizer,
                     InitType init)
        : inRow(iRow), inCol(iCol), fRow(fr), fCol(fc),
          input(iRow, iCol), filter(fr, fc), bias(1,1),
          output(iRow-fr+1, iCol-fc+1), delta(iRow-fr+1, iCol-fc+1),
          gFilter(fr, fc), gBias(1,1), opt(optimizer), stride(st) {
        filter = InitWeight<double>(fr, fc, init);
        bias.fill(0.0);
    }

    void feedforward(const d_matrix<double>& in) {
        input = in;
        output = convolute<double>(input, filter, stride);
        output = matrixPlus(output, bias);
    }

    void backprop(const d_matrix<double>& ext_delta) {
        delta = ext_delta;
        int outRow = output.getRow();
        int outCol = output.getCol();
        // gradient for filter
        gFilter.fill(0.0);
        for(int i=0;i<fRow;i++){
            for(int j=0;j<fCol;j++){
                double sum=0.0;
                for(int r=0;r<outRow;r++){
                    for(int c=0;c<outCol;c++){
                        sum += input(i+r, j+c) * delta(r,c);
                    }
                }
                gFilter(i,j)=sum;
            }
        }
        double bsum=0.0;
        for(int r=0;r<outRow;r++)
            for(int c=0;c<outCol;c++)
                bsum += delta(r,c);
        gBias(0,0)=bsum;
        opt->update(filter, bias, gFilter, gBias);
        filter.cpyToDev();
        bias.cpyToDev();
    }

    d_matrix<double>& getOutput(){ return output; }
};

//-------------------------------------------------------------
// Simple Batch Normalization Layer
//-------------------------------------------------------------
class BatchNormLayer {
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
    BatchNormLayer(int size, Optimizer* optimizer, double eps_=1e-5)
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

#endif // PERCEPTRON_VER2_HPP
