#include <d_matrix_2.hpp>

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
    
    class Adam : public optimizer {
        double lr, beta1, beta2, eps;
        int t;
        d2::d_matrix_2<double> mW, vW, mB, vB;
    public:
        Adam(int row, int col, double lr_, double b1=0.9, double b2=0.999, double e=1e-8)
            : lr(lr_), beta1(b1), beta2(b2), eps(e), t(0),
              mW(row, col), vW(row, col), mB(row,1), vB(row,1) {
            mW.fill(0.0); vW.fill(0.0); mB.fill(0.0); vB.fill(0.0);
        }
        void update(d2::d_matrix_2<double>& W, d2::d_matrix_2<double>& B, const d2::d_matrix_2<double>& gW, const d2::d_matrix_2<double>& gB) override {
            t++;
            mW = d2::matrixPlus(d2::ScalaProduct(mW,beta1), d2::ScalaProduct(gW,1.0-beta1));
            vW = d2::matrixPlus(d2::ScalaProduct(vW,beta2), d2::ScalaProduct(d2::HadamardProduct(gW,gW),1.0-beta2));
            mB = d2::matrixPlus(d2::ScalaProduct(mB,beta1), d2::ScalaProduct(gB,1.0-beta1));
            vB = d2::matrixPlus(d2::ScalaProduct(vB,beta2), d2::ScalaProduct(d2::HadamardProduct(gB,gB),1.0-beta2));
    
            double bc1 = 1.0 - std::pow(beta1, t);
            double bc2 = 1.0 - std::pow(beta2, t);
            d2::d_matrix_2<double> mW_hat = d2::ScalaProduct(mW, 1.0/bc1);
            d2::d_matrix_2<double> vW_hat = d2::ScalaProduct(vW, 1.0/bc2);
            d2::d_matrix_2<double> mB_hat = d2::ScalaProduct(mB, 1.0/bc1);
            d2::d_matrix_2<double> vB_hat = d2::ScalaProduct(vB, 1.0/bc2);
    
            auto denomW = d2::ScalaPlus(d2::MatrixActivate<double, d2::sqr>(vW_hat), eps);
            auto denomB = d2::ScalaPlus(d2::MatrixActivate<double, d2::sqr>(vB_hat), eps);
            auto invW   = d2::MatrixActivate<double, d2::devide>(denomW);
            auto invB   = d2::MatrixActivate<double, d2::devide>(denomB);
    
            W = d2::matrixPlus(W, d2::ScalaProduct(d2::HadamardProduct(mW_hat, invW), -lr));
            B = d2::matrixPlus(B, d2::ScalaProduct(d2::HadamardProduct(mB_hat, invB), -lr));
        }
    };

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
    public:
        PerceptronLayer(int i, int o, optimizer* optimizer, d2::InitType init)
            : inputSize(i), outputSize(o),
              input(i,1), weight(o,i), bias(o,1),
              output(o,1), delta(o,1), gradW(o,i), gradB(o,1), opt(optimizer) {
            weight = d2::InitWeight<double>(o,i,init);
            bias.fill(0.01);
        }
    
        void feedforward(const d2::d_matrix_2<double>& in) {
            input = in;
            output = d2::matrixPlus(d2::matrixMP(weight, input), bias);
        }
    
        void calcGrad(PerceptronLayer* next, const d2::d_matrix_2<double>& ext_delta, const d2::d_matrix_2<double>& act_deriv) {
            d2::d_matrix_2<double> grad_input = ext_delta;
            if(next != nullptr) {
                auto wd = d2::matrixMP(next->weight.transpose(), next->delta);
                wd.cpyToDev();
                grad_input = wd;
            }
            delta = d2::HadamardProduct(grad_input, act_deriv);
            gradW = d2::matrixMP(delta, input.transpose());
            gradB = delta;
        }
    
        void backprop(PerceptronLayer* next, const d2::d_matrix_2<double>& ext_delta, const d2::d_matrix_2<double>& act_deriv) {
            calcGrad(next, ext_delta, act_deriv);
            opt->update(weight, bias, gradW, gradB);
        }
    
        d2::d_matrix_2<double>& getOutput(){ return output; }
    };

}//namespace perceptron_2

