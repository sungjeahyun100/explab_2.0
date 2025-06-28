#include "perceptronVer2.hpp"
#include "utility.hpp"
using mat = d_matrix<double>;


class XORsolver_SGD{
    private:
        SGD opt;
        perceptronLayer inputlayer;
        ActivateLayer inputAct;
        perceptronLayer hiddenlayer1;
        ActivateLayer hidden1Act;
        perceptronLayer outputlayer;
        ActivateLayer outputAct;
        LossLayer loss;

        //XOR dataset
        std::vector<mat> X = {
            mat({{0},{0}}), mat({{0},{1}}),
            mat({{1},{0}}), mat({{1},{1}})
        };
        std::vector<mat> Y = {
            mat({{0}}), mat({{1}}),
            mat({{1}}), mat({{0}})
        };
    public:
        XORsolver_SGD(int in, int h1, int h2, int out):inputlayer(in, h1, &opt, InitType::He), inputAct(h1, 1, ActivationType::LReLU),
                                                           hiddenlayer1(h1, h2, &opt, InitType::He), hidden1Act(h2, 1, ActivationType::LReLU),
                                                           outputlayer(h2, out, &opt, InitType::He), outputAct(out, 1, ActivationType::LReLU),
                                                           loss(out, 1, LossType::MSE), opt(0.0001) {}
        std::pair<d_matrix<double>, double> forward(d_matrix<double>& input, d_matrix<double>& target){
            inputlayer.feedforward(input); 
            inputAct.pushInput(inputlayer.getOutput()); inputAct.Active();
            hiddenlayer1.feedforward(inputAct.getOutput());
            hidden1Act.pushInput(hiddenlayer1.getOutput()); hidden1Act.Active();
            outputlayer.feedforward(hidden1Act.getOutput());
            outputAct.pushInput(outputlayer.getOutput()); outputAct.Active();
            loss.pushTarget(target); loss.pushOutput(outputAct.getOutput());
            double l = loss.getLoss();

            std::pair<d_matrix<double>, double> result = {outputAct.getOutput(), l};

            return result;
        }

        void training(int epochs){
            mat dummy(1, 1);
            auto startTime = std::chrono::steady_clock::now();
            for(int epoch = 0; epoch < epochs; epoch++){
                int ta = 0;
                double totalLoss = 0;
                for(auto input : X){
                    auto R = forward(input, Y[ta]);

                    totalLoss += R.second;

                    outputlayer.backprop(nullptr, loss.getGrad(), outputAct.d_Active(outputlayer.getOutput()));
                    hiddenlayer1.backprop(&outputlayer, dummy, hidden1Act.d_Active(hiddenlayer1.getOutput()));
                    inputlayer.backprop(&hiddenlayer1, dummy, inputAct.d_Active(inputlayer.getOutput()));
                    ta++;
                }
                double average = totalLoss/4;
                std::string progress = "Epoch" + std::to_string(epoch) + " loss:" + std::to_string(average);
                printProgressBar(epoch, epochs, startTime, progress);
            }
            std::cout << "                                                                                                                       " << std::endl;
            std::cout << "[Done] training complete." << std::endl;
            auto totalElapsed = std::chrono::steady_clock::now() - startTime;
            int totalSec = std::chrono::duration_cast<std::chrono::seconds>(totalElapsed).count();
            std::cout << "총 실행 시간: " << totalSec << " 초" << std::endl;
            std::cout << "결과물:" << std::endl;
            int ta = 0;
            for(auto input : X){
                auto R = forward(input, Y[ta]);
                std::cout << "input" << ta << ":" << input << "기대:" << Y[ta] << "실제:" << R.first << "로스율(MSE):" << R.second << std::endl;
                ta++;
            }
        }
};


class XORsolver_Adam {
private:
    perceptronLayer inputlayer;
    Adam inputOpt;
    ActivateLayer inputAct;

    perceptronLayer hiddenlayer1;
    Adam hidden1Opt;
    ActivateLayer hidden1Act;

    perceptronLayer outputlayer;
    Adam outputOpt;
    ActivateLayer outputAct;

    LossLayer loss;
    std::vector<mat> X;
    std::vector<mat> Y;

public:
    XORsolver_Adam(int in_dim, int h1_dim, int h2_dim, int out_dim, double lr = 0.001)
      : inputOpt(lr),
        inputlayer(in_dim, h1_dim, &inputOpt, InitType::He), inputAct(h1_dim, 1, ActivationType::LReLU),
        hidden1Opt(lr),
        hiddenlayer1(h1_dim, h2_dim, &hidden1Opt, InitType::He), hidden1Act(h2_dim, 1, ActivationType::LReLU),
        outputOpt(lr),
        outputlayer(h2_dim, out_dim, &outputOpt, InitType::He), outputAct(out_dim, 1, ActivationType::LReLU),
        loss(out_dim, 1, LossType::MSE),
        X({ mat({{0},{0}}), mat({{0},{1}}), mat({{1},{0}}), mat({{1},{1}}) }),
        Y({ mat({{0}}),       mat({{1}}),       mat({{1}}),       mat({{0}})       })
    {}

    // Forward pass: returns pair(output, loss)
    std::pair<mat, double> forward(const mat& input, const mat& target) {
        inputlayer.feedforward(input);
        inputAct.pushInput(inputlayer.getOutput()); inputAct.Active();

        hiddenlayer1.feedforward(inputAct.getOutput());
        hidden1Act.pushInput(hiddenlayer1.getOutput()); hidden1Act.Active();

        outputlayer.feedforward(hidden1Act.getOutput());
        outputAct.pushInput(outputlayer.getOutput()); outputAct.Active();

        loss.pushTarget(target);
        loss.pushOutput(outputAct.getOutput());
        double l = loss.getLoss();

        return { outputAct.getOutput(), l };
    }

    // Training loop
    void training(int epochs) {
        auto startTime = std::chrono::steady_clock::now();
        int samples = static_cast<int>(X.size());
        mat dummy(1, 1);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            for (int i = 0; i < samples; ++i) {
                auto [out, l] = forward(X[i], Y[i]);
                totalLoss += l;

                // Backpropagation
                outputlayer.backprop(nullptr,           loss.getGrad(),             outputAct.d_Active(outputlayer.getOutput()));
                hiddenlayer1.backprop(&outputlayer,    dummy ,      hidden1Act.d_Active(hiddenlayer1.getOutput()));
                inputlayer.backprop(&hiddenlayer1,     dummy ,     inputAct.d_Active(inputlayer.getOutput()));
            }

            double avgLoss = totalLoss / samples;
            std::string progress = "Epoch " + std::to_string(epoch) + " loss: " + std::to_string(avgLoss);
            printProgressBar(epoch, epochs, startTime, progress);
        }

        std::cout << "[Done] training complete." << std::endl;
        auto totalElapsed = std::chrono::steady_clock::now() - startTime;
        std::cout << "총 실행 시간: "
                  << std::chrono::duration_cast<std::chrono::seconds>(totalElapsed).count()
                  << " 초" << std::endl;

        std::cout << "결과물:" << std::endl;
        for (int i = 0; i < static_cast<int>(X.size()); ++i) {
            auto [out, l] = forward(X[i], Y[i]);
            std::cout << "input: " << X[i]
                      << " 기대: " << Y[i]
                      << " 실제: " << out
                      << " 로스율(MSE): " << l << std::endl;
        }
    }
};

int main() {
    XORsolver_Adam solver(2, 4, 4, 1);
    solver.training(500);
    return 0;
}


