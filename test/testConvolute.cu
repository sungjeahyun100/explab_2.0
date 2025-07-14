#include "perceptronVer2.hpp"
#include "utility.hpp"


/*
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
*/


class MINSTsolver_SGD{
    private:
        SGD opt;
        ConvLayer c1_input; ActivateLayer c1_inputAct;
        perceptronLayer input; ActivateLayer inputAct;
        perceptronLayer hidden1; ActivateLayer hidden1Act;
        perceptronLayer output; ActivateLayer outputAct;
        LossLayer loss;

        std::vector<std::pair<mat,uint8_t>> train_set;
        std::vector<std::pair<mat,uint8_t>> test_set;
    public:
        MINSTsolver_SGD(int c1_inRow, int c1_inCol, int c1_kRow, int c1_kCol, int c1_st, int c1_outChannel, int c1_output, int h1, int h2, int p_output) : 
        opt(0.0001),
        c1_input(c1_inRow, c1_inCol, c1_kRow, c1_kCol, c1_st, c1_outChannel, &opt, InitType::Xavier), c1_inputAct(c1_output, 1, ActivationType::Tanh),
        input(c1_output, h1, &opt, InitType::Xavier), inputAct(h1, 1, ActivationType::Tanh),
        hidden1(h1, h2, &opt, InitType::Xavier), hidden1Act(h2, 1, ActivationType::Tanh),
        output(h2, p_output, &opt, InitType::Xavier), outputAct(p_output, 1, ActivationType::Tanh),
        loss(p_output, 1, LossType::MSE) {
            auto imgs = load_mnist_images("../test/train-images-idx3-ubyte");
            auto lbls = load_mnist_labels("../test/train-labels-idx1-ubyte");
            auto tr2 = load_mnist_images("../test/t10k-images-idx3-ubyte");
            auto la2 = load_mnist_labels("../test/t10k-labels-idx1-ubyte");
            train_set.reserve(imgs.size());
            for (size_t i = 0; i < imgs.size(); ++i) {
                train_set.emplace_back(std::move(imgs[i]), lbls[i]);
            }
            for(int j = 0; j < tr2.size(); j++){
                test_set.emplace_back(std::move(tr2[j]), la2[j]);
            }
        }

        std::pair<mat, double> forward(const mat& in, const mat& target){
        }

        void train(int epochs, int batchSize = 64){

        }
};

/*
class MINSTsolver_Adam{};
*/


int main() {
    return 0;
}


