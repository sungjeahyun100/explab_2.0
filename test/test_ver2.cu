#include <ver2/perceptron_2.hpp>
#include <chrono>

void printProgressBar(int current, int total, std::chrono::steady_clock::time_point startTime, std::string processname) {
    int width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(width * progress);
    
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    int elapsedSec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% ";
    std::cout << '[' << processname << ']';
    std::cout << "(경과 시간: " << elapsedSec << " 초)\r";
    std::cout.flush();
}

namespace p2 = perceptron_2;

class XORsolver_Adam{
    private:
        p2::Adam opt1;
        p2::Adam opt2;
        p2::Adam opt3;
        p2::PerceptronLayer input_layer;
        p2::ActivateLayer inAct;
        p2::PerceptronLayer hidden1_layer;
        p2::ActivateLayer hidden1Act;
        p2::PerceptronLayer output_layer;
        p2::ActivateLayer outputAct;
        p2::LossLayer loss;

        std::vector<d2::d_matrix_2<double>> XORdata;
        std::vector<d2::d_matrix_2<double>> XORtarget;

    public:
        XORsolver_Adam(int in, int h1, int h2, int out, d2::InitType init, p2::LossType l, p2::ActType act, double lr) : 
        opt1(h1, in, lr), opt2(h2, h1, lr), opt3(out, h2, lr),
        input_layer(in, h1, &opt1, init),
        inAct(h1, 1, act), 
        hidden1_layer(h1, h2, &opt2, init),
        hidden1Act(h2, 1, act), 
        output_layer(h2, out, &opt3, init),
        outputAct(out, 1, act), loss(out, 1, l)
        {
            XORtarget.resize(4);
            XORtarget[1](0, 0) = 1;
            XORtarget[2](0, 0) = 1;

            XORdata.resize(4);
            XORdata[0].resize(2, 1);
            XORdata[0](0, 0) = 0;
            XORdata[0](1, 0) = 0;
            XORdata[1].resize(2, 1);
            XORdata[1](0, 0) = 1;
            XORdata[1](1, 0) = 0;
            XORdata[2].resize(2, 1);
            XORdata[2](0, 0) = 0;
            XORdata[2](1, 0) = 1;
            XORdata[3].resize(2, 1);
            XORdata[3](0, 0) = 1;
            XORdata[3](1, 0) = 1;

            for(int i = 0; i < 4; i++){
                XORdata[i].cpyToDev();
                XORtarget[i].cpyToDev();
            }
        }
        
        std::pair<d2::d_matrix_2<double>, double> forward(const d2::d_matrix_2<double> train_input, const d2::d_matrix_2<double> target){
            input_layer.feedforward(train_input);
            inAct.pushInput(input_layer.getOutput()); inAct.Active();
            hidden1_layer.feedforward(inAct.getOutput());
            hidden1Act.pushInput(hidden1_layer.getOutput()); hidden1Act.Active();
            output_layer.feedforward(hidden1Act.getOutput());
            outputAct.pushInput(output_layer.getOutput()); outputAct.Active();

            loss.pushOutput(outputAct.getOutput());
            loss.pushTarget(target);

            return {outputAct.getOutput(), loss.getLoss()};
        }
        
        void backprop(int epochs){
            auto start = std::chrono::steady_clock::now();
            for(int epoch = 1; epoch <= epochs; epoch++){
                double totalLoss = 0;
                double avgloss = 0;
                for(int i = 0; i < 4; i++){
                    auto [out, l] = forward(XORdata[i], XORtarget[i]);
    
                    output_layer.backprop(nullptr, loss.getGrad(), outputAct.d_Active(output_layer.getOutput()));
                    hidden1_layer.backprop(&output_layer, {}, hidden1Act.d_Active(hidden1_layer.getOutput()));
                    input_layer.backprop(&hidden1_layer, {}, inAct.d_Active(input_layer.getOutput()));
    
                    totalLoss += l; 
                }
                avgloss = totalLoss/4.0l;
                printProgressBar(epoch, epochs, start, "avgloss:" + std::to_string(avgloss));
            }
            std::cout << std::endl;
            std::cout << "all process done in " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count() << "ms" << std::endl;

            double totalLoss = 0;
            double avgloss = 0;
            for(int i = 0; i < 4; i++){
                auto [out, l] = forward(XORdata[i], XORtarget[i]);
                std::cout << i << "번 입력 ";
                XORdata[i].printMatrix();
                std::cout << i << "번 입력 행렬의 연산 결과 ";
                out.printMatrix();
                std::cout << i << "번 타겟 ";
                XORtarget[i].printMatrix();
                std::cout << "loss:" << l << std::endl;
                totalLoss += l; 
            }
            avgloss = totalLoss/4.0l;
            std::cout << "평균 로스:" << avgloss << std::endl;
        }

};

int main(){
    XORsolver_Adam test(2, 4, 4, 1, d2::InitType::He, p2::LossType::MSE, p2::ActType::LReLU, 0.0001);
    test.backprop(5000);
    return 0;
}
