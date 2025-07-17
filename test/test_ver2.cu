#include <ver2/perceptron_2.hpp>
#include <chrono>
#include <ver2/utility.hpp>

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

        d2::d_matrix_2<double> XORdata = 
        {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        d2::d_matrix_2<double> XORtarget = 
        {
            {0},
            {1},
            {1},
            {0}
        };

    public:
        XORsolver_Adam(int in, int h1, int h2, int out, d2::InitType init, p2::LossType l, p2::ActType act, double lr, int n) : 
        opt1(h1, in, lr), opt2(h2, h1, lr), opt3(out, h2, lr),
        input_layer(n, in, h1, &opt1, init),
        inAct(n, h1, act), 
        hidden1_layer(n, h1, h2, &opt2, init),
        hidden1Act(n, h2, act), 
        output_layer(n, h2, out, &opt3, init),
        outputAct(n, out, act), loss(n, out, l)
        {}
        
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
                double avgloss = 0;
                auto [out, l] = forward(XORdata, XORtarget);

                output_layer.backprop(nullptr, loss.getGrad(), outputAct.d_Active(output_layer.getOutput()));
                hidden1_layer.backprop(&output_layer, {}, hidden1Act.d_Active(hidden1_layer.getOutput()));
                input_layer.backprop(&hidden1_layer, {}, inAct.d_Active(input_layer.getOutput()));

                avgloss = l;
                printProgressBar(epoch, epochs, start, "avgloss:" + std::to_string(avgloss));
            }
            std::cout << std::endl;
            std::cout << "all process done in " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count() << "ms" << std::endl;

            auto [out, l] = forward(XORdata, XORtarget);
            std::cout << "타겟:";
            XORtarget.printMatrix();
            std::cout << "출력:";
            out.printMatrix();
            std::cout << "평균 로스:" << l << std::endl;
        }

};

int main(){
    XORsolver_Adam test(2, 4, 4, 1, d2::InitType::Xavier, p2::LossType::CrossEntropy, p2::ActType::Tanh, 0.0001, 4);
    test.backprop(10000);
    return 0;
}


