#include <ver2/perceptron_2.hpp>
#include <ver2/utility.hpp>

namespace p2 = perceptron_2;


class mnist_solver_adam {
private:
    int batch_size;

    // --- 합성곱 층들 ---
    p2::Adam conv1_opt;      p2::ActivateLayer conv1Act;
    p2::convLayer      conv1;
    p2::Adam conv2_opt;      p2::ActivateLayer conv2Act;   
    p2::convLayer      conv2;

    // --- 완전 연결(FC) 층들 ---
    p2::Adam fc1_opt;        p2::ActivateLayer fc1Act;
    p2::PerceptronLayer fc1;       
    p2::Adam fc2_opt;        p2::ActivateLayer fc2Act;
    p2::PerceptronLayer fc2;       

    // --- 손실층 ---
    p2::LossLayer       loss;

public:
    mnist_solver_adam(int bs)
    : batch_size(bs),

      // conv1: N=bs, C=1, H=W=28, K=8, R=S=3, pad=1, stride=1
      conv1_opt(8, 1*3*3, 0.001),
      conv1Act(bs, 8, p2::ActType::ReLU),
      conv1(bs,1,28,28,  8,3,3, 1,1, 1,1, &conv1_opt),

      // conv2: N=bs, C=8, H=W=28, K=16, R=S=3, pad=1, stride=2 → out:16×14×14
      conv2_opt(16,8*3*3, 0.001),
      conv2Act(bs, 16, p2::ActType::ReLU),
      conv2(bs,8,28,28, 16,3,3, 1,1, 2,2, &conv2_opt),

      // fc1: 16*14*14 → 128
      fc1_opt(128,16*14*14, 0.001),
      fc1Act(bs, 128, p2::ActType::ReLU),
      fc1(bs, 16*14*14, 128, &fc1_opt, d2::InitType::He),

      // fc2: 128 → 10 (클래스 개수)
      fc2_opt(10, 128, 0.001),
      fc2Act(bs, 10, p2::ActType::Identity),
      fc2(bs, 128, 10, &fc2_opt, d2::InitType::He),

      // 크로스엔트로피 손실
      loss(bs, 10, p2::LossType::CrossEntropy)
    {}

    // 한 배치에 대한 순전파
    d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& X) {
        // conv1
        conv1.forward(X);
        conv1Act.pushInput(conv1.getOutput()); conv1Act.Active();

        // conv2
        conv2.forward(conv1Act.getOutput());
        conv2Act.pushInput(conv2.getOutput()); conv2Act.Active();

        // 평탄화 → fc1
        auto flat = conv2Act.getOutput().reshape(16*14*14, batch_size);
        fc1.feedforward(flat);
        fc1Act.pushInput(fc1.getOutput()); fc1Act.Active();

        // fc2
        fc2.feedforward(fc1Act.getOutput());
        fc2Act.pushInput(fc2.getOutput()); fc2Act.Active();

        return fc2Act.getOutput();
    }

    // 학습 루프
    void train(const d2::d_matrix_2<double>& X, const d2::d_matrix_2<double>& Y, int epochs) {
        d2::d_matrix_2<double> dummy;
        auto start = std::chrono::steady_clock::now();
        for(int e = 1; e <= epochs; ++e) {
            // 순전파
            auto Ypred = forward(X);

            // 손실 계산
            loss.pushOutput(Ypred);
            loss.pushTarget(Y);
            double L = loss.getLoss();

            // 역전파: FC 층들
            auto grad2 = loss.getGrad();  // dL/dz_fc2
            fc2.backprop(nullptr, grad2, fc2Act.d_Active(fc2.getOutput()));
            fc1.backprop(&fc2, dummy, fc1Act.d_Active(fc1.getOutput()));

            auto dy2 = conv2.backward(&fc1, dummy, conv2Act.d_Active(conv2.getOutput()));
            auto dy1 = conv1.backward(nullptr, dy2, conv1Act.d_Active(conv1.getOutput()));

            // 진행 상황 표시
            printProgressBar(e, epochs, start, "loss:" + std::to_string(L));
        }
        std::cout << std::endl;
        std::cout << "총 학습 시간: "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - start
                     ).count() << "초\n";
    }
};

int main(){
    constexpr int BATCH  = 50;
    constexpr int EPOCHS = 10;

    // MNIST 데이터 로드
    auto X = load_images_matrix("../test/train-images-idx3-ubyte");
    auto Y = load_labels_matrix("../test/train-labels-idx1-ubyte", 10);

    mnist_solver_adam solver(BATCH);
    solver.train(X, Y, EPOCHS);

    return 0;
}

