#include <ver2/perceptron_2.hpp>
#include <ver2/utility.hpp>
#include <cmath>

namespace p2 = perceptron_2;

/*
    python 
import gdb
m = gdb.parse_and_eval("d_A")                               
rows = int(m['row'])                                      
cols = int(m['col'])                                      
for i in range(rows):                                     
    row_vals = []                                         
    for j in range(cols):                                 
        elt = gdb.parse_and_eval(f"d_A.operator()({i},{j})")
        if elt.type.code == gdb.TYPE_CODE_REF:            
            elt = elt.referenced_value()                  
        row_vals.append(str(elt))                         
    print("[" + " ".join(row_vals) + "]")                 
end
    */

/*
    python 
import gdb
m = gdb.parse_and_eval("delta")
v = gdb.parse_and_eval("grad_input")                               
rows = int(m['row'])                                      
cols = int(m['col'])                   
result = bool(True)         
for i in range(rows):                                     
    row_vals = []                                         
    row_vals2 = []
    for j in range(cols):                                 
        elt = gdb.parse_and_eval(f"delta.operator()({i},{j})")
        elt_2 = gdb.parse_and_eval(f"grad_input.operator()({i},{j})")
        if elt.type.code == gdb.TYPE_CODE_REF and elt_2.type.code == gdb.TYPE_CODE_REF:            
            elt = elt.referenced_value()                  
            elt_2 = elt_2.referenced_value()
        row_vals.append(str(elt))                                    
        row_vals2.append(str(elt_2))
    if row_vals != row_vals2:
        result = False
print(result)
end
*/

class mnist_solver_adam {
private:
    int batch_size;

    //모델 cudaStream 설정
    p2::handleStream hs;

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
      conv1_opt(8, 1*3*3, 0.0001, p2::layerType::conv),
      conv1Act(p2::ActType::LReLU),
      conv1(bs,1,28,28,  8,3,3, 1,1, 1,1, &conv1_opt, d2::InitType::He, hs.model_str),

      // conv2: N=bs, C=8, H=W=28, K=16, R=S=3, pad=1, stride=2 → out:16×14×14
      conv2_opt(16, 8*3*3, 0.0001, p2::layerType::conv),
      conv2Act(p2::ActType::LReLU),
      conv2(bs, 8,28,28, 16,3,3, 1,1, 2,2, &conv2_opt, d2::InitType::He, hs.model_str),

      // fc1: 16*14*14 → 128
      fc1_opt(16*14*14, 128, 0.0001),
      fc1Act(p2::ActType::LReLU),
      fc1(bs, 16*14*14, 128, &fc1_opt, d2::InitType::He, hs.model_str),

      // fc2: 128 → 10 (클래스 개수)
      fc2_opt(128, 10, 0.0001),
      fc2Act(p2::ActType::Identity),
      fc2(bs, 128, 10, &fc2_opt, d2::InitType::He, hs.model_str),

      // 크로스엔트로피 손실
      loss(p2::LossType::CrossEntropy)
    {}

    // 한 배치에 대한 순전파
    d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& X, cudaStream_t str) {

        conv1.forward(X, str);
        auto conv_a1 = conv1Act.Active(conv1.getOutput(), str);

        // conv2
        conv2.forward(conv_a1, str);
        auto conv_a2 = conv2Act.Active(conv2.getOutput(), str);

        // fc1
        fc1.feedforward(conv_a2, str);
        auto fc_a1 = fc1Act.Active(fc1.getOutput(), str);

        // fc2
        fc2.feedforward(fc_a1, str);
        fc2Act.Active(fc2.getOutput(), str);


        return fc2Act.Active(fc2.getOutput(), str);
    }

    // 학습 루프
    void train(d2::d_matrix_2<double>& X, d2::d_matrix_2<double>& Y, int epochs) {
        int N = X.getRow();
        int B = batch_size;
        int num_batches = (N + B - 1) / B;
        auto start = std::chrono::steady_clock::now();
        std::vector<d2::d_matrix_2<double>> batch(num_batches), labels(num_batches);
        for(int i = 0; i < num_batches; ++i){
            batch[i] = X.getBatch(B, i*B);
            labels[i] = Y.getBatch(B, i*B);
            printProgressBar(i+1, num_batches, start, "batch loading... (batch " + std::to_string(i+1) + "/" + std::to_string(num_batches) + ")");
        }
        std::cout << std::endl;
        std::cout << "[batch load complete]" << std::endl;
        std::string prograss_avgloss;
        for(int e = 1; e <= epochs; e++) {
            double avgloss = 0;
            for(int j = 0; j < num_batches; j++){
                // 순전파
                auto Ypred = forward(batch[j], hs.model_str);
    
                // 손실 계산
                double L = loss.getLoss(Ypred, labels[j], hs.model_str);

                avgloss += L;
                if(std::isnan(L)){
                    std::cerr << "Loss is NaN at batch " << j+1 << ", epoch " << e << std::endl;
                    std::cerr << "Ypred (first 10 elements): ";
                    Ypred.cpyToHost(); // Ensure host data is valid
                    for(int k=0; k<std::min(10, Ypred.size()); ++k) std::cerr << Ypred.getHostPointer()[k] << " ";
                    std::cerr << std::endl;
                    std::cerr << "labels[j] (first 10 elements): ";
                    labels[j].cpyToHost(); // Ensure host data is valid
                    for(int k=0; k<std::min(10, labels[j].size()); ++k) std::cerr << labels[j].getHostPointer()[k] << " ";
                    std::cerr << std::endl;
                    throw std::runtime_error("invalide error in loss calc.");
                }

                // 역전파: FC 층들
                auto grad2 = loss.getGrad(Ypred, labels[j], hs.model_str);  // dL/dz_fc2
                //grad2.cpyToHost();
                auto fc_dy2 = fc2.backprop(grad2, fc2Act.d_Active(fc2.getOutput(), hs.model_str), hs.model_str);
                auto fc_dy1 = fc1.backprop(fc_dy2, fc1Act.d_Active(fc1.getOutput(), hs.model_str), hs.model_str);
    
                auto dy2 = conv2.backward(fc_dy1, conv2Act.d_Active(conv2.getOutput(), hs.model_str), hs.model_str);
                auto dy1 = conv1.backward(dy2, conv1Act.d_Active(conv1.getOutput(), hs.model_str), hs.model_str);
    
                // 진행 상황 표시
                std::string prograss_batch = "batch" + std::to_string(j+1);
                std::string prograss_loss = "loss:" + std::to_string(L);
                printProgressBar(e, epochs, start, prograss_avgloss + " | " + prograss_batch + " 의 " + prograss_loss);
            }
            avgloss = avgloss/static_cast<double>(num_batches);
            prograss_avgloss = "[epoch" + std::to_string(e+1) + "/" + std::to_string(epochs) + "의 avgloss]:" + std::to_string(avgloss);
        }
        std::cout << std::endl;
        std::cout << "총 학습 시간: "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - start
                     ).count() << "초\n";
    }
};

int main(){
    constexpr int BATCH  = 60;
    constexpr int EPOCHS = 100;

    // MNIST 데이터 로드
    auto X = load_images_matrix("/home/sjh100/바탕화면/explab_ver2/test/train-images-idx3-ubyte");
    auto Y = load_labels_matrix("/home/sjh100/바탕화면/explab_ver2/test/train-labels-idx1-ubyte", 10);

    mnist_solver_adam solver(BATCH);
    solver.train(X, Y, EPOCHS);

    return 0;
}

