#include <ver2/perceptron_2.hpp>
#include <ver2/utility.hpp>
#include <ver2/GOLdatabase_2.hpp>
#include <chrono>
#include <iostream>

namespace p2 = perceptron_2;

class GOLsolver_1{
    private:
        p2::handleStream hs;
        p2::ActivateLayer act;
        p2::LossLayer loss;

        p2::Adam conv1_opt;
        p2::Adam conv2_opt;
        p2::Adam conv3_opt;

        p2::Adam fc1_opt;
        p2::Adam fc2_opt;
        p2::Adam fc3_opt;
        p2::Adam fc_out_opt;

        p2::convLayer conv1;
        p2::convLayer conv2;
        p2::convLayer conv3;

        p2::PerceptronLayer fc1;
        p2::PerceptronLayer fc2;
        p2::PerceptronLayer fc3;
        p2::PerceptronLayer fc_out;

        int batch;
    public:
        GOLsolver_1(int bs) : batch(bs),
         // Conv layer 최적화기들 (filter수, 입력채널*kernel크기, learning_rate, layerType, stream)
         conv1_opt(8, 1*5*5, 0.00001, p2::layerType::conv, hs.model_str),
         conv2_opt(16, 8*3*3, 0.00001, p2::layerType::conv, hs.model_str), 
         conv3_opt(32, 16*3*3, 0.00001, p2::layerType::conv, hs.model_str),
         
         // FC layer 최적화기들 (출력크기, 입력크기, learning_rate, layerType, stream)
         fc1_opt(128, 2*2*32, 0.00001, p2::layerType::perceptron, hs.model_str),
         fc2_opt(64, 128, 0.00001, p2::layerType::perceptron, hs.model_str),
         fc3_opt(32, 64, 0.00001, p2::layerType::perceptron, hs.model_str),
         fc_out_opt(8, 32, 0.00001, p2::layerType::perceptron, hs.model_str),
         
         // Conv layers: 10x10 -> feature extraction
         conv1(bs, 1, 10, 10,   8, 5, 5,  1, 1,  1, 1, &conv1_opt, d2::InitType::He, hs.model_str), // 10x10x1 -> 6x6x8
         conv2(bs, 8, 6, 6,     16, 3, 3, 1, 1,  1, 1, &conv2_opt, d2::InitType::He, hs.model_str), // 6x6x8 -> 4x4x16  
         conv3(bs, 16, 4, 4,    32, 3, 3, 1, 1,  1, 1, &conv3_opt, d2::InitType::He, hs.model_str), // 4x4x16 -> 2x2x32
         
         // FC layers: flattened features -> prediction
         fc1(bs, 2*2*32, 128, &fc1_opt, d2::InitType::He, hs.model_str),     // 128 flattened -> 128
         fc2(bs, 128, 64, &fc2_opt, d2::InitType::He, hs.model_str),         // 128 -> 64  
         fc3(bs, 64, 32, &fc3_opt, d2::InitType::He, hs.model_str),          // 64 -> 32
         fc_out(bs, 32, 8, &fc_out_opt, d2::InitType::He, hs.model_str)     // 32 -> 8 (output)
         
         {}
        std::pair<d2::d_matrix_2<double>, double> forward(d2::d_matrix_2<double> X, d2::d_matrix_2<double> target, cudaStream_t str = 0){
            // Conv layers with activation
            conv1.forward(X, str);
            auto conv1_activated = act.Active(conv1.getOutput(), p2::ActType::LReLU, str);
            
            conv2.forward(conv1_activated, str);
            auto conv2_activated = act.Active(conv2.getOutput(), p2::ActType::LReLU, str);
            
            conv3.forward(conv2_activated, str);
            auto conv3_activated = act.Active(conv3.getOutput(), p2::ActType::LReLU, str);
            
            // FC layers with activation
            fc1.feedforward(conv3_activated, str);
            auto fc1_activated = act.Active(fc1.getOutput(), p2::ActType::Tanh, str);
            
            fc2.feedforward(fc1_activated, str);
            auto fc2_activated = act.Active(fc2.getOutput(), p2::ActType::Tanh, str);
            
            fc3.feedforward(fc2_activated, str);
            auto fc3_activated = act.Active(fc3.getOutput(), p2::ActType::Tanh, str);
            
            fc_out.feedforward(fc3_activated, str);
            auto final_output = act.Active(fc_out.getOutput(), p2::ActType::Softsign, str);
            
            // Loss calculation  
            double loss_val = loss.getLoss(final_output, target, p2::LossType::CrossEntropy, str);
            
            return {final_output, loss_val};
        }
        
        void backward(d2::d_matrix_2<double> final_output, d2::d_matrix_2<double> target, cudaStream_t str = 0){
            // Get loss gradient
            auto loss_grad = loss.getGrad(final_output, target, p2::LossType::CrossEntropy, str);
            
            // Backward through FC layers
            auto fc_out_act_deriv = act.d_Active(fc_out.getOutput(), p2::ActType::Softsign, str);
            auto delta_fc_out = fc_out.backprop(loss_grad, fc_out_act_deriv, str);
            
            auto fc3_act_deriv = act.d_Active(fc3.getOutput(), p2::ActType::Tanh, str);
            auto delta_fc3 = fc3.backprop(delta_fc_out, fc3_act_deriv, str);
            
            auto fc2_act_deriv = act.d_Active(fc2.getOutput(), p2::ActType::Tanh, str);
            auto delta_fc2 = fc2.backprop(delta_fc3, fc2_act_deriv, str);
            
            auto fc1_act_deriv = act.d_Active(fc1.getOutput(), p2::ActType::Tanh, str);
            auto delta_fc1 = fc1.backprop(delta_fc2, fc1_act_deriv, str);
            
            // Backward through conv layers
            auto conv3_act_deriv = act.d_Active(conv3.getOutput(), p2::ActType::LReLU, str);
            auto delta_conv3 = conv3.backward(delta_fc1, conv3_act_deriv, str);
            
            auto conv2_act_deriv = act.d_Active(conv2.getOutput(), p2::ActType::LReLU, str);
            auto delta_conv2 = conv2.backward(delta_conv3, conv2_act_deriv, str);
            
            auto conv1_act_deriv = act.d_Active(conv1.getOutput(), p2::ActType::LReLU, str);
            conv1.backward(delta_conv2, conv1_act_deriv, str);
        }

        void train(int epochs){
            auto start = std::chrono::steady_clock::now();
            
            // GOL 데이터 로드 (배치 형태로 직접 로드)
            auto [X, Y] = GOL_2::LoadingDataBatch(hs.model_str);  // 이미 (N, features) 형태
            
            int N = X.getRow();      // 전체 데이터 개수
            int input_size = X.getCol();   // 입력 크기 (100)
            int output_size = Y.getCol();  // 출력 크기 (8)
            
            std::cout << "[데이터 로드 완료] " << N << "개 샘플, 입력크기: " << input_size << ", 출력크기: " << output_size << std::endl;
            
            int B = batch;           // 배치 크기
            int num_batches = (N + B - 1) / B;  // 총 배치 수
            
            // 배치별로 데이터 미리 분할
            std::vector<d2::d_matrix_2<double>> batch_data(num_batches), batch_labels(num_batches);
            for(int i = 0; i < num_batches; ++i){
                batch_data[i] = X.getBatch(B, i*B);
                batch_labels[i] = Y.getBatch(B, i*B);
                printProgressBar(i+1, num_batches, start, "batch loading... (batch " + std::to_string(i+1) + "/" + std::to_string(num_batches) + ")");
            }
            std::cout << std::endl;
            std::cout << "[배치 로드 완료] 총 " << N << "개 데이터, " << num_batches << "개 배치" << std::endl;
            
            // Loss 데이터 저장을 위한 파일 생성
            std::ofstream epoch_loss_file("../graph/epoch_loss.txt");
            std::ofstream batch_loss_file("../graph/batch_loss.txt");
            
            // 훈련 루프
            std::string progress_avgloss;
            for(int e = 1; e <= epochs; e++) {
                double avgloss = 0;
                
                for(int j = 0; j < num_batches; j++){
                    // 순전파
                    auto [output, loss_val] = forward(batch_data[j], batch_labels[j], hs.model_str);
                    
                    avgloss += loss_val;
                    
                    // NaN 체크
                    if(std::isnan(loss_val)){
                        std::cerr << "Loss is NaN at batch " << j+1 << ", epoch " << e << std::endl;
                        std::cerr << "Output (first 10 elements): ";
                        output.cpyToHost();
                        for(int k=0; k<std::min(10, output.size()); ++k) 
                            std::cerr << output.getHostPointer()[k] << " ";
                        std::cerr << std::endl;
                        std::cerr << "Labels (first 10 elements): ";
                        batch_labels[j].cpyToHost();
                        for(int k=0; k<std::min(10, batch_labels[j].size()); ++k) 
                            std::cerr << batch_labels[j].getHostPointer()[k] << " ";
                        std::cerr << std::endl;
                        throw std::runtime_error("Invalid error in loss calc.");
                    }
                    
                    // 역전파
                    backward(output, batch_labels[j], hs.model_str);
                    
                    // 배치별 loss 저장
                    batch_loss_file << e << " " << j+1 << " " << loss_val << std::endl;
                    
                    // 진행 상황 표시
                    std::string progress_batch = "batch" + std::to_string(j+1);
                    std::string progress_loss = "loss:" + std::to_string(loss_val);
                    printProgressBar(e, epochs, start, progress_avgloss + " | " + progress_batch + " 의 " + progress_loss);
                }
                
                avgloss = avgloss / static_cast<double>(num_batches);
                progress_avgloss = "[epoch" + std::to_string(e+1) + "/" + std::to_string(epochs) + "의 avgloss]:" + std::to_string(avgloss);
                
                // Epoch별 평균 loss 저장
                epoch_loss_file << e << " " << avgloss << std::endl;
                
            }
            
            // 파일 닫기
            epoch_loss_file.close();
            batch_loss_file.close();
            
            std::cout << std::endl;
            std::cout << "총 학습 시간: "
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - start
                         ).count() << "초" << std::endl;
        }

};


int main(){
    try {
        // CUDA 디바이스 확인
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) 
        {
            std::cerr << "[FATAL] No CUDA device: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        std::cout << "CUDA devices found: " << deviceCount << std::endl;
        
        // 설정
        constexpr int BATCH_SIZE = 50;
        constexpr int EPOCHS = 100;
        
        std::cout << "\n=== GOL CNN Solver 훈련 시작 ===" << std::endl;
        std::cout << "Batch Size: " << BATCH_SIZE << std::endl;
        std::cout << "Epochs: " << EPOCHS << std::endl;
        
        // 솔버 생성
        GOLsolver_1 solver(BATCH_SIZE);
        
        // 훈련 실행
        solver.train(EPOCHS);
        
        std::cout << "\n=== 훈련 완료! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


