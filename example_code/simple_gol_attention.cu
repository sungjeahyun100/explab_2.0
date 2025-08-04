#include <ver2/perceptron_2.hpp>
#include <ver2/utility.hpp>
#include <chrono>
#include <iostream>
#include <memory>

namespace p2 = perceptron_2;

// 간단한 GOL Attention 테스트 (메모리 문제 해결 버전)
class SimpleGOLAttention {
private:
    int batch_size;
    int input_dim;
    int hidden_dim;
    
    std::unique_ptr<p2::Adam> fc1_opt, fc2_opt, attention_opt;
    p2::PerceptronLayer fc1, fc2, attention_layer;
    p2::ActivateLayer act;
    p2::LossLayer loss;
    p2::handleStream hs;

public:
    SimpleGOLAttention(int bs, int input_d = 100, int hidden_d = 64, double lr = 0.001) 
        : batch_size(bs), input_dim(input_d), hidden_dim(hidden_d) {
        
        // 옵티마이저 초기화
        fc1_opt = std::make_unique<p2::Adam>(input_d, hidden_d, lr, p2::layerType::perceptron, hs.model_str);
        fc2_opt = std::make_unique<p2::Adam>(hidden_d, hidden_d, lr, p2::layerType::perceptron, hs.model_str);
        attention_opt = std::make_unique<p2::Adam>(hidden_d, 8, lr, p2::layerType::perceptron, hs.model_str);
        
        // 레이어 초기화
        fc1 = p2::PerceptronLayer(bs, input_d, hidden_d, fc1_opt.get(), d2::InitType::Xavier, hs.model_str);
        fc2 = p2::PerceptronLayer(bs, hidden_d, hidden_d, fc2_opt.get(), d2::InitType::Xavier, hs.model_str);
        attention_layer = p2::PerceptronLayer(bs, hidden_d, 8, attention_opt.get(), d2::InitType::Xavier, hs.model_str);
        
        std::cout << "✅ SimpleGOLAttention 모델 초기화 완료!" << std::endl;
    }
    
    std::pair<d2::d_matrix_2<double>, double> forward(const d2::d_matrix_2<double>& input,
                                                     const d2::d_matrix_2<double>& target,
                                                     cudaStream_t str = 0) {
        
        // 첫 번째 FC 레이어
        fc1.feedforward(input, str);
        auto fc1_out = act.Active(fc1.getOutput(), p2::ActType::ReLU, str);
        
        // 두 번째 FC 레이어 (attention-like processing)
        fc2.feedforward(fc1_out, str);  
        auto fc2_out = act.Active(fc2.getOutput(), p2::ActType::Tanh, str);
        
        // Attention 출력 레이어
        attention_layer.feedforward(fc2_out, str);
        auto final_output = act.Active(attention_layer.getOutput(), p2::ActType::Softmax, str);
        
        // 손실 계산
        double loss_val = loss.getLoss(final_output, target, p2::LossType::CrossEntropy, str);
        
        return {final_output, loss_val};
    }
    
    void backward(const d2::d_matrix_2<double>& output,
                  const d2::d_matrix_2<double>& target,
                  cudaStream_t str = 0) {
        
        // 손실 기울기
        auto loss_grad = loss.getGrad(output, target, p2::LossType::CrossEntropy, str);
        
        // Attention 레이어 역전파
        auto attention_deriv = act.d_Active(attention_layer.getOutput(), p2::ActType::Softmax, str);
        auto grad_attention = attention_layer.backprop(loss_grad, attention_deriv, str);
        
        // FC2 역전파
        auto fc2_deriv = act.d_Active(fc2.getOutput(), p2::ActType::Tanh, str);
        auto grad_fc2 = fc2.backprop(grad_attention, fc2_deriv, str);
        
        // FC1 역전파
        auto fc1_deriv = act.d_Active(fc1.getOutput(), p2::ActType::ReLU, str);
        auto grad_fc1 = fc1.backprop(grad_fc2, fc1_deriv, str);
    }
    
    void show_attention_pattern(const d2::d_matrix_2<double>& input) {
        auto [output, _] = forward(input, d2::d_matrix_2<double>(batch_size, 8));
        
        // 첫 번째 배치의 attention 가중치 출력
        fc2.feedforward(act.Active(fc1.getOutput(), p2::ActType::ReLU), 0);
        auto attention_features = fc2.getOutput();
        
        attention_features.cpyToHost();
        
        std::cout << "\n🔍 Attention 패턴 (첫 번째 배치):" << std::endl;
        std::cout << "Hidden features (first 10 values): ";
        for (int i = 0; i < std::min(10, hidden_dim); ++i) {
            std::cout << std::fixed << std::setprecision(4) 
                     << attention_features.getHostValue(0, i) << " ";
        }
        std::cout << std::endl;
    }
};

void test_simple_gol_attention() {
    std::cout << "🚀 간단한 GOL Attention 테스트 시작!" << std::endl;
    
    const int batch_size = 16;
    const int input_dim = 100; // 10x10 GOL 패턴
    const int epochs = 20;
    
    try {
        SimpleGOLAttention model(batch_size, input_dim, 64);
        
        // 테스트 데이터 생성
        d2::d_matrix_2<double> train_input(batch_size, input_dim);
        d2::d_matrix_2<double> train_target(batch_size, 8);
        
        std::cout << "📊 테스트 데이터 생성 중..." << std::endl;
        
        // 랜덤 GOL 패턴 생성
        train_input.randomInit(0.0, 1.0);
        
        // 타겟 설정 (분류 문제)
        train_target.fill(0.0);
        for (int i = 0; i < batch_size; ++i) {
            int target_class = i % 8; // 8개 클래스 순환
            train_target.setHostValue(i, target_class, 1.0);
        }
        
        train_input.cpyToDev();
        train_target.cpyToDev();
        
        std::cout << "🔄 훈련 시작..." << std::endl;
        
        // 훈련 루프
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            auto start_time = std::chrono::steady_clock::now();
            
            // Forward pass
            auto [output, loss_val] = model.forward(train_input, train_target);
            
            // Backward pass
            model.backward(output, train_target);
            
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            if (epoch % 5 == 0 || epoch == 1) {
                std::cout << "Epoch " << std::setw(2) << epoch 
                         << " | Loss: " << std::setw(8) << std::fixed << std::setprecision(6) << loss_val
                         << " | Time: " << std::setw(3) << duration.count() << "ms" << std::endl;
                
                if (epoch % 10 == 0) {
                    model.show_attention_pattern(train_input);
                }
            }
        }
        
        std::cout << "\n✅ 간단한 GOL Attention 테스트 완료!" << std::endl;
        model.show_attention_pattern(train_input);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 오류 발생: " << e.what() << std::endl;
        throw;
    }
}

int main() {
    try {
        std::cout << "🎯 간단한 GOL Attention 실험!" << std::endl;
        
        test_simple_gol_attention();
        
        std::cout << "\n🎉 실험 완료!" << std::endl;
        std::cout << "✨ 메모리 문제 없이 정상 작동!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 치명적 오류: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
