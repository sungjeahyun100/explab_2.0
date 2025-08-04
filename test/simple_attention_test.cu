#include <iostream>
#include <vector>
#include <ver2/d_matrix_2.hpp>
#include <ver2/perceptron_2.hpp>

namespace d2 = d_matrix_ver2;
namespace p2 = perceptron_2;

// 간단한 Self-Attention 구현 (테스트용)
class SimpleAttention {
private:
    int d_model;
    int seq_len;
    int batch_size;
    
    d2::d_matrix_2<double> Q, K, V;
    d2::d_matrix_2<double> scores;
    d2::d_matrix_2<double> attention_weights;
    d2::d_matrix_2<double> output;

public:
    SimpleAttention(int batch_size, int seq_len, int d_model, cudaStream_t str = 0)
        : batch_size(batch_size), seq_len(seq_len), d_model(d_model),
          Q(batch_size * seq_len, d_model, str),
          K(batch_size * seq_len, d_model, str),
          V(batch_size * seq_len, d_model, str),
          scores(batch_size * seq_len, seq_len, str),
          attention_weights(batch_size * seq_len, seq_len, str),
          output(batch_size * seq_len, d_model, str) {}

    d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& input, cudaStream_t str = 0) {
        // 간단화를 위해 Q=K=V=input으로 사용 (self-attention)
        Q = input;
        K = input;
        V = input;
        
        // Q * K^T 계산 (간단한 행렬 곱셈)
        scores = d2::matrixMP(Q, K.transpose(str), str);
        
        // 스케일링 (sqrt(d_model)로 나누기)
        double scale = 1.0 / sqrt(static_cast<double>(d_model));
        scores = d2::ScalaProduct(scores, scale, str);
        
        // Softmax 적용 (여기서는 간단히 처리)
        attention_weights = d2::MatrixActivate<double, d2::sigmoid>(scores, str);
        
        // Attention weights * V
        output = d2::matrixMP(attention_weights, V, str);
        
        return output;
    }
    
    const d2::d_matrix_2<double>& getAttentionWeights() const {
        return attention_weights;
    }
};

// 간단한 Transformer 블록
class SimpleTransformerBlock {
private:
    SimpleAttention* attention;
    p2::PerceptronLayer* ffn1;
    p2::PerceptronLayer* ffn2;
    p2::ActivateLayer act;
    
    d2::d_matrix_2<double> residual;

public:
    SimpleTransformerBlock(int batch_size, int seq_len, int d_model, int d_ff,
                          p2::optimizer* opt1, p2::optimizer* opt2, cudaStream_t str = 0)
        : residual(batch_size * seq_len, d_model, str) {
        
        attention = new SimpleAttention(batch_size, seq_len, d_model, str);
        ffn1 = new p2::PerceptronLayer(batch_size * seq_len, d_model, d_ff, opt1, d2::InitType::Xavier, str);
        ffn2 = new p2::PerceptronLayer(batch_size * seq_len, d_ff, d_model, opt2, d2::InitType::Xavier, str);
    }
    
    ~SimpleTransformerBlock() {
        delete attention;
        delete ffn1;
        delete ffn2;
    }
    
    d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& input, cudaStream_t str = 0) {
        // Self-Attention with residual connection
        residual = input;
        auto attn_output = attention->forward(input, str);
        auto after_attn = d2::matrixPlus(residual, attn_output, str);
        
        // Feed Forward Network with residual connection
        residual = after_attn;
        ffn1->feedforward(after_attn, str);
        auto ffn1_output = act.Active(ffn1->getOutput(), p2::ActType::ReLU, str);
        ffn2->feedforward(ffn1_output, str);
        auto final_output = d2::matrixPlus(residual, ffn2->getOutput(), str);
        
        return final_output;
    }
    
    void backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str = 0) {
        // FFN backward
        auto ffn2_deriv = d2::d_matrix_2<double>(ffn2->getOutput().getRow(), ffn2->getOutput().getCol(), str);
        ffn2_deriv.fill(1.0); // Identity derivative for residual connection
        auto grad_ffn2 = ffn2->backprop(grad_output, ffn2_deriv, str);
        
        auto ffn1_deriv = act.d_Active(ffn1->getOutput(), p2::ActType::ReLU, str);
        auto grad_ffn1 = ffn1->backprop(grad_ffn2, ffn1_deriv, str);
        
        // 간단화를 위해 attention backward는 생략
    }
};

// 간단한 테스트용 모델
class SimpleTransformerModel {
private:
    int batch_size, seq_len, d_model, output_dim;
    SimpleTransformerBlock* transformer_block;
    p2::PerceptronLayer* output_layer;
    p2::ActivateLayer act;
    
    std::unique_ptr<p2::Adam> opt1, opt2, opt_out;

public:
    SimpleTransformerModel(int batch_size, int seq_len, int d_model, int output_dim, 
                          double lr = 0.001, cudaStream_t str = 0)
        : batch_size(batch_size), seq_len(seq_len), d_model(d_model), output_dim(output_dim) {
        
        // 옵티마이저 생성
        opt1 = std::make_unique<p2::Adam>(d_model, d_model * 4, lr, p2::layerType::perceptron, str);
        opt2 = std::make_unique<p2::Adam>(d_model * 4, d_model, lr, p2::layerType::perceptron, str);
        opt_out = std::make_unique<p2::Adam>(d_model, output_dim, lr, p2::layerType::perceptron, str);
        
        // 레이어 생성
        transformer_block = new SimpleTransformerBlock(batch_size, seq_len, d_model, d_model * 4,
                                                      opt1.get(), opt2.get(), str);
        output_layer = new p2::PerceptronLayer(batch_size * seq_len, d_model, output_dim,
                                              opt_out.get(), d2::InitType::Xavier, str);
    }
    
    ~SimpleTransformerModel() {
        delete transformer_block;
        delete output_layer;
    }
    
    d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& input, cudaStream_t str = 0) {
        auto transformer_output = transformer_block->forward(input, str);
        output_layer->feedforward(transformer_output, str);
        return act.Active(output_layer->getOutput(), p2::ActType::Softmax, str);
    }
    
    void backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str = 0) {
        auto output_deriv = act.d_Active(output_layer->getOutput(), p2::ActType::Softmax, str);
        auto grad_transformer = output_layer->backprop(grad_output, output_deriv, str);
        transformer_block->backward(grad_transformer, str);
    }
};

int main() {
    try {
        std::cout << "간단한 Attention 메커니즘 테스트 시작!" << std::endl;
        
        // 파라미터 설정
        const int batch_size = 2;
        const int seq_len = 8;
        const int d_model = 32;
        const int output_dim = 10;
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 모델 생성
        SimpleTransformerModel model(batch_size, seq_len, d_model, output_dim, 0.001, stream);
        
        // 더미 입력 데이터 생성
        d2::d_matrix_2<double> input(batch_size * seq_len, d_model, stream);
        input = d2::InitWeight<double>(input.getRow(), input.getCol(), d2::InitType::Uniform, stream);
        
        std::cout << "Forward pass 실행 중..." << std::endl;
        
        // Forward pass
        auto output = model.forward(input, stream);
        
        // 결과 확인
        output.cpyToHost();
        auto host_output = output.getHostData();
        
        std::cout << "모델 출력 형태: " << output.getRow() << " x " << output.getCol() << std::endl;
        std::cout << "출력 샘플 (첫 10개 값): ";
        for (int i = 0; i < std::min(10, static_cast<int>(host_output.size())); ++i) {
            std::cout << std::fixed << std::setprecision(4) << host_output[i] << " ";
        }
        std::cout << std::endl;
        
        // 간단한 training loop 테스트
        std::cout << "\n간단한 훈련 테스트..." << std::endl;
        
        // 더미 타겟 생성
        d2::d_matrix_2<double> target(batch_size * seq_len, output_dim, stream);
        target.fill(0.0);
        // 첫 번째 클래스를 타겟으로 설정
        std::vector<double> target_data(batch_size * seq_len * output_dim, 0.0);
        for (int i = 0; i < batch_size * seq_len; ++i) {
            target_data[i * output_dim] = 1.0; // one-hot encoding
        }
        target.setHostData(target_data);
        target.cpyToDev();
        
        for (int epoch = 1; epoch <= 5; ++epoch) {
            // Forward pass
            auto pred = model.forward(input, stream);
            
            // 간단한 MSE loss 계산
            auto loss_grad = d2::matrixPlus(pred, d2::ScalaProduct(target, -1.0, stream), stream);
            
            // Backward pass
            model.backward(loss_grad, stream);
            
            // Loss 계산 (간단한 MSE)
            pred.cpyToHost();
            target.cpyToHost();
            
            auto pred_data = pred.getHostData();
            auto target_data = target.getHostData();
            
            double loss = 0.0;
            for (size_t i = 0; i < pred_data.size(); ++i) {
                double diff = pred_data[i] - target_data[i];
                loss += diff * diff;
            }
            loss /= pred_data.size();
            
            std::cout << "Epoch " << epoch << ", Loss: " << std::fixed << std::setprecision(6) << loss << std::endl;
        }
        
        cudaStreamDestroy(stream);
        
        std::cout << "\n간단한 Attention 테스트 완료!" << std::endl;
        std::cout << "✅ Forward pass 성공" << std::endl;
        std::cout << "✅ Backward pass 성공" << std::endl;
        std::cout << "✅ 기본적인 훈련 루프 동작 확인" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 오류 발생: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
