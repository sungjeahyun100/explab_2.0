#include <iostream>
#include <vector>
#include <memory>
#include <ver2/attention.hpp>
#include <ver2/perceptron_2.hpp>
#include <ver2/d_matrix_2.hpp>

namespace d2 = d_matrix_ver2;
namespace p2 = perceptron_2;

class TransformerModel {
private:
    int batch_size;
    int seq_len;
    int d_model;
    int num_heads;
    int num_layers;
    int d_ff;
    
    // 위치 인코딩
    attention::PositionalEncoding* pos_encoding;
    
    // Transformer 인코더 블록들
    std::vector<attention::TransformerEncoderBlock*> encoder_blocks;
    
    // 최종 출력을 위한 fully connected layer
    p2::PerceptronLayer* output_layer;
    p2::ActivateLayer act;
    
    // 각 블록을 위한 옵티마이저들
    std::vector<std::unique_ptr<p2::optimizer>> optimizers;
    
    cudaStream_t stream;

public:
    TransformerModel(int batch_size, int seq_len, int d_model, int num_heads, 
                    int num_layers, int d_ff, int output_dim, double lr = 0.001, 
                    cudaStream_t str = 0) 
        : batch_size(batch_size), seq_len(seq_len), d_model(d_model), 
          num_heads(num_heads), num_layers(num_layers), d_ff(d_ff), stream(str) {
        
        // 위치 인코딩 초기화
        pos_encoding = new attention::PositionalEncoding(seq_len, d_model, str);
        
        // 각 레이어를 위한 옵티마이저들 생성
        int total_optimizers = num_layers * 10 + 1; // 각 블록당 10개 + 출력 레이어 1개
        optimizers.reserve(total_optimizers);
        
        for (int i = 0; i < total_optimizers; ++i) {
            optimizers.emplace_back(std::make_unique<p2::Adam>(d_model, d_model, lr, p2::layerType::perceptron, str));
        }
        
        // Transformer 인코더 블록들 생성
        encoder_blocks.reserve(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            int base_idx = i * 8;
            encoder_blocks.push_back(new attention::TransformerEncoderBlock(
                batch_size, seq_len, d_model, num_heads, d_ff,
                optimizers[base_idx].get(),     // Q optimizer
                optimizers[base_idx + 1].get(), // K optimizer  
                optimizers[base_idx + 2].get(), // V optimizer
                optimizers[base_idx + 3].get(), // O optimizer
                optimizers[base_idx + 4].get(), // FFN1 optimizer
                optimizers[base_idx + 5].get(), // FFN2 optimizer
                optimizers[base_idx + 6].get(), // Norm1 gamma optimizer
                optimizers[base_idx + 7].get(), // Norm1 beta optimizer
                optimizers[base_idx + 8].get(), // opt_norm2_gamma
                optimizers[base_idx + 9].get(), // opt_norm2_beta
                p2::ActType::ReLU,
                d2::InitType::Xavier,
                str
            ));
        }
        
        // 출력 레이어 생성
        output_layer = new p2::PerceptronLayer(batch_size * seq_len, d_model, output_dim, optimizers.back().get(), d2::InitType::Xavier, str);
    }
    
    ~TransformerModel() {
        delete pos_encoding;
        for (auto* block : encoder_blocks) {
            delete block;
        }
        delete output_layer;
    }
    
    // Forward pass
    d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& input, cudaStream_t str = 0) {
        // 1. 위치 인코딩 적용
        auto embedded = pos_encoding->apply(input, str);
        
        // 2. Transformer 인코더 블록들을 통과
        auto current_output = embedded;
        for (auto* block : encoder_blocks) {
            current_output = block->forward(current_output, str);
        }
        
        // 3. 최종 출력 레이어
        output_layer->feedforward(current_output, str);
        auto final_output = act.Active(output_layer->getOutput(), p2::ActType::Softmax, str);
        
        cudaStreamSynchronize(str);
        return final_output;
    }
    
    // Backward pass  
    void backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str = 0) {
        // 1. 출력 레이어 역전파
        auto output_deriv = act.d_Active(output_layer->getOutput(), p2::ActType::Softmax, str);
        auto grad_current = output_layer->backprop(grad_output, output_deriv, str);
        
        // 2. Transformer 인코더 블록들 역전파 (역순)
        for (int i = encoder_blocks.size() - 1; i >= 0; --i) {
            grad_current = encoder_blocks[i]->backward(grad_current, str);
        }
        
        cudaStreamSynchronize(str);
    }
    
    // 체스 AI를 위한 특화된 forward (보드 상태 -> 다음 수)
    d2::d_matrix_2<double> predict_chess_move(const d2::d_matrix_2<double>& board_state, cudaStream_t str = 0) {
        return forward(board_state, str);
    }
    
    // Game of Life 예측을 위한 특화된 forward  
    d2::d_matrix_2<double> predict_gol_evolution(const d2::d_matrix_2<double>& initial_state, cudaStream_t str = 0) {
        return forward(initial_state, str);
    }
};

// 체스 AI 전용 Transformer 클래스
class ChessTransformer : public TransformerModel {
private:
    static constexpr int CHESS_BOARD_SIZE = 8;
    static constexpr int CHESS_PIECES = 12; // 6 pieces * 2 colors
    static constexpr int CHESS_INPUT_DIM = CHESS_BOARD_SIZE * CHESS_BOARD_SIZE * CHESS_PIECES;
    static constexpr int CHESS_OUTPUT_DIM = 64 * 64; // from_square * to_square

public:
    ChessTransformer(int batch_size, double lr = 0.001, cudaStream_t str = 0)
        : TransformerModel(batch_size, CHESS_BOARD_SIZE * CHESS_BOARD_SIZE, 
                          512, 8, 6, 2048, CHESS_OUTPUT_DIM, lr, str) {}
    
    // 체스 보드 상태를 입력으로 받아 최적의 수를 예측
    std::pair<int, int> predict_best_move(const d2::d_matrix_2<double>& board_state, cudaStream_t str = 0) {
        auto move_probabilities = predict_chess_move(board_state, str);
        
        // 확률이 가장 높은 수를 찾기 (GPU에서 수행해야 함)
        move_probabilities.cpyToHost();
        auto host_data = move_probabilities.getHostData();
        
        int best_move_idx = 0;
        double best_prob = host_data[0];
        for (int i = 1; i < CHESS_OUTPUT_DIM; ++i) {
            if (host_data[i] > best_prob) {
                best_prob = host_data[i];
                best_move_idx = i;
            }
        }
        
        int from_square = best_move_idx / 64;
        int to_square = best_move_idx % 64;
        
        return {from_square, to_square};
    }
};

// Game of Life 예측 전용 Transformer 클래스
class GOLTransformer : public TransformerModel {
private:
    static constexpr int GOL_BOARD_SIZE = 10; // 10x10 GOL 보드
    static constexpr int GOL_INPUT_DIM = GOL_BOARD_SIZE * GOL_BOARD_SIZE;
    static constexpr int GOL_OUTPUT_DIM = 8; // 살아있는 셀 개수 예측 (0-7 범위를 8개 클래스로)

public:
    GOLTransformer(int batch_size, double lr = 0.001, cudaStream_t str = 0)
        : TransformerModel(batch_size, GOL_INPUT_DIM, 256, 4, 4, 1024, GOL_OUTPUT_DIM, lr, str) {}
    
    // GOL 초기 상태로부터 최종 살아있는 셀 개수 예측
    int predict_final_cell_count(const d2::d_matrix_2<double>& initial_state, cudaStream_t str = 0) {
        auto predictions = predict_gol_evolution(initial_state, str);
        
        // 가장 확률이 높은 클래스 찾기
        predictions.cpyToHost();
        auto host_data = predictions.getHostData();
        
        int predicted_class = 0;
        double max_prob = host_data[0];
        for (int i = 1; i < GOL_OUTPUT_DIM; ++i) {
            if (host_data[i] > max_prob) {
                max_prob = host_data[i];
                predicted_class = i;
            }
        }
        
        return predicted_class;
    }
};

// 사용 예제 함수들
void test_chess_transformer() {
    std::cout << "체스 Transformer 테스트 시작..." << std::endl;
    
    const int batch_size = 1;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    ChessTransformer chess_ai(batch_size, 0.001, stream);
    
    // 더미 체스 보드 상태 생성 (실제로는 FEN notation이나 다른 인코딩 사용)
    d2::d_matrix_2<double> board_state(batch_size, 64, stream);
    board_state.fill(0.0);
    
    // 최적의 수 예측
    auto [from_sq, to_sq] = chess_ai.predict_best_move(board_state, stream);
    
    std::cout << "예측된 최적의 수: " << from_sq << " -> " << to_sq << std::endl;
    
    cudaStreamDestroy(stream);
}

void test_gol_transformer() {
    std::cout << "Game of Life Transformer 테스트 시작..." << std::endl;
    
    const int batch_size = 1;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    GOLTransformer gol_ai(batch_size, 0.001, stream);
    
    // 더미 GOL 초기 상태 생성
    d2::d_matrix_2<double> initial_state(batch_size, 100, stream);
    initial_state.fill(0.5); // 랜덤한 초기 상태
    
    // 최종 셀 개수 예측
    int predicted_count = gol_ai.predict_final_cell_count(initial_state, stream);
    
    std::cout << "예측된 최종 살아있는 셀 개수 클래스: " << predicted_count << std::endl;
    
    cudaStreamDestroy(stream);
}

int main() {
    try {
        std::cout << "Attention 메커니즘 테스트 시작!" << std::endl;
        
        // 기본 Transformer 테스트
        const int batch_size = 2;
        const int seq_len = 10;
        const int d_model = 64;
        const int num_heads = 4;
        const int output_dim = 10;
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        TransformerModel model(batch_size, seq_len, d_model, num_heads, 2, 256, output_dim, 0.001, stream);
        
        // 더미 입력 데이터
        d2::d_matrix_2<double> input(batch_size * seq_len, d_model, stream);
        input.fill(0.1);
        
        // Forward pass
        auto output = model.forward(input, stream);
        
        // 출력 확인
        output.cpyToHost();
        auto host_output = output.getHostData();
        
        std::cout << "모델 출력 (첫 5개 값): ";
        for (int i = 0; i < std::min(5, static_cast<int>(host_output.size())); ++i) {
            std::cout << host_output[i] << " ";
        }
        std::cout << std::endl;
        
        // 특화된 모델들 테스트
        test_chess_transformer();
        test_gol_transformer();
        
        cudaStreamDestroy(stream);
        
        std::cout << "모든 테스트 완료!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
