#include <ver2/perceptron_2.hpp>
#include <ver2/utility.hpp>
#include <ver2/GOLdatabase_2.hpp>
#include <chrono>
#include <iostream>
#include <memory>

namespace p2 = perceptron_2;

// Simple Attention mechanism for GOL pattern analysis
class GOLAttentionLayer {
private:
    int batch_size;
    int pattern_size; // 10x10 = 100
    int d_model;
    
    d2::d_matrix_2<double> Q, K, V;
    d2::d_matrix_2<double> attention_scores;
    d2::d_matrix_2<double> attention_weights;
    d2::d_matrix_2<double> attended_output;
    
    p2::PerceptronLayer* W_q;
    p2::PerceptronLayer* W_k; 
    p2::PerceptronLayer* W_v;
    p2::PerceptronLayer* W_o;
    
    p2::ActivateLayer act;

public:
    GOLAttentionLayer(int bs, int pattern_size, int d_model, 
                     p2::optimizer* opt_q, p2::optimizer* opt_k, 
                     p2::optimizer* opt_v, p2::optimizer* opt_o, 
                     cudaStream_t str = 0)
        : batch_size(bs), pattern_size(pattern_size), d_model(d_model),
          Q(bs, d_model, str), K(bs, d_model, str), V(bs, d_model, str),
          attention_scores(bs, pattern_size, str),
          attention_weights(bs, pattern_size, str),
          attended_output(bs, d_model, str) {
        
        W_q = new p2::PerceptronLayer(bs, pattern_size, d_model, opt_q, d2::InitType::Xavier, str);
        W_k = new p2::PerceptronLayer(bs, pattern_size, d_model, opt_k, d2::InitType::Xavier, str);
        W_v = new p2::PerceptronLayer(bs, pattern_size, d_model, opt_v, d2::InitType::Xavier, str);
        W_o = new p2::PerceptronLayer(bs, d_model, d_model, opt_o, d2::InitType::Xavier, str);
    }
    
    ~GOLAttentionLayer() {
        delete W_q;
        delete W_k;
        delete W_v;
        delete W_o;
    }
    
    d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& input, cudaStream_t str = 0) {
        // Generate Q, K, V from input
        W_q->feedforward(input, str);
        Q = W_q->getOutput();
        
        W_k->feedforward(input, str);
        K = W_k->getOutput();
        
        W_v->feedforward(input, str);
        V = W_v->getOutput();
        
        // Compute attention scores: Q * K^T
        attention_scores = d2::matrixMP(Q, K.transpose(str), str);
        
        // Scale by sqrt(d_model)
        double scale = 1.0 / sqrt(static_cast<double>(d_model));
        attention_scores = d2::ScalaProduct(attention_scores, scale, str);
        
        // Apply softmax to get attention weights
        attention_weights = act.Active(attention_scores, p2::ActType::Softmax, str);
        
        // Apply attention to values
        attended_output = d2::matrixMP(attention_weights, V, str);
        
        // Final output projection
        W_o->feedforward(attended_output, str);
        
        return W_o->getOutput();
    }
    
    d2::d_matrix_2<double> backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str = 0) {
        // Backward through output projection
        auto grad_attended = W_o->backprop(grad_output, d2::d_matrix_2<double>(), str);
        
        // Backward through attention mechanism (simplified)
        auto grad_V = d2::matrixMP(attention_weights.transpose(str), grad_attended, str);
        auto grad_weights = d2::matrixMP(grad_attended, V.transpose(str), str);
        
        // Backward through softmax
        auto softmax_deriv = act.d_Active(attention_scores, p2::ActType::Softmax, str);
        auto grad_scores = d2::HadamardProduct(grad_weights, softmax_deriv, str);
        
        // Backward through Q, K, V projections
        auto grad_Q = d2::matrixMP(grad_scores, K, str);
        auto grad_K = d2::matrixMP(grad_scores.transpose(str), Q, str);
        
        auto grad_input_q = W_q->backprop(grad_Q, d2::d_matrix_2<double>(), str);
        auto grad_input_k = W_k->backprop(grad_K, d2::d_matrix_2<double>(), str);
        auto grad_input_v = W_v->backprop(grad_V, d2::d_matrix_2<double>(), str);
        
        // Sum gradients
        auto grad_input = d2::matrixPlus(grad_input_q, 
                         d2::matrixPlus(grad_input_k, grad_input_v, str), str);
        
        return grad_input;
    }
    
    const d2::d_matrix_2<double>& getAttentionWeights() const {
        return attention_weights;
    }
};

// Enhanced GOL solver with Attention mechanism
class GOLsolver_Attention {
private:
    p2::handleStream hs;
    p2::ActivateLayer act;
    p2::LossLayer loss;

    // Attention layer optimizers
    std::unique_ptr<p2::Adam> attn_q_opt, attn_k_opt, attn_v_opt, attn_o_opt;
    
    // Conv layer optimizers
    std::unique_ptr<p2::Adam> conv1_opt, conv2_opt, conv3_opt;
    
    // FC layer optimizers  
    std::unique_ptr<p2::Adam> fc1_opt, fc2_opt, fc3_opt, fc_out_opt;

    // Attention layer for pattern analysis
    GOLAttentionLayer* attention_layer;
    
    // Convolutional layers for spatial feature extraction
    p2::convLayer conv1, conv2, conv3;
    
    // Fully connected layers
    p2::PerceptronLayer fc1, fc2, fc3, fc_out;

    int batch_size;
    int pattern_size = 100; // 10x10
    int attention_dim = 64;

public:
    GOLsolver_Attention(int bs, double lr = 0.0001) : batch_size(bs) {
        
        // Initialize optimizers
        attn_q_opt = std::make_unique<p2::Adam>(pattern_size, attention_dim, lr, p2::layerType::perceptron, hs.model_str);
        attn_k_opt = std::make_unique<p2::Adam>(pattern_size, attention_dim, lr, p2::layerType::perceptron, hs.model_str);
        attn_v_opt = std::make_unique<p2::Adam>(pattern_size, attention_dim, lr, p2::layerType::perceptron, hs.model_str);
        attn_o_opt = std::make_unique<p2::Adam>(attention_dim, attention_dim, lr, p2::layerType::perceptron, hs.model_str);
        
        conv1_opt = std::make_unique<p2::Adam>(8, 1*5*5, lr, p2::layerType::conv, hs.model_str);
        conv2_opt = std::make_unique<p2::Adam>(16, 8*3*3, lr, p2::layerType::conv, hs.model_str);
        conv3_opt = std::make_unique<p2::Adam>(32, 16*3*3, lr, p2::layerType::conv, hs.model_str);
        
        fc1_opt = std::make_unique<p2::Adam>(128, 2*2*32 + attention_dim, lr, p2::layerType::perceptron, hs.model_str);
        fc2_opt = std::make_unique<p2::Adam>(64, 128, lr, p2::layerType::perceptron, hs.model_str);
        fc3_opt = std::make_unique<p2::Adam>(32, 64, lr, p2::layerType::perceptron, hs.model_str);
        fc_out_opt = std::make_unique<p2::Adam>(8, 32, lr, p2::layerType::perceptron, hs.model_str);
        
        // Initialize layers
        attention_layer = new GOLAttentionLayer(bs, pattern_size, attention_dim,
                                               attn_q_opt.get(), attn_k_opt.get(), 
                                               attn_v_opt.get(), attn_o_opt.get(), hs.model_str);
        
        // Conv layers: 10x10 spatial pattern analysis
        conv1 = p2::convLayer(bs, 1, 10, 10, 8, 5, 5, 1, 1, 1, 1, conv1_opt.get(), d2::InitType::He, hs.model_str);
        conv2 = p2::convLayer(bs, 8, 6, 6, 16, 3, 3, 1, 1, 1, 1, conv2_opt.get(), d2::InitType::He, hs.model_str);
        conv3 = p2::convLayer(bs, 16, 4, 4, 32, 3, 3, 1, 1, 1, 1, conv3_opt.get(), d2::InitType::He, hs.model_str);
        
        // FC layers: combined features -> prediction
        fc1 = p2::PerceptronLayer(bs, 2*2*32 + attention_dim, 128, fc1_opt.get(), d2::InitType::He, hs.model_str);
        fc2 = p2::PerceptronLayer(bs, 128, 64, fc2_opt.get(), d2::InitType::He, hs.model_str);
        fc3 = p2::PerceptronLayer(bs, 64, 32, fc3_opt.get(), d2::InitType::He, hs.model_str);
        fc_out = p2::PerceptronLayer(bs, 32, 8, fc_out_opt.get(), d2::InitType::He, hs.model_str);
    }
    
    ~GOLsolver_Attention() {
        delete attention_layer;
    }
    
    std::pair<d2::d_matrix_2<double>, double> forward(const d2::d_matrix_2<double>& X, 
                                                     const d2::d_matrix_2<double>& target, 
                                                     cudaStream_t str = 0) {
        
        // 1. Reshape input for attention (batch_size, 100) for flattened 10x10 patterns
        d2::d_matrix_2<double> flattened_input(batch_size, pattern_size, str);
        // In practice, you'd need a proper reshape kernel here
        flattened_input = X; // Assuming X is already flattened
        
        // 2. Attention mechanism for pattern analysis
        auto attention_features = attention_layer->forward(flattened_input, str);
        
        // 3. Convolutional feature extraction (reshape back to spatial)
        // Assuming X is reshaped to (batch, 1, 10, 10) format for conv layers
        conv1.forward(X, str);
        auto conv1_out = act.Active(conv1.getOutput(), p2::ActType::LReLU, str);
        
        conv2.forward(conv1_out, str);
        auto conv2_out = act.Active(conv2.getOutput(), p2::ActType::LReLU, str);
        
        conv3.forward(conv2_out, str);
        auto conv3_out = act.Active(conv3.getOutput(), p2::ActType::LReLU, str);
        
        // 4. Combine attention features with conv features
        // Flatten conv output: 2x2x32 = 128 features
        // Concatenate with attention features: 128 + 64 = 192 total features
        auto combined_features = d2::concatenate(conv3_out, attention_features, str);
        
        // 5. Fully connected layers
        fc1.feedforward(combined_features, str);
        auto fc1_out = act.Active(fc1.getOutput(), p2::ActType::Tanh, str);
        
        fc2.feedforward(fc1_out, str);
        auto fc2_out = act.Active(fc2.getOutput(), p2::ActType::Tanh, str);
        
        fc3.feedforward(fc2_out, str);
        auto fc3_out = act.Active(fc3.getOutput(), p2::ActType::Tanh, str);
        
        fc_out.feedforward(fc3_out, str);
        auto final_output = act.Active(fc_out.getOutput(), p2::ActType::Softsign, str);
        
        // 6. Loss calculation
        double loss_val = loss.getLoss(final_output, target, p2::LossType::CrossEntropy, str);
        
        return {final_output, loss_val};
    }
    
    void backward(const d2::d_matrix_2<double>& final_output, 
                  const d2::d_matrix_2<double>& target, 
                  cudaStream_t str = 0) {
        
        // Get loss gradient
        auto loss_grad = loss.getGrad(final_output, target, p2::LossType::CrossEntropy, str);
        
        // Backward through FC layers
        auto fc_out_deriv = act.d_Active(fc_out.getOutput(), p2::ActType::Softsign, str);
        auto grad_fc_out = fc_out.backprop(loss_grad, fc_out_deriv, str);
        
        auto fc3_deriv = act.d_Active(fc3.getOutput(), p2::ActType::Tanh, str);
        auto grad_fc3 = fc3.backprop(grad_fc_out, fc3_deriv, str);
        
        auto fc2_deriv = act.d_Active(fc2.getOutput(), p2::ActType::Tanh, str);
        auto grad_fc2 = fc2.backprop(grad_fc3, fc2_deriv, str);
        
        auto fc1_deriv = act.d_Active(fc1.getOutput(), p2::ActType::Tanh, str);
        auto grad_fc1 = fc1.backprop(grad_fc2, fc1_deriv, str);
        
        // Split gradient for conv and attention paths
        // grad_fc1 shape: (batch_size, conv_features + attention_features)
        // Need to split this into conv_grad and attention_grad
        
        // Simplified: backward through both paths
        // In practice, you'd need proper gradient splitting
        
        // Backward through attention layer
        attention_layer->backward(grad_fc1, str);
        
        // Backward through conv layers
        auto conv3_deriv = act.d_Active(conv3.getOutput(), p2::ActType::LReLU, str);
        auto grad_conv3 = conv3.backward(grad_fc1, conv3_deriv, str);
        
        auto conv2_deriv = act.d_Active(conv2.getOutput(), p2::ActType::LReLU, str);
        auto grad_conv2 = conv2.backward(grad_conv3, conv2_deriv, str);
        
        auto conv1_deriv = act.d_Active(conv1.getOutput(), p2::ActType::LReLU, str);
        auto grad_conv1 = conv1.backward(grad_conv2, conv1_deriv, str);
    }
    
    void visualize_attention(const d2::d_matrix_2<double>& input, cudaStream_t str = 0) {
        // Forward pass to compute attention weights
        d2::d_matrix_2<double> flattened_input(batch_size, pattern_size, str);
        flattened_input = input;
        
        attention_layer->forward(flattened_input, str);
        auto attention_weights = attention_layer->getAttentionWeights();
        
        // Copy to host and print attention patterns
        attention_weights.cpyToHost();
        auto weights_data = attention_weights.getHostData();
        
        std::cout << "\n=== Attention Visualization ===" << std::endl;
        std::cout << "Pattern positions with highest attention:" << std::endl;
        
        for (int batch = 0; batch < std::min(batch_size, 2); ++batch) {
            std::cout << "Batch " << batch << ":" << std::endl;
            
            // Find top 5 attention positions
            std::vector<std::pair<double, int>> attention_pairs;
            for (int pos = 0; pos < pattern_size; ++pos) {
                int idx = batch * pattern_size + pos;
                attention_pairs.push_back({weights_data[idx], pos});
            }
            
            std::sort(attention_pairs.rbegin(), attention_pairs.rend());
            
            for (int i = 0; i < std::min(5, static_cast<int>(attention_pairs.size())); ++i) {
                int pos = attention_pairs[i].second;
                int row = pos / 10;
                int col = pos % 10;
                std::cout << "  Position (" << row << "," << col << "): " 
                         << std::fixed << std::setprecision(4) 
                         << attention_pairs[i].first << std::endl;
            }
        }
        std::cout << "=========================" << std::endl;
    }
};

// Training function for GOL with Attention
void train_gol_with_attention() {
    std::cout << "🚀 Game of Life Attention 모델 훈련 시작!" << std::endl;
    
    const int batch_size = 32;
    const int epochs = 50;
    const double learning_rate = 0.0001;
    
    GOLsolver_Attention model(batch_size, learning_rate);
    
    // Load GOL dataset (simplified - you'd use your actual dataset)
    std::cout << "📊 데이터셋 로딩..." << std::endl;
    
    // For demonstration, create dummy data
    d2::d_matrix_2<double> train_input(batch_size, 100);
    d2::d_matrix_2<double> train_target(batch_size, 8);
    
    train_input.randomInit(0.0, 1.0);
    train_target.fill(0.0);
    // Set first class as target for demo
    for (int i = 0; i < batch_size; ++i) {
        train_target.setHostValue(i, 0, 1.0);
    }
    
    train_input.cpyToDev();
    train_target.cpyToDev();
    
    std::cout << "🔄 훈련 시작..." << std::endl;
    
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Forward pass
        auto [output, loss] = model.forward(train_input, train_target);
        
        // Backward pass
        model.backward(output, train_target);
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (epoch % 10 == 0 || epoch == 1) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                     << " | Loss: " << std::setw(8) << std::fixed << std::setprecision(6) << loss
                     << " | Time: " << std::setw(4) << duration.count() << "ms" << std::endl;
            
            // Visualize attention every 20 epochs
            if (epoch % 20 == 0) {
                model.visualize_attention(train_input);
            }
        }
    }
    
    std::cout << "✅ 훈련 완료!" << std::endl;
    std::cout << "🔍 최종 Attention 패턴:" << std::endl;
    model.visualize_attention(train_input);
}

int main() {
    try {
        std::cout << "🎯 GOL Attention 모델 실험 시작!" << std::endl;
        
        train_gol_with_attention();
        
        std::cout << "\n🎉 실험 완료!" << std::endl;
        std::cout << "✨ Attention 메커니즘이 GOL 패턴 분석에 적용되었습니다!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 오류 발생: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
