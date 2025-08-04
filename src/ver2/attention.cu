#include "attention.hpp"
#include <cmath>
#include <algorithm>

namespace attention {

    // CUDA kernels for attention computation
    __global__ void scaled_dot_product_attention_kernel(
        const double* Q, const double* K, const double* V,
        double* scores, double* output,
        int batch_size, int num_heads, int seq_len, int d_k) {
        
        int batch_idx = blockIdx.x;
        int head_idx = blockIdx.y;
        int seq_i = threadIdx.x;
        int seq_j = threadIdx.y;
        
        if (batch_idx >= batch_size || head_idx >= num_heads || 
            seq_i >= seq_len || seq_j >= seq_len) return;
        
        // Calculate Q * K^T
        double score = 0.0;
        for (int k = 0; k < d_k; ++k) {
            int q_idx = batch_idx * num_heads * seq_len * d_k + 
                       head_idx * seq_len * d_k + seq_i * d_k + k;
            int k_idx = batch_idx * num_heads * seq_len * d_k + 
                       head_idx * seq_len * d_k + seq_j * d_k + k;
            score += Q[q_idx] * K[k_idx];
        }
        
        // Scale by sqrt(d_k)
        score /= sqrt(static_cast<double>(d_k));
        
        int score_idx = batch_idx * num_heads * seq_len * seq_len + 
                       head_idx * seq_len * seq_len + seq_i * seq_len + seq_j;
        scores[score_idx] = score;
    }

    __global__ void attention_softmax_kernel(
        double* scores, double* attention_weights,
        int batch_size, int num_heads, int seq_len) {
        
        int batch_idx = blockIdx.x;
        int head_idx = blockIdx.y;
        int seq_i = threadIdx.x;
        
        if (batch_idx >= batch_size || head_idx >= num_heads || seq_i >= seq_len) return;
        
        // Find max for numerical stability
        double max_val = -INFINITY;
        for (int j = 0; j < seq_len; ++j) {
            int idx = batch_idx * num_heads * seq_len * seq_len + 
                     head_idx * seq_len * seq_len + seq_i * seq_len + j;
            max_val = fmax(max_val, scores[idx]);
        }
        
        // Compute softmax
        double sum = 0.0;
        for (int j = 0; j < seq_len; ++j) {
            int idx = batch_idx * num_heads * seq_len * seq_len + 
                     head_idx * seq_len * seq_len + seq_i * seq_len + j;
            double exp_val = exp(scores[idx] - max_val);
            attention_weights[idx] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int j = 0; j < seq_len; ++j) {
            int idx = batch_idx * num_heads * seq_len * seq_len + 
                     head_idx * seq_len * seq_len + seq_i * seq_len + j;
            attention_weights[idx] /= sum;
        }
    }

    __global__ void attention_output_kernel(
        const double* attention_weights, const double* V, double* output,
        int batch_size, int num_heads, int seq_len, int d_v) {
        
        int batch_idx = blockIdx.x;
        int head_idx = blockIdx.y;
        int seq_i = threadIdx.x;
        int dim_j = threadIdx.y;
        
        if (batch_idx >= batch_size || head_idx >= num_heads || 
            seq_i >= seq_len || dim_j >= d_v) return;
        
        double sum = 0.0;
        for (int j = 0; j < seq_len; ++j) {
            int weight_idx = batch_idx * num_heads * seq_len * seq_len + 
                           head_idx * seq_len * seq_len + seq_i * seq_len + j;
            int v_idx = batch_idx * num_heads * seq_len * d_v + 
                       head_idx * seq_len * d_v + j * d_v + dim_j;
            sum += attention_weights[weight_idx] * V[v_idx];
        }
        
        int out_idx = batch_idx * num_heads * seq_len * d_v + 
                     head_idx * seq_len * d_v + seq_i * d_v + dim_j;
        output[out_idx] = sum;
    }

    // MultiHeadAttention implementation
    MultiHeadAttention::MultiHeadAttention(int batch_size, int seq_len, int d_model, int num_heads,
                                         p2::optimizer* opt_q, p2::optimizer* opt_k, 
                                         p2::optimizer* opt_v, p2::optimizer* opt_o,
                                         d2::InitType init, cudaStream_t str)
        : batch_size(batch_size), seq_len(seq_len), d_model(d_model), num_heads(num_heads),
          d_k(d_model / num_heads), d_v(d_model / num_heads), stream(str),
          opt_q(opt_q), opt_k(opt_k), opt_v(opt_v), opt_o(opt_o),
          W_q(batch_size * seq_len, d_model, d_model, opt_q, init, str),
          W_k(batch_size * seq_len, d_model, d_model, opt_k, init, str),
          W_v(batch_size * seq_len, d_model, d_model, opt_v, init, str),
          W_o(batch_size * seq_len, d_model, d_model, opt_o, init, str),
          Q(batch_size * seq_len, d_model, str),
          K(batch_size * seq_len, d_model, str),
          V(batch_size * seq_len, d_model, str),
          scores(batch_size * num_heads * seq_len, seq_len, str),
          attention_weights(batch_size * num_heads * seq_len, seq_len, str),
          context(batch_size * num_heads * seq_len, d_v, str),
          multi_head_output(batch_size * seq_len, d_model, str),
          grad_Q(batch_size * seq_len, d_model, str),
          grad_K(batch_size * seq_len, d_model, str),
          grad_V(batch_size * seq_len, d_model, str),
          grad_scores(batch_size * num_heads * seq_len, seq_len, str),
          grad_attention_weights(batch_size * num_heads * seq_len, seq_len, str),
          grad_context(batch_size * num_heads * seq_len, d_v, str) {
        
        if (d_model % num_heads != 0) {
            throw std::runtime_error("d_model must be divisible by num_heads");
        }
    }

    MultiHeadAttention::~MultiHeadAttention() {}

    d2::d_matrix_2<double> MultiHeadAttention::forward(const d2::d_matrix_2<double>& input, cudaStream_t str) {
        // 1. Linear transformations to get Q, K, V
        W_q.feedforward(input, str);
        Q = W_q.getOutput();
        
        W_k.feedforward(input, str);
        K = W_k.getOutput();
        
        W_v.feedforward(input, str);
        V = W_v.getOutput();
        
        // 2. Reshape for multi-head attention
        // Q, K, V: [batch_size * seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k/d_v]
        
        // 3. Compute scaled dot-product attention
        dim3 grid(batch_size, num_heads);
        dim3 block(seq_len, seq_len);
        
        scaled_dot_product_attention_kernel<<<grid, block, 0, str>>>(
            Q.getDevPointer(), K.getDevPointer(), V.getDevPointer(),
            scores.getDevPointer(), nullptr,
            batch_size, num_heads, seq_len, d_k);
        
        // 4. Apply softmax to attention scores
        dim3 softmax_grid(batch_size, num_heads);
        dim3 softmax_block(seq_len);
        
        attention_softmax_kernel<<<softmax_grid, softmax_block, 0, str>>>(
            scores.getDevPointer(), attention_weights.getDevPointer(),
            batch_size, num_heads, seq_len);
        
        // 5. Apply attention weights to values
        dim3 output_grid(batch_size, num_heads);
        dim3 output_block(seq_len, d_v);
        
        attention_output_kernel<<<output_grid, output_block, 0, str>>>(
            attention_weights.getDevPointer(), V.getDevPointer(), context.getDevPointer(),
            batch_size, num_heads, seq_len, d_v);
        
        // 6. Concatenate heads and apply output projection
        // Reshape context from [batch_size, num_heads, seq_len, d_v] to [batch_size * seq_len, d_model]
        // This would need a proper reshape kernel in practice
        
        W_o.feedforward(context, str); // Simplified - needs proper reshaping
        multi_head_output = W_o.getOutput();
        
        cudaStreamSynchronize(str);
        return multi_head_output;
    }

    d2::d_matrix_2<double> MultiHeadAttention::backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str) {
        // Backward pass through output projection
        auto grad_context_reshaped = W_o.backprop(grad_output, d2::d_matrix_2<double>(), str);
        
        // Backward through attention mechanism (simplified)
        // In practice, this would involve complex gradient calculations through attention weights
        
        // Backward through Q, K, V projections
        auto grad_input_q = W_q.backprop(grad_Q, d2::d_matrix_2<double>(), str);
        auto grad_input_k = W_k.backprop(grad_K, d2::d_matrix_2<double>(), str);
        auto grad_input_v = W_v.backprop(grad_V, d2::d_matrix_2<double>(), str);
        
        // Sum gradients from Q, K, V paths
        auto grad_input = d2::matrixPlus(grad_input_q, 
                         d2::matrixPlus(grad_input_k, grad_input_v, str), str);
        
        cudaStreamSynchronize(str);
        return grad_input;
    }

    // PositionalEncoding implementation
    PositionalEncoding::PositionalEncoding(int max_seq_len, int d_model, cudaStream_t str)
        : max_seq_len(max_seq_len), d_model(d_model), pe_matrix(max_seq_len, d_model, str) {
        
        // Initialize positional encoding matrix on host then copy to device
        std::vector<double> pe_host(max_seq_len * d_model);
        
        for (int pos = 0; pos < max_seq_len; ++pos) {
            for (int i = 0; i < d_model; ++i) {
                if (i % 2 == 0) {
                    pe_host[pos * d_model + i] = sin(pos / pow(10000.0, 2.0 * i / d_model));
                } else {
                    pe_host[pos * d_model + i] = cos(pos / pow(10000.0, 2.0 * (i-1) / d_model));
                }
            }
        }
        
        // Copy to device
        pe_matrix.setHostData(pe_host);
        pe_matrix.cpyToDev();
    }

    d2::d_matrix_2<double> PositionalEncoding::apply(const d2::d_matrix_2<double>& input, cudaStream_t str) {
        // Add positional encoding to input
        // This is a simplified version - would need proper broadcasting in practice
        return d2::matrixPlus(input, pe_matrix, str);
    }

    // LayerNorm implementation
    LayerNorm::LayerNorm(int d_model, p2::optimizer* opt_gamma, p2::optimizer* opt_beta, 
                        double eps, cudaStream_t str)
        : d_model(d_model), eps(eps), opt_gamma(opt_gamma), opt_beta(opt_beta),
          gamma(1, d_model, str), beta(1, d_model, str),
          mean(1, d_model, str), var(1, d_model, str) {
        
        gamma.fill(1.0);  // Initialize gamma to 1
        beta.fill(0.0);   // Initialize beta to 0
    }

    d2::d_matrix_2<double> LayerNorm::forward(const d2::d_matrix_2<double>& input, cudaStream_t str) {
        // Compute mean and variance along the last dimension
        // This is simplified - would need proper reduction kernels
        
        // Normalize: (x - mean) / sqrt(var + eps)
        // Scale and shift: gamma * normalized + beta
        
        // For now, return input (placeholder)
        return input;
    }

    d2::d_matrix_2<double> LayerNorm::backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str) {
        // Backward pass for layer normalization
        // This involves gradients w.r.t. input, gamma, and beta
        
        // For now, return grad_output (placeholder)
        return grad_output;
    }

    // FeedForwardNetwork implementation
    FeedForwardNetwork::FeedForwardNetwork(int batch_size, int seq_len, int d_model, int d_ff,
                                         p2::optimizer* opt1, p2::optimizer* opt2,
                                         p2::ActType activation, d2::InitType init, cudaStream_t str)
        : d_model(d_model), d_ff(d_ff),
          fc1(batch_size * seq_len, d_model, d_ff, opt1, init, str),
          fc2(batch_size * seq_len, d_ff, d_model, opt2, init, str) {}

    d2::d_matrix_2<double> FeedForwardNetwork::forward(const d2::d_matrix_2<double>& input, cudaStream_t str) {
        fc1.feedforward(input, str);
        auto activated = act.Active(fc1.getOutput(), p2::ActType::ReLU, str);
        fc2.feedforward(activated, str);
        return fc2.getOutput();
    }

    d2::d_matrix_2<double> FeedForwardNetwork::backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str) {
        auto grad_fc2 = fc2.backprop(grad_output, d2::d_matrix_2<double>(), str);
        auto fc1_output_deriv = act.d_Active(fc1.getOutput(), p2::ActType::ReLU, str);
        auto grad_fc1 = fc1.backprop(grad_fc2, fc1_output_deriv, str);
        return grad_fc1;
    }

    // TransformerEncoderBlock implementation
    TransformerEncoderBlock::TransformerEncoderBlock(
        int batch_size, int seq_len, int d_model, int num_heads, int d_ff,
        p2::optimizer* opt_q, p2::optimizer* opt_k, p2::optimizer* opt_v, p2::optimizer* opt_o,
        p2::optimizer* opt_ffn1, p2::optimizer* opt_ffn2,
        p2::optimizer* opt_norm1_gamma, p2::optimizer* opt_norm1_beta,
        p2::optimizer* opt_norm2_gamma, p2::optimizer* opt_norm2_beta,
        p2::ActType ffn_activation, d2::InitType init, cudaStream_t str)
        : residual1(batch_size * seq_len, d_model, str),
          residual2(batch_size * seq_len, d_model, str) {
        
        mha = new MultiHeadAttention(batch_size, seq_len, d_model, num_heads,
                                   opt_q, opt_k, opt_v, opt_o, init, str);
        norm1 = new LayerNorm(d_model, opt_norm1_gamma, opt_norm1_beta, 1e-6, str);
        ffn = new FeedForwardNetwork(batch_size, seq_len, d_model, d_ff,
                                   opt_ffn1, opt_ffn2, ffn_activation, init, str);
        norm2 = new LayerNorm(d_model, opt_norm2_gamma, opt_norm2_beta, 1e-6, str);
    }

    TransformerEncoderBlock::~TransformerEncoderBlock() {
        delete mha;
        delete norm1;
        delete ffn;
        delete norm2;
    }

    d2::d_matrix_2<double> TransformerEncoderBlock::forward(const d2::d_matrix_2<double>& input, cudaStream_t str) {
        // Multi-Head Attention with residual connection
        residual1 = input;
        auto mha_output = mha->forward(input, str);
        auto add_norm1 = norm1->forward(d2::matrixPlus(residual1, mha_output, str), str);
        
        // Feed Forward Network with residual connection
        residual2 = add_norm1;
        auto ffn_output = ffn->forward(add_norm1, str);
        auto add_norm2 = norm2->forward(d2::matrixPlus(residual2, ffn_output, str), str);
        
        return add_norm2;
    }

    d2::d_matrix_2<double> TransformerEncoderBlock::backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str) {
        // Backward through second residual connection and layer norm
        auto grad_norm2 = norm2->backward(grad_output, str);
        auto grad_ffn = ffn->backward(grad_norm2, str);
        auto grad_residual2 = grad_norm2; // gradient through residual connection
        auto grad_add_norm1 = d2::matrixPlus(grad_residual2, grad_ffn, str);
        
        // Backward through first residual connection and layer norm
        auto grad_norm1 = norm1->backward(grad_add_norm1, str);
        auto grad_mha = mha->backward(grad_norm1, str);
        auto grad_input = d2::matrixPlus(grad_norm1, grad_mha, str); // gradient through residual connection
        
        return grad_input;
    }

} // namespace attention
