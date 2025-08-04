#pragma once

#include <ver2/d_matrix_2.hpp>
#include <ver2/perceptron_2.hpp>
#include <cudnn.h>
#include <cmath>

namespace d2 = d_matrix_ver2;
namespace p2 = perceptron_2;

namespace attention {

    // Multi-Head Self-Attention Layer
    class MultiHeadAttention {
    private:
        int num_heads;
        int d_model;    // 전체 임베딩 차원
        int d_k;        // key/query 차원 (d_model / num_heads)
        int d_v;        // value 차원 (d_model / num_heads)
        int seq_len;    // 시퀀스 길이
        int batch_size; // 배치 크기

        // Weight matrices for Q, K, V
        p2::PerceptronLayer W_q, W_k, W_v;
        p2::PerceptronLayer W_o;  // output projection

        // 임시 텐서들
        d2::d_matrix_2<double> Q, K, V;          // Query, Key, Value
        d2::d_matrix_2<double> scores;           // Attention scores
        d2::d_matrix_2<double> attention_weights; // Softmax된 attention weights
        d2::d_matrix_2<double> context;          // Context vector
        d2::d_matrix_2<double> multi_head_output; // 멀티헤드 연결 결과

        // Backward용 gradients
        d2::d_matrix_2<double> grad_Q, grad_K, grad_V;
        d2::d_matrix_2<double> grad_scores, grad_attention_weights;
        d2::d_matrix_2<double> grad_context;

        // 옵티마이저들
        p2::optimizer* opt_q;
        p2::optimizer* opt_k; 
        p2::optimizer* opt_v;
        p2::optimizer* opt_o;

        cudaStream_t stream;

        // Softmax 함수들
        __device__ double softmax_kernel_helper(double* arr, int idx, int size);
        void softmax_inplace(d2::d_matrix_2<double>& matrix, int dim);
        void softmax_backward(const d2::d_matrix_2<double>& grad_output, 
                            const d2::d_matrix_2<double>& softmax_output,
                            d2::d_matrix_2<double>& grad_input);

    public:
        MultiHeadAttention(int batch_size, int seq_len, int d_model, int num_heads,
                          p2::optimizer* opt_q, p2::optimizer* opt_k, 
                          p2::optimizer* opt_v, p2::optimizer* opt_o,
                          d2::InitType init = d2::InitType::Xavier,
                          cudaStream_t str = 0);

        ~MultiHeadAttention();

        // Forward pass
        d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& input, cudaStream_t str = 0);

        // Backward pass  
        d2::d_matrix_2<double> backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str = 0);

        // Getters
        const d2::d_matrix_2<double>& getAttentionWeights() const { return attention_weights; }
        const d2::d_matrix_2<double>& getOutput() const { return multi_head_output; }
    };

    // Scaled Dot-Product Attention kernels
    __global__ void scaled_dot_product_attention_kernel(
        const double* Q, const double* K, const double* V,
        double* scores, double* output,
        int batch_size, int num_heads, int seq_len, int d_k);

    __global__ void attention_softmax_kernel(
        double* scores, double* attention_weights,
        int batch_size, int num_heads, int seq_len);

    __global__ void attention_backward_kernel(
        const double* grad_output, const double* Q, const double* K, const double* V,
        const double* attention_weights,
        double* grad_Q, double* grad_K, double* grad_V,
        int batch_size, int num_heads, int seq_len, int d_k);

    // Position Encoding
    class PositionalEncoding {
    private:
        int max_seq_len;
        int d_model;
        d2::d_matrix_2<double> pe_matrix;

    public:
        PositionalEncoding(int max_seq_len, int d_model, cudaStream_t str = 0);
        
        d2::d_matrix_2<double> apply(const d2::d_matrix_2<double>& input, cudaStream_t str = 0);
    };

    // Layer Normalization
    class LayerNorm {
    private:
        int d_model;
        d2::d_matrix_2<double> gamma, beta;  // learnable parameters
        d2::d_matrix_2<double> mean, var;    // for backward pass
        double eps;
        
        p2::optimizer* opt_gamma;
        p2::optimizer* opt_beta;

    public:
        LayerNorm(int d_model, p2::optimizer* opt_gamma, p2::optimizer* opt_beta, 
                  double eps = 1e-6, cudaStream_t str = 0);
        
        d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& input, cudaStream_t str = 0);
        d2::d_matrix_2<double> backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str = 0);
    };

    // Feed Forward Network (FFN) for Transformer
    class FeedForwardNetwork {
    private:
        p2::PerceptronLayer fc1, fc2;
        p2::ActivateLayer act;
        int d_model, d_ff;

    public:
        FeedForwardNetwork(int batch_size, int seq_len, int d_model, int d_ff,
                          p2::optimizer* opt1, p2::optimizer* opt2,
                          p2::ActType activation = p2::ActType::ReLU,
                          d2::InitType init = d2::InitType::Xavier,
                          cudaStream_t str = 0);

        d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& input, cudaStream_t str = 0);
        d2::d_matrix_2<double> backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str = 0);
    };

    // Transformer Encoder Block
    class TransformerEncoderBlock {
    private:
        MultiHeadAttention* mha;  
        LayerNorm* norm1;
        FeedForwardNetwork* ffn;
        LayerNorm* norm2;
        
        // Residual connection용 임시 텐서들
        d2::d_matrix_2<double> residual1, residual2;

    public:
        TransformerEncoderBlock(int batch_size, int seq_len, int d_model, int num_heads, int d_ff,
                               p2::optimizer* opt_q, p2::optimizer* opt_k, p2::optimizer* opt_v, p2::optimizer* opt_o,
                               p2::optimizer* opt_ffn1, p2::optimizer* opt_ffn2,
                               p2::optimizer* opt_norm1_gamma, p2::optimizer* opt_norm1_beta,
                               p2::optimizer* opt_norm2_gamma, p2::optimizer* opt_norm2_beta,
                               p2::ActType ffn_activation = p2::ActType::ReLU,
                               d2::InitType init = d2::InitType::Xavier,
                               cudaStream_t str = 0);
        
        ~TransformerEncoderBlock();

        d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& input, cudaStream_t str = 0);
        d2::d_matrix_2<double> backward(const d2::d_matrix_2<double>& grad_output, cudaStream_t str = 0);
    };
}
