#include "perceptronVer2.hpp"
int main(){
    SGDOptimizer opt(0.01);
    PerceptronLayer layer(2,2,&opt, InitType::He);
    d_matrix<double> input({1.0, 2.0});
    layer.feedforward(input);
    return 0;
}
