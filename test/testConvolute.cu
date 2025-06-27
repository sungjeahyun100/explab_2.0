#include<perceptron.hpp>

int main(){
    d_matrix<double> C(15, 15);
    C.fill(1.0l);
    C.cpyToDev();

    d_matrix<double> C_k(4, 4);
    C_k.fill(2.0l);
    C_k.cpyToDev();

    auto R1 = convolute(C, C_k, 1);
    auto R2 = convolute(C, C_k, 2);
    std::cout << "C행렬\n";
    C.printMatrix();
    std::cout << "C_k행렬\n";
    C_k.printMatrix();

    std::cout << "결과1:\n";
    R1.printMatrix();
    std::cout << "결과2:\n";
    R2.printMatrix();

    return 0;
}