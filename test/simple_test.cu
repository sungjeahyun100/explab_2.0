#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include "../src/ver2/d_matrix_2.hpp"

using namespace d_matrix_ver2;

int main() {
    try {
        std::cout << "d_matrix_2 function separation test start!" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        const int rows = 2, cols = 2;
        
        // Create float matrices
        d_matrix_2<float> a(rows, cols), b(rows, cols);
        a.fill(2.0f);
        b.fill(3.0f);
        a.cpyToDev();
        b.cpyToDev();
        
        std::cout << "Matrix A (2x2, value=2.0) and B (2x2, value=3.0) created" << std::endl;
        
        // HadamardProduct test
        std::cout << "Testing HadamardProduct..." << std::endl;
        d_matrix_2<float> result = HadamardProduct(a, b);
        result.cpyToHost();
        float val = result.getHostValue(0, 0);
        std::cout << "HadamardProduct result: " << val << " (expected: 6.0)" << std::endl;
        assert(std::abs(val - 6.0f) < 1e-5);
        
        // ScalaProduct test
        std::cout << "Testing ScalaProduct..." << std::endl;
        d_matrix_2<float> scaled = ScalaProduct(a, 5.0f);
        scaled.cpyToHost();
        val = scaled.getHostValue(0, 0);
        std::cout << "ScalaProduct result: " << val << " (expected: 10.0)" << std::endl;
        assert(std::abs(val - 10.0f) < 1e-5);
        
        // matrixPlus test
        std::cout << "Testing matrixPlus..." << std::endl;
        d_matrix_2<float> sum = matrixPlus(a, b);
        sum.cpyToHost();
        val = sum.getHostValue(0, 0);
        std::cout << "matrixPlus result: " << val << " (expected: 5.0)" << std::endl;
        assert(std::abs(val - 5.0f) < 1e-5);
        
        // matrixMP test
        std::cout << "Testing matrixMP..." << std::endl;
        d_matrix_2<float> mat1(2, 3), mat2(3, 2);
        mat1.fill(1.0f);
        mat2.fill(2.0f);
        mat1.cpyToDev();
        mat2.cpyToDev();
        
        d_matrix_2<float> mult_result = matrixMP(mat1, mat2);
        mult_result.cpyToHost();
        val = mult_result.getHostValue(0, 0);
        std::cout << "matrixMP result: " << val << " (expected: 6.0)" << std::endl;
        assert(std::abs(val - 6.0f) < 1e-5);
        
        std::cout << "\nAll basic tests passed!" << std::endl;
        std::cout << "d_matrix_2.hpp functions successfully separated to .cu file!" << std::endl;
        std::cout << "Tested functions:" << std::endl;
        std::cout << "   • HadamardProduct - element-wise multiplication OK" << std::endl;
        std::cout << "   • ScalaProduct - scalar multiplication OK" << std::endl;
        std::cout << "   • matrixPlus - matrix addition OK" << std::endl;
        std::cout << "   • matrixMP - matrix multiplication OK" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
