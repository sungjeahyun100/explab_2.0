#include <d_matrix.hpp>
#include <vector>
#include <unordered_map>

int main() {
    // STL 컨테이너 호환성 테스트
    std::unordered_map<d_matrix<double>, d_matrix<double>> test_un;
    std::vector<d_matrix<double>> test_v;

    // 1) 행렬 생성 및 초기화
    d_matrix<double> A(2, 2);
    A(0, 0) = 1.0; A(0, 1) = 2.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0;
    A.cpyToDev();

    // 2) 전치 연산 테스트
    auto A_t = A.transpose();

    // 3) unordered_map에 삽입
    test_un.emplace(A, A_t);
    std::cout << "Inserted A -> A_t into unordered_map.";

    // 4) 조회 테스트
    auto it = test_un.find(A);
    if (it != test_un.end()) {
        std::cout << "Found A in map. Value (A_t):\n";
        it->second.printMatrix();
        std::cout << "Valus (A):\n";
        it->first.printMatrix();
    } else {
        std::cout << "A not found in map.";
    }

    // 5) vector에 push_back
    test_v.push_back(A);
    test_v.push_back(A_t);
    std::cout << "Vector size after push: " << test_v.size() << "";

    // 6) vector 순회 및 출력
    for (size_t i = 0; i < test_v.size(); ++i) {
        std::cout << "Matrix #" << i << ":";
        test_v[i].printMatrix();
    }

    return 0;
}
