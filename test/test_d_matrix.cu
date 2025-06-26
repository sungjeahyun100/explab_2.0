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

    d_matrix<double> B(2, 2);
    B.fill(2.0);

    d_matrix<double> N(2, 3);
    N.fill(3.0);

    // 2) 전치 연산 테스트
    auto A_t = A.transpose();
    auto N_t = N.transpose();

    // HP test
    auto A_hp = HadamardProduct<double>(A, A_t);

    //tiled test
    auto C = matrixMP<double>(A, B);
    auto D = matrixMP<double>(B, N);

    // 3) unordered_map에 삽입
    test_un.emplace(A, A_t);
    std::cout << "Inserted A -> A_t into unordered_map.\n";
    test_un.emplace(N, N_t);
    std::cout << "Inserted N -> N_t into unordered_map.\n";

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
    auto another = test_un.find(N);
    if (another != test_un.end()) {
        std::cout << "Found N in map. Value (N_t):\n";
        another->second.printMatrix();
        std::cout << "Valus (N):\n";
        another->first.printMatrix();
    } else {
        std::cout << "N not found in map.";
    }

    std::cout << "A*B 행렬곱 결과:" << std::endl;
    C.printMatrix();
    std::cout << "B*N 행렬곱 결과:" << std::endl;
    D.printMatrix();
    std::cout << "A HP A_t 결과:" << std::endl;
    A_hp.printMatrix();

    // 5) vector에 push_back
    test_v.push_back(A);
    test_v.push_back(A_t);
    test_v.push_back(B);
    test_v.push_back(C);
    test_v.push_back(N);
    std::cout << "Vector size after push: " << test_v.size() << "\n";

    // 6) vector 순회 및 출력
    for (size_t i = 0; i < test_v.size(); ++i) {
        std::cout << "Matrix #" << i << ":";
        test_v[i].printMatrix();
    }

    return 0;
}
