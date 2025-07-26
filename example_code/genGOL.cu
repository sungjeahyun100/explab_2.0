#include<ver2/GOLdatabase_2.hpp>

int main(){
    int sample;
    std::cout << "생성할 샘플 개수를 입력하시오:";
    std::cin >> sample; 
    GOL_2::generateGameOfLifeData(sample, 0.3);
}
