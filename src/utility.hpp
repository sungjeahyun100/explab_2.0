#include <iostream>
#include <chrono>


void printProgressBar(int current, int total, std::chrono::steady_clock::time_point startTime, std::string processname) {
    int width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(width * progress);
    
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    int elapsedSec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% ";
    std::cout << '[' << processname << ']';
    std::cout << "(경과 시간: " << elapsedSec << " 초)\r";
    std::cout.flush();
}
