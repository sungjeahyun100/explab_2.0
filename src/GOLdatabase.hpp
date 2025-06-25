
/**TODO:
 * make generateGameOfLifeData func
 * make LoadingData func
 * 일단 이렇게 두개. 간단해보이나 둘다 구현해야할 세부사항이 말도 안돼게 많다.
 * 일단 미리 정해놔야 할 것들을 정리해보자.
 * 우선 파일 저장소는 내가 있는 파일인 ~/explab경로에 추가적으로 폴더를 만들어놨으니 거기에 파일을 생성하고 저장한다.(경로:home/sjh100/explab/dataset)
 * 그리고 파일 확장자는 전부 .txt로 통일함.
 * 메인에선 이렇게 쓸거다. 
 * generateGameOfLifeData();
 * d_matrix<double> inputData = LoadingData();
 * 나중에 인터페이스 구현은 향후에 html을 배워서 구현할 예정.
 * 일단 가중치를 학습시키고 그걸 저장해서 따로 불러와서 쓸거다.
 * 
 * 1번째 함수)우선 저 함수는 두 부분으로 나뉜다. 패턴 구현부와 라벨링부.//이거 파일 나눌지 하나로 합칠지 고민중
 * 패턴 구현부는 말 안해도 알겠지만 10*10개의 칸의 패턴 전부를 생성해내는 구현부다. 대략 2^100 정도의 아주 큰 용량이 예상되어서 처음엔 칸수를 좀 줄일까도 생각했으나, 이 컴퓨터 스펙이면 해볼 만 하다 판단 후 강행중. 
 * 그 다음 라벨링부는 좀 네이밍이 애매해서 잘 안 와닿을 수도 있겠으나 인공지능 지도학습에서 답지 부분을 생성해내는 곳이라고 생각하면 된다.
 * 그러니까 저 생성된 패턴들을 규칙에 따라 실험해보고, 발산 및 안정화나 소멸을 판별해내는 부분이다. 이 부분 때문에 2^100의 경우의 수가 과하다고 생각했으나, 솔직히 rtx 40 시리즈 정도면 해볼만 하지 않겠는가
 * 문제는 패턴의 발산 여부의 판별인데, 만약 패턴이 발산했을 경우 이를 데이터 상에서의 판별이 쉽지 않다. 지금은 생존칸의 면적값이 선형적으로 증가하면(여기서의 "선형적으로 증가"의 의미는 정말 수학적으로 완벽히 일정하게 증가함 을 의미) 발산, 면적값이 일정한데, 생존칸
 * 면적값이 양수면 안정화. 0이면 소멸로 로직을 짜보고 내 컴퓨터 자원이 바닥나면 그때 가서 차선책을 논의할 계획(일단 판별 커트라인(증가, 일정의 판별로 100세데동안 일정히 증가만 하면 발산, 일정하면 일정 뭐 이런 느낌)은 100세데 정도로 잡음).
 * 
 * 2번째 함수) 이것도 패턴 반환과 라벨링 반환 두개로 나뉜다. 일단 반환 로직은 동일하게 가져갈건데, 대충 파일 가져와서 읽고, d_matrix형태로 전환 후 반환 이런 느낌으로.
 */


#pragma once
#include "d_matrix.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <random>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <deque>
#include <cstdlib>
#include <thread>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#define SEMPLE 2000

extern const int BOARDWIDTH;
extern const int BOARDHEIGHT;
constexpr int BIT_WIDTH = 8;  // 예: 0~255 범위 표현용

__global__ void nextGenKernel(int* current, int* next, int width, int height);

d_matrix<int> nextGen(const d_matrix<int>& current);

int countAlive(const d_matrix<int>& mat);

void generateGameOfLifeData(int filenum, double ratio);

std::vector<std::pair<d_matrix<double>, d_matrix<double>>> LoadingData();

d_matrix<int> simulateAndLabelingtopattern(const d_matrix<int>& initialPattern, int fileId);

void printProgressBar(int current, int total, std::chrono::steady_clock::time_point startTime, std::string processname);
