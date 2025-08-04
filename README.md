# explab_ver2

CUDA 기반 딥러닝 라이브러리 및 AI 게임 실험 프로젝트

**빌드실행환경**: GeForce RTX 4060 laptop, 13th Gen Intel(R) Core(TM) i7-13650HX 64bit, Ubuntu 24.04 LTS  
**언어 표준**: C++20  
**CUDA 버전**: 12.x 이상 권장

## 프로젝트 구조

```
explab_ver2/
├── src/
│   ├── ver1/           # 레거시 구현
│   └── ver2/           # 최적화된 구현 (메인)
│       ├── d_matrix_2.hpp      # 매트릭스 클래스 선언
│       ├── d_matrix_2.cu       # 매트릭스 클래스 구현
│       ├── perceptron_2.hpp    # 신경망 레이어
│       ├── attention.hpp       # 어텐션 메커니즘 선언
│       ├── attention.cu        # 어텐션 구현
│       ├── GOLdatabase_2.hpp   # Game of Life 데이터베이스
│       └── GOLdatabase_2.cu    # GOL 구현
├── test/               # 테스트 파일들
├── scripts/            # 컴파일 스크립트들
├── build/              # 빌드 결과물
├── dataset/            # 데이터셋
└── example_code/       # 예제 코드
```

## 최근 업데이트 (v2.1 - 2025.08.05)

### ✅ 코드 분리 및 최적화 완료
- **헤더/구현 분리**: `d_matrix_2.hpp`의 모든 함수 구현을 `.cu` 파일로 분리
- **컴파일 최적화**: 템플릿 인스턴스화 명시적 처리로 링킹 성능 개선
- **검증 완료**: 모든 주요 매트릭스 연산 함수 테스트 통과

### 🔧 검증된 함수들
- `HadamardProduct` - 원소별 곱셈 ✅
- `ScalaProduct` - 스칼라 곱셈 ✅
- `matrixPlus` - 매트릭스 덧셈 ✅
- `matrixMP` - 매트릭스 곱셈 ✅
- `softmax` - 소프트맥스 활성화 ✅
- `InitWeight` - 가중치 초기화 ✅
- `concatenate` - 매트릭스 연결 ✅

### 🗂️ 프로젝트 정리
- 백업 파일 정리 완료
- 컴파일 스크립트를 `scripts/` 디렉토리로 통합
- 테스트 파일 정리 및 구조화

현재 진행중인 실험:

Game of Life의 예측 불가능성을 인공지능에게 풀어보게 시키기

transformer 모듈을 적용한 체스 인공지능 만들기

## 새로 추가된 기능: Attention 메커니즘

### Multi-Head Self-Attention
- `src/ver2/attention.hpp/cu`: 완전한 Transformer 아키텍처 구현
- CUDA 가속 Self-Attention 메커니즘
- 위치 인코딩 (Positional Encoding)
- Layer Normalization
- Feed Forward Network

### 특화된 모델들
- **ChessTransformer**: 체스 AI를 위한 Transformer 모델
- **GOLTransformer**: Game of Life 예측을 위한 Transformer 모델
- **TransformerModel**: 범용 Transformer 베이스 클래스

### 주요 구성 요소
1. **MultiHeadAttention**: 멀티헤드 어텐션 메커니즘
2. **PositionalEncoding**: 시퀀스 위치 정보 인코딩
3. **LayerNorm**: 레이어 정규화
4. **FeedForwardNetwork**: 피드포워드 네트워크
5. **TransformerEncoderBlock**: 완전한 Transformer 인코더 블록

### 컴파일 및 실행
```bash
# 간단한 Attention 테스트 (추천)
./compile_simple_attention.sh
./build/simple_attention_test

# 전체 Attention 시스템 (고급)
./compile_attention.sh
./build/attention_test

# GOL Attention 모델
./compile_gol_attention.sh
./build/gol_attention_model

# 통합 테스트 실행
./test_attention.sh
```

### 핵심 기능
- **CUDA 가속**: 모든 행렬 연산이 GPU에서 수행
- **메모리 최적화**: 스트림 기반 비동기 처리
- **모듈화 설계**: 재사용 가능한 레이어 구조
- **C++20 활용**: 최신 언어 기능 적극 활용

### 사용 예제
- `example_code/attention_example.cu`: 체스 AI 및 GOL 예측 예제
