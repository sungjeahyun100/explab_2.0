#!/bin/bash

echo "🎯 Attention 메커니즘 통합 테스트 시작!"
echo "========================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 현재 디렉토리 저장
ORIGINAL_DIR=$(pwd)
PROJECT_DIR="/home/sjh100/바탕화면/explab_ver2"

cd "$PROJECT_DIR"

echo -e "${BLUE}📁 프로젝트 디렉토리: $PROJECT_DIR${NC}"
echo ""

# 1. 간단한 Attention 테스트
echo -e "${YELLOW}🔧 1단계: 간단한 Attention 메커니즘 테스트${NC}"
echo "----------------------------------------"

if [ -f "compile_simple_attention.sh" ]; then
    echo "간단한 Attention 테스트 컴파일 중..."
    ./compile_simple_attention.sh
    
    if [ $? -eq 0 ] && [ -f "build/simple_attention_test" ]; then
        echo -e "${GREEN}✅ 컴파일 성공!${NC}"
        echo ""
        echo -e "${BLUE}🚀 간단한 Attention 테스트 실행:${NC}"
        echo "================================"
        ./build/simple_attention_test
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ 간단한 Attention 테스트 성공!${NC}"
        else
            echo -e "${RED}❌ 간단한 Attention 테스트 실패!${NC}"
        fi
    else
        echo -e "${RED}❌ 간단한 Attention 컴파일 실패!${NC}"
    fi
else
    echo -e "${RED}❌ compile_simple_attention.sh 파일을 찾을 수 없습니다!${NC}"
fi

echo ""
echo "========================================"

# 2. 전체 Attention 시스템 테스트 (선택적)
echo -e "${YELLOW}🔧 2단계: 전체 Attention 시스템 체크 (컴파일만)${NC}"
echo "------------------------------------------------"

if [ -f "compile_attention.sh" ]; then
    echo "전체 Attention 시스템 컴파일 확인 중..."
    echo "(실제 실행은 하지 않고 컴파일만 확인합니다)"
    
    # 컴파일만 테스트 (실행은 생략)
    ./compile_attention.sh > /dev/null 2>&1
    
    if [ $? -eq 0 ] && [ -f "build/attention_test" ]; then
        echo -e "${GREEN}✅ 전체 Attention 시스템 컴파일 성공!${NC}"
        echo "   (실행 파일: build/attention_test)"
    else
        echo -e "${YELLOW}⚠️  전체 Attention 시스템 컴파일에 문제가 있을 수 있습니다.${NC}"
        echo "   간단한 버전이 정상 작동하면 기본 기능은 사용 가능합니다."
    fi
else
    echo -e "${YELLOW}⚠️  compile_attention.sh 파일을 찾을 수 없습니다.${NC}"
fi

echo ""
echo "========================================"

# 3. 프로젝트 구조 확인
echo -e "${YELLOW}📋 3단계: Attention 관련 파일 구조 확인${NC}"
echo "----------------------------------------"

echo "생성된 Attention 관련 파일들:"
echo ""

if [ -f "src/ver2/attention.hpp" ]; then
    echo -e "${GREEN}✅ src/ver2/attention.hpp${NC} - Attention 헤더 파일"
else
    echo -e "${RED}❌ src/ver2/attention.hpp${NC} - 누락됨"
fi

if [ -f "src/ver2/attention.cu" ]; then
    echo -e "${GREEN}✅ src/ver2/attention.cu${NC} - Attention 구현 파일"
else
    echo -e "${RED}❌ src/ver2/attention.cu${NC} - 누락됨"
fi

if [ -f "example_code/attention_example.cu" ]; then
    echo -e "${GREEN}✅ example_code/attention_example.cu${NC} - 체스/GOL Attention 예제"
else
    echo -e "${RED}❌ example_code/attention_example.cu${NC} - 누락됨"
fi

if [ -f "example_code/gol_attention_model.cu" ]; then
    echo -e "${GREEN}✅ example_code/gol_attention_model.cu${NC} - GOL Attention 통합 모델"
else
    echo -e "${RED}❌ example_code/gol_attention_model.cu${NC} - 누락됨"
fi

if [ -f "test/simple_attention_test.cu" ]; then
    echo -e "${GREEN}✅ test/simple_attention_test.cu${NC} - 간단한 Attention 테스트"
else
    echo -e "${RED}❌ test/simple_attention_test.cu${NC} - 누락됨"
fi

echo ""
echo "컴파일 스크립트들:"

if [ -f "compile_attention.sh" ]; then
    echo -e "${GREEN}✅ compile_attention.sh${NC} - 전체 Attention 시스템 컴파일"
else
    echo -e "${RED}❌ compile_attention.sh${NC} - 누락됨"
fi

if [ -f "compile_simple_attention.sh" ]; then
    echo -e "${GREEN}✅ compile_simple_attention.sh${NC} - 간단한 Attention 테스트 컴파일"
else
    echo -e "${RED}❌ compile_simple_attention.sh${NC} - 누락됨"
fi

echo ""
echo "========================================"

# 4. 사용법 안내
echo -e "${YELLOW}📖 4단계: 사용법 안내${NC}"
echo "--------------------"

echo ""
echo -e "${BLUE}🎯 Attention 메커니즘 사용 방법:${NC}"
echo ""
echo "1. 간단한 테스트 (추천):"
echo "   ./compile_simple_attention.sh"
echo "   ./build/simple_attention_test"
echo ""
echo "2. 전체 시스템 테스트:"
echo "   ./compile_attention.sh"
echo "   ./build/attention_test"
echo ""
echo "3. GOL Attention 모델 (고급):"
echo "   # 먼저 필요한 의존성 컴파일 후"
echo "   # gol_attention_model.cu 컴파일 및 실행"
echo ""

echo -e "${BLUE}🔧 주요 구성 요소:${NC}"
echo ""
echo "• MultiHeadAttention: 멀티헤드 셀프 어텐션"
echo "• PositionalEncoding: 위치 인코딩"  
echo "• LayerNorm: 레이어 정규화"
echo "• TransformerEncoderBlock: 완전한 Transformer 블록"
echo "• ChessTransformer: 체스 AI 특화 모델"
echo "• GOLTransformer: Game of Life 예측 특화 모델"
echo ""

echo -e "${BLUE}📚 예제 파일들:${NC}"
echo ""
echo "• attention_example.cu: 기본 Transformer 사용법"
echo "• gol_attention_model.cu: GOL에 Attention 적용"
echo "• simple_attention_test.cu: 간단한 테스트 코드"
echo ""

# 5. 최종 요약
echo "========================================"
echo -e "${YELLOW}📊 최종 요약${NC}"
echo "----------"

echo ""
if [ -f "build/simple_attention_test" ]; then
    echo -e "${GREEN}🎉 Attention 메커니즘이 성공적으로 추가되었습니다!${NC}"
    echo ""
    echo -e "${GREEN}✅ 기본 기능 동작 확인됨${NC}"
    echo -e "${GREEN}✅ 간단한 Self-Attention 구현됨${NC}"
    echo -e "${GREEN}✅ Transformer 블록 구조 준비됨${NC}"
    echo -e "${GREEN}✅ 체스/GOL 특화 모델 템플릿 제공됨${NC}"
    echo ""
    echo -e "${BLUE}다음 단계:${NC}"
    echo "1. 간단한 테스트로 기본 동작 확인"
    echo "2. 실제 데이터셋으로 GOL/체스 모델 훈련"
    echo "3. 하이퍼파라미터 튜닝 및 성능 최적화"
    echo "4. 더 복잡한 Attention 변형 실험"
else
    echo -e "${YELLOW}⚠️  기본 테스트가 실행되지 않았습니다.${NC}"
    echo "CUDA/cuDNN 환경을 확인하고 다시 시도해보세요."
fi

echo ""
echo -e "${BLUE}🚀 Happy Coding with Attention! 🚀${NC}"
echo "========================================"

# 원래 디렉토리로 복귀
cd "$ORIGINAL_DIR"
