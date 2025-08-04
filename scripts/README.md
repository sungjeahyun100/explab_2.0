# 컴파일 스크립트 디렉토리

이 디렉토리에는 다양한 컴포넌트를 컴파일하기 위한 스크립트들이 포함되어 있습니다.

## 스크립트 목록

### 기본 테스트
- `compile_simple_test.sh` - 기본 매트릭스 연산 테스트 컴파일
- `compile_test_matrix.sh` - 매트릭스 기능 테스트 컴파일
- `test_compile.sh` - 전체 시스템 테스트 컴파일

### 어텐션 관련
- `compile_attention.sh` - 어텐션 메커니즘 컴파일
- `compile_simple_attention.sh` - 간단한 어텐션 테스트 컴파일

### Game of Life 관련
- `compile_gol_attention.sh` - GOL + 어텐션 모델 컴파일
- `compile_safe_gol_attention.sh` - 안전한 GOL 어텐션 컴파일
- `compile_simple_gol_attention.sh` - 간단한 GOL 어텐션 테스트

## 사용법

프로젝트 루트 디렉토리에서:
```bash
cd scripts
./compile_simple_test.sh
```

또는 직접 실행:
```bash
./scripts/compile_simple_test.sh
```

## 참고사항

모든 스크립트는 실행 권한이 설정되어 있으며, CUDA 12.x 환경에서 테스트되었습니다.
