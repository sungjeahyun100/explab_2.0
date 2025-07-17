# explab_ver2

인공지능 및 게임 관련 주제를 실험해보기 위한 코드베이스. + 기존 explab레포의 최적화 & API재설계를 위한 레포.

빌드실행환경:Gforce Rtx 4060 labtop, 13th Gen Intel(R) Core(TM) i7-13650HX 64bit, ubuntu 24.04 LST

도커파일 실행 명령어: docker run --gpus all -it --rm -v "$(pwd)":/workspace -w /workspace explab_2

TODO:
기존 파일의 비효율성 개선(d_matrix, perceptron)
구체적인 실험 설계의 대대적인 수정

현재 진행중인 실험:

Game of Life의 예측 불가능성을 인공지능에게 풀어보게 시키기

transformer 모듈을 적용한 체스 인공지능 만들기

