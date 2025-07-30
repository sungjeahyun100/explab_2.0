#pragma once
#include "chess_2.hpp"

// 핀 정보를 저장하는 구조체
struct PinInfo {
    int pinnedRank, pinnedFile;
    int pinnerRank, pinnerFile;
    int deltaRank, deltaFile;  // 핀 방향
    bool isActive;
};

// 체크 회피를 위한 정보
struct CheckInfo {
    bool inCheck;
    int checkerCount;
    int checkerRank[2], checkerFile[2];  // 최대 2개까지
    uint64_t blockingSquares;  // 체크를 막을 수 있는 칸들
};

class OptimizedChessboard : public chessboard {
private:
    std::vector<PinInfo> currentPins;
    CheckInfo currentCheckInfo;
    
    // 빠른 킹 위치 추적
    int whiteKingRank, whiteKingFile;
    int blackKingRank, blackKingFile;
    
public:
    OptimizedChessboard(positionType pT, cudaStream_t str);
    
    // 핀 정보 사전 계산
    void calculatePins(pieceColor kingColor);
    
    // 체크 정보 계산
    void calculateCheckInfo(pieceColor kingColor);
    
    // 더 효율적인 합법 수 생성
    void genLegalMovesOptimized();
    
    // 빠른 체크 테스트 (전체 재계산 없이)
    bool wouldBeInCheckFast(const PGN& move, pieceColor kingColor);
    
    // 핀된 기물인지 확인
    bool isPinnedPiece(int rank, int file, pieceColor color);
    
    // 핀 방향으로만 움직일 수 있는지 확인
    bool isValidPinnedMove(const PGN& move);
};
