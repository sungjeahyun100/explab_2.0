#include "chess_optimized.hpp"
#include "chess_utils.hpp"

OptimizedChessboard::OptimizedChessboard(positionType pT, cudaStream_t str) : chessboard(pT, str) {
    // 킹 위치 초기화
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            piece p = board(i, j);
            if (p.T == pieceType::KING) {
                if (p.C == pieceColor::WHITE) {
                    whiteKingRank = i;
                    whiteKingFile = j;
                } else if (p.C == pieceColor::BLACK) {
                    blackKingRank = i;
                    blackKingFile = j;
                }
            }
        }
    }
}

void OptimizedChessboard::calculatePins(pieceColor kingColor) {
    currentPins.clear();
    
    int kingRank = (kingColor == pieceColor::WHITE) ? whiteKingRank : blackKingRank;
    int kingFile = (kingColor == pieceColor::WHITE) ? whiteKingFile : blackKingFile;
    
    // 8방향으로 핀 검사
    int directions[8][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1},     // 직선
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}   // 대각선
    };
    
    for (int d = 0; d < 8; d++) {
        int dr = directions[d][0];
        int df = directions[d][1];
        
        int r = kingRank + dr;
        int f = kingFile + df;
        
        int friendlyPieceRank = -1, friendlyPieceFile = -1;
        bool foundFriendly = false;
        
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            piece p = board(r, f);
            
            if (p.T != pieceType::NONE) {
                if (p.C == kingColor) {
                    if (!foundFriendly) {
                        friendlyPieceRank = r;
                        friendlyPieceFile = f;
                        foundFriendly = true;
                    } else {
                        // 두 번째 아군 기물 - 핀 불가능
                        break;
                    }
                } else {
                    // 적군 기물 발견
                    if (foundFriendly) {
                        // 핀 가능한 기물인지 확인
                        bool canPin = false;
                        if (d < 4) {  // 직선 방향
                            canPin = (p.T == pieceType::ROOK || p.T == pieceType::QUEEN);
                        } else {  // 대각선 방향
                            canPin = (p.T == pieceType::BISHOP || p.T == pieceType::QUEEN);
                        }
                        
                        if (canPin) {
                            PinInfo pin;
                            pin.pinnedRank = friendlyPieceRank;
                            pin.pinnedFile = friendlyPieceFile;
                            pin.pinnerRank = r;
                            pin.pinnerFile = f;
                            pin.deltaRank = dr;
                            pin.deltaFile = df;
                            pin.isActive = true;
                            currentPins.push_back(pin);
                        }
                    }
                    break;
                }
            }
            
            r += dr;
            f += df;
        }
    }
}

void OptimizedChessboard::calculateCheckInfo(pieceColor kingColor) {
    currentCheckInfo.inCheck = false;
    currentCheckInfo.checkerCount = 0;
    currentCheckInfo.blockingSquares = 0;
    
    int kingRank = (kingColor == pieceColor::WHITE) ? whiteKingRank : blackKingRank;
    int kingFile = (kingColor == pieceColor::WHITE) ? whiteKingFile : blackKingFile;
    
    // 킹을 공격하는 기물들 찾기
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            piece p = board(i, j);
            if (p.T == pieceType::NONE || p.C == kingColor) continue;
            
            // 이 기물이 킹을 공격할 수 있는지 확인
            if (chess_utils::isSquareAttackedByPiece(kingRank, kingFile, i, j, p.T, p.C, board)) {
                currentCheckInfo.inCheck = true;
                if (currentCheckInfo.checkerCount < 2) {
                    currentCheckInfo.checkerRank[currentCheckInfo.checkerCount] = i;
                    currentCheckInfo.checkerFile[currentCheckInfo.checkerCount] = j;
                }
                currentCheckInfo.checkerCount++;
            }
        }
    }
    
    // 단일 체크인 경우 블로킹 스퀘어 계산
    if (currentCheckInfo.checkerCount == 1) {
        int checkerR = currentCheckInfo.checkerRank[0];
        int checkerF = currentCheckInfo.checkerFile[0];
        piece checker = board(checkerR, checkerF);
        
        // 슬라이딩 피스인 경우 중간 칸들을 블로킹 스퀘어로 설정
        if (checker.T == pieceType::ROOK || checker.T == pieceType::BISHOP || checker.T == pieceType::QUEEN) {
            int dr = (kingRank > checkerR) ? 1 : (kingRank < checkerR) ? -1 : 0;
            int df = (kingFile > checkerF) ? 1 : (kingFile < checkerF) ? -1 : 0;
            
            int r = checkerR + dr;
            int f = checkerF + df;
            
            while (r != kingRank || f != kingFile) {
                if (r >= 0 && r < 8 && f >= 0 && f < 8) {
                    currentCheckInfo.blockingSquares |= (1ULL << (r * 8 + f));
                }
                r += dr;
                f += df;
            }
        }
        
        // 체커를 잡는 것도 가능한 수
        currentCheckInfo.blockingSquares |= (1ULL << (checkerR * 8 + checkerF));
    }
}

bool OptimizedChessboard::wouldBeInCheckFast(const PGN& move, pieceColor kingColor) {
    // 킹이 움직이는 경우
    if (move.type == pieceType::KING) {
        // 목적지가 공격받는지 확인
        pieceColor enemyColor = (kingColor == pieceColor::WHITE) ? pieceColor::BLACK : pieceColor::WHITE;
        
        // 임시로 킹을 이동시키고 확인
        piece tempKing = board(move.fromRank, move.fromFile);
        piece tempDest = board(move.toRank, move.toFile);
        
        board(move.toRank, move.toFile) = tempKing;
        board(move.fromRank, move.fromFile) = piece();
        
        bool wouldBeCheck = false;
        
        // 적군 기물들이 새로운 킹 위치를 공격하는지 확인
        for (int i = 0; i < 8 && !wouldBeCheck; i++) {
            for (int j = 0; j < 8 && !wouldBeCheck; j++) {
                piece p = board(i, j);
                if (p.T != pieceType::NONE && p.C == enemyColor) {
                    if (chess_utils::isSquareAttackedByPiece(move.toRank, move.toFile, i, j, p.T, p.C, board)) {
                        wouldBeCheck = true;
                    }
                }
            }
        }
        
        // 원상복구
        board(move.fromRank, move.fromFile) = tempKing;
        board(move.toRank, move.toFile) = tempDest;
        
        return wouldBeCheck;
    }
    
    // 킹이 아닌 기물의 경우
    // 1. 핀된 기물인지 확인
    if (isPinnedPiece(move.fromRank, move.fromFile, kingColor)) {
        return !isValidPinnedMove(move);
    }
    
    // 2. 디스커버드 어택 확인 (간단한 버전)
    // 현재 체크 상태가 아니고, 이동으로 인해 킹이 노출되는지 확인
    
    return false;  // 대부분의 경우 안전
}

bool OptimizedChessboard::isPinnedPiece(int rank, int file, pieceColor color) {
    for (const auto& pin : currentPins) {
        if (pin.isActive && pin.pinnedRank == rank && pin.pinnedFile == file) {
            return true;
        }
    }
    return false;
}

bool OptimizedChessboard::isValidPinnedMove(const PGN& move) {
    for (const auto& pin : currentPins) {
        if (pin.pinnedRank == move.fromRank && pin.pinnedFile == move.fromFile) {
            // 핀 방향으로만 움직일 수 있음
            int moveDr = move.toRank - move.fromRank;
            int moveDf = move.toFile - move.fromFile;
            
            // 움직임이 핀 방향과 평행한지 확인
            if (pin.deltaRank == 0) {
                return moveDr == 0;  // 수평 핀
            } else if (pin.deltaFile == 0) {
                return moveDf == 0;  // 수직 핀
            } else {
                // 대각선 핀
                return (moveDr * pin.deltaFile == moveDf * pin.deltaRank);
            }
        }
    }
    return true;  // 핀되지 않은 기물
}

void OptimizedChessboard::genLegalMovesOptimized() {
    // 먼저 의사 합법 수 생성 (기존 genLegalMoves의 첫 부분)
    genLegalMoves();  // 기존 메서드로 의사 합법 수 생성
    
    std::vector<PGN> optimizedLegalMoves;
    
    // 현재 플레이어 결정 (로그 크기로 판단)
    pieceColor currentPlayer = (log.size() % 2 == 0) ? pieceColor::WHITE : pieceColor::BLACK;
    
    // 핀과 체크 정보 계산
    calculatePins(currentPlayer);
    calculateCheckInfo(currentPlayer);
    
    if (currentCheckInfo.inCheck) {
        if (currentCheckInfo.checkerCount == 1) {
            // 단일 체크: 킹 이동, 체커 잡기, 블로킹
            for (const auto& move : moveList) {
                if (move.color != currentPlayer) continue;
                
                if (move.type == pieceType::KING) {
                    // 킹 이동은 빠른 체크 테스트
                    if (!wouldBeInCheckFast(move, currentPlayer)) {
                        optimizedLegalMoves.push_back(move);
                    }
                } else {
                    // 블로킹하거나 체커를 잡는 수인지 확인
                    uint64_t moveSquare = 1ULL << (move.toRank * 8 + move.toFile);
                    if (currentCheckInfo.blockingSquares & moveSquare) {
                        if (!isPinnedPiece(move.fromRank, move.fromFile, currentPlayer) ||
                            isValidPinnedMove(move)) {
                            optimizedLegalMoves.push_back(move);
                        }
                    }
                }
            }
        } else {
            // 더블 체크: 킹만 움직일 수 있음
            for (const auto& move : moveList) {
                if (move.color == currentPlayer && move.type == pieceType::KING) {
                    if (!wouldBeInCheckFast(move, currentPlayer)) {
                        optimizedLegalMoves.push_back(move);
                    }
                }
            }
        }
    } else {
        // 체크가 아닌 경우: 핀된 기물만 특별 처리
        for (const auto& move : moveList) {
            if (move.color != currentPlayer) continue;
            
            if (isPinnedPiece(move.fromRank, move.fromFile, currentPlayer)) {
                if (isValidPinnedMove(move)) {
                    optimizedLegalMoves.push_back(move);
                }
            } else {
                optimizedLegalMoves.push_back(move);
            }
        }
    }
    
    moveList = std::move(optimizedLegalMoves);
}
