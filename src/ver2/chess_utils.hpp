// 빠른 체크 감지를 위한 헬퍼 함수들
#pragma once
#include "chess_2.hpp"

namespace chess_utils {
    
    // 킹 위치를 빠르게 찾기 위한 캐시
    struct KingPositions {
        int whiteKingRank = -1, whiteKingFile = -1;
        int blackKingRank = -1, blackKingFile = -1;
        bool isValid = false;
        
        void update(const d2::d_matrix_2<piece>& board) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    if (board(i, j).T == pieceType::KING) {
                        if (board(i, j).C == pieceColor::WHITE) {
                            whiteKingRank = i;
                            whiteKingFile = j;
                        } else {
                            blackKingRank = i;
                            blackKingFile = j;
                        }
                    }
                }
            }
            isValid = true;
        }
    };
    
    // 단일 기물의 위협 영역을 빠르게 계산
    inline bool isSquareAttackedByPiece(
        int targetRank, int targetFile,
        int pieceRank, int pieceFile,
        pieceType type, pieceColor color,
        const d2::d_matrix_2<piece>& board
    ) {
        int dr = targetRank - pieceRank;
        int df = targetFile - pieceFile;
        
        switch (type) {
            case pieceType::PAWN: {
                int direction = (color == pieceColor::WHITE) ? -1 : 1;
                return (dr == direction && abs(df) == 1);
            }
            
            case pieceType::KING:
                return (abs(dr) <= 1 && abs(df) <= 1 && (dr != 0 || df != 0));
                
            case pieceType::KNIGHT:
                return (abs(dr) == 2 && abs(df) == 1) || (abs(dr) == 1 && abs(df) == 2);
                
            case pieceType::ROOK:
                if (dr == 0 || df == 0) {
                    return isLineClear(pieceRank, pieceFile, targetRank, targetFile, board);
                }
                return false;
                
            case pieceType::BISHOP:
                if (abs(dr) == abs(df)) {
                    return isLineClear(pieceRank, pieceFile, targetRank, targetFile, board);
                }
                return false;
                
            case pieceType::QUEEN:
                if (dr == 0 || df == 0 || abs(dr) == abs(df)) {
                    return isLineClear(pieceRank, pieceFile, targetRank, targetFile, board);
                }
                return false;
                
            default:
                return false;
        }
    }
    
    // 두 점 사이의 직선이 비어있는지 확인
    inline bool isLineClear(
        int fromRank, int fromFile,
        int toRank, int toFile,
        const d2::d_matrix_2<piece>& board
    ) {
        int dr = (toRank > fromRank) ? 1 : (toRank < fromRank) ? -1 : 0;
        int df = (toFile > fromFile) ? 1 : (toFile < fromFile) ? -1 : 0;
        
        int r = fromRank + dr;
        int f = fromFile + df;
        
        while (r != toRank || f != toFile) {
            if (board(r, f).T != pieceType::NONE) {
                return false;
            }
            r += dr;
            f += df;
        }
        return true;
    }
    
    // 특정 칸이 특정 색에 의해 공격받는지 빠르게 확인
    inline bool isSquareAttacked(
        int rank, int file, 
        pieceColor byColor,
        const d2::d_matrix_2<piece>& board
    ) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                piece p = board(i, j);
                if (p.T != pieceType::NONE && p.C == byColor) {
                    if (isSquareAttackedByPiece(rank, file, i, j, p.T, p.C, board)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    // 이동 후 킹이 체크상태인지 빠르게 확인
    inline bool wouldKingBeInCheck(
        const PGN& move,
        pieceColor kingColor,
        KingPositions& kingPos,
        d2::d_matrix_2<piece>& board
    ) {
        // 임시로 이동 실행
        piece movingPiece = board(move.fromRank, move.fromFile);
        piece capturedPiece = board(move.toRank, move.toFile);
        
        board(move.toRank, move.toFile) = movingPiece;
        board(move.fromRank, move.fromFile) = piece();
        
        // 킹 위치 업데이트 (킹이 움직인 경우)
        int checkRank, checkFile;
        if (move.type == pieceType::KING) {
            checkRank = move.toRank;
            checkFile = move.toFile;
        } else {
            checkRank = (kingColor == pieceColor::WHITE) ? kingPos.whiteKingRank : kingPos.blackKingRank;
            checkFile = (kingColor == pieceColor::WHITE) ? kingPos.whiteKingFile : kingPos.blackKingFile;
        }
        
        pieceColor enemyColor = (kingColor == pieceColor::WHITE) ? pieceColor::BLACK : pieceColor::WHITE;
        bool inCheck = isSquareAttacked(checkRank, checkFile, enemyColor, board);
        
        // 원상복구
        board(move.fromRank, move.fromFile) = movingPiece;
        board(move.toRank, move.toFile) = capturedPiece;
        
        return inCheck;
    }
}
