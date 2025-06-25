#pragma once
#include "d_matrix.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <regex>
#include <unordered_map>
#include <cstdint>
#include <cctype>


enum class pieceType{
    KING,
    QUEEN,
    BISHOP,
    KNIGHT,
    ROOK,
    PAWN,
    NONE
};

enum class pieceColor{
    WHITE,
    BLACK,
    NONE
};

enum class squareType{
    NONE,
    BLACK_THREAT,
    WHITE_THREAT,
    BOTH_THREAT
};

enum class specialMove{
    KING_SIDE_CASTLING,
    QUEEN_SIDE_CASTLING,
    ENPASSANT,
    PROMOTION,
    NONE
};

struct PGNnotation{
    pieceColor color = pieceColor::NONE;
    pieceType type = pieceType::NONE;
    int fromFile = -1;
    int fromRank = -1;
    int toFile = -1;
    int toRank = -1;
    pieceType promotion = pieceType::NONE;
    specialMove stype = specialMove::NONE;
    bool operator==(const PGNnotation& o) const {
        return color==o.color && type==o.type
            && fromFile==o.fromFile && fromRank==o.fromRank
            && toFile==o.toFile     && toRank==o.toRank;
    }
};
using PGN = PGNnotation;

class piece{
    public:
        piece() = default;
        piece(pieceColor c, pieceType t, int r, int f);
        pieceColor C = pieceColor::NONE;
        pieceType T = pieceType::NONE;
        squareType sT = squareType::NONE;
        int rank = -1;
        int file = -1;
};

//디버깅용 포지션 생성기

enum class positionType{
    DEFAULT,//체스 시작 포지션
    TEST_WHITE_ENPASSANT,
    TEST_BLACK_ENPASSANT,
    TEST_WHITE_QUEEN_SIDE_CASTLING,
    TEST_WHITE_KING_SIDE_CASTLING,
    TEST_BLACK_QUEEN_SIDE_CASTLING,
    TEST_BLACK_KING_SIDE_CASTLING,
    TEST_PIECE_QUEEN,
    TEST_PIECE_KING,
    TEST_PIECE_KNIGHT,
    TEST_PIECE_ROOK,
    TEST_PIECE_BISHOP,
    TEST_PIECE_PAWN,
    TEST_WHITE_CHECK,
    TEST_BLACK_CHECK
};

struct testPosition{
    d_matrix<piece> testboard;
    std::vector<PGN> log = {};
    testPosition();
};
using testP = testPosition;

extern testP default_p;
extern testP test_white_enpassant_p;
extern testP test_black_enpassant_p;
extern testP test_white_queen_side_castling_p;
extern testP test_white_king_side_castling_p;
extern testP test_black_queen_side_castling_p;
extern testP test_black_king_side_castling_p;
extern testP test_piece_queen_p;
extern testP test_piece_king_p;
extern testP test_piece_knight_p;
extern testP test_piece_rook_p;
extern testP test_piece_bishop_p;
extern testP test_piece_pawn_p;
extern testP test_white_check_p;
extern testP test_black_check_p;

extern std::unordered_map<positionType, testP> positionTableForDebug;

class chessboard{
    private:
        d_matrix<piece> board;
        std::vector<PGN> log;
        std::vector<PGN> moveList;
        std::vector<PGN> whiteThreatsq;
        std::vector<PGN> blackThreatsq;
        positionType ptype;
    public:
        chessboard(positionType pT);
        void loadPosition(const testP& t);
        void genLegalMoves();
        void calculateThreatSquare(pieceColor color);
        const std::vector<PGN>& getMoveList() const { return moveList; }
        const std::vector<PGN>& getWhiteThreatSq() const { return whiteThreatsq; }
        const std::vector<PGN>& getBlackThreatSq() const { return blackThreatsq; }
        bool isBlackChacked();
        bool isWhiteChacked();
        void updateBoard(const PGN& mv);
        void printBoard();
};

void debugChessboardvar(chessboard* board);
PGN translatePGN(const std::string& mv);
std::string translateMove(const PGN& mv);

