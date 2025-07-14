// 체스판 인덱스 시각화 (row, col):
//   파일:   0 1 2 3 4 5 6 7   (a b c d e f g h)
// 랭크(행): 0 1 2 3 4 5 6 7   (8 7 6 5 4 3 2 1)
// 예: board(0,0) = a8, board(7,7) = h1
//
#include "chess.hpp"

#define MAX_MOVES_PER_PIECE 27  // 퀸이 최악 슬라이딩 시 최대 이동수

testPosition::testPosition() : testboard(8, 8) {
    log.clear();
}

// ──── DEFAULT 포지션 ────
static testP make_default_p() {
    testP p;
    // 흑 폰
    for (int i = 0; i < 8; ++i) {
        p.testboard(1, i) = piece(pieceColor::BLACK, pieceType::PAWN, 1, i);
    }
    // 백 폰
    for (int i = 0; i < 8; ++i) {
        p.testboard(6, i) = piece(pieceColor::WHITE, pieceType::PAWN, 6, i);
    }
    // 기타 기물
    // 흑 백라인
    p.testboard(0, 0) = piece(pieceColor::BLACK, pieceType::ROOK,   0, 0);
    p.testboard(0, 1) = piece(pieceColor::BLACK, pieceType::KNIGHT, 0, 1);
    p.testboard(0, 2) = piece(pieceColor::BLACK, pieceType::BISHOP, 0, 2);
    p.testboard(0, 3) = piece(pieceColor::BLACK, pieceType::QUEEN,  0, 3);
    p.testboard(0, 4) = piece(pieceColor::BLACK, pieceType::KING,   0, 4);
    p.testboard(0, 5) = piece(pieceColor::BLACK, pieceType::BISHOP, 0, 5);
    p.testboard(0, 6) = piece(pieceColor::BLACK, pieceType::KNIGHT, 0, 6);
    p.testboard(0, 7) = piece(pieceColor::BLACK, pieceType::ROOK,   0, 7);
    // 백 라인
    p.testboard(7, 0) = piece(pieceColor::WHITE, pieceType::ROOK,   7, 0);
    p.testboard(7, 1) = piece(pieceColor::WHITE, pieceType::KNIGHT, 7, 1);
    p.testboard(7, 2) = piece(pieceColor::WHITE, pieceType::BISHOP, 7, 2);
    p.testboard(7, 3) = piece(pieceColor::WHITE, pieceType::QUEEN,  7, 3);
    p.testboard(7, 4) = piece(pieceColor::WHITE, pieceType::KING,   7, 4);
    p.testboard(7, 5) = piece(pieceColor::WHITE, pieceType::BISHOP, 7, 5);
    p.testboard(7, 6) = piece(pieceColor::WHITE, pieceType::KNIGHT, 7, 6);
    p.testboard(7, 7) = piece(pieceColor::WHITE, pieceType::ROOK,   7, 7);

    // 기본 포지션이니까 이동 로그는 비워둠
    p.log.clear();
    return p;
}

// ──── TEST_WHITE_ENPASSANT 포지션 ────
// 중앙열에 백 폰(e5)과 흑 폰(d5)를 배치. (행 3열 4, 행 3열 3)
// "we5d6" 순서대로 로그에 넣으면 다음 수로 흑이 d7→d5 두 칸 전진했을 때 설정됨
static testP make_test_white_enpassant_p() {
    testP p;
    // 폰만 놓는다; 빈칸은 기본적으로 NONE
    int rank = 3;           // 0-based에서 4번째 행 (백 관점 e5/d5)
    int white_file = 4;     // 'e'
    int black_file = 3;     // 'd'
    // 백 폰을 e5에 배치
    p.testboard(rank, white_file) = piece(pieceColor::WHITE, pieceType::PAWN, rank, white_file);
    // 흑 폰을 d5에 배치
    p.testboard(rank, black_file) = piece(pieceColor::BLACK, pieceType::PAWN, rank, black_file);

    // 이제 흑이 d7(d 파일 1행)에서 d5(3행)로 두 칸 전진했다는 로그 작성
    // fromFile=3, fromRank=1 → toFile=3, toRank=3
    p.log.push_back(PGN{ pieceColor::BLACK, pieceType::PAWN, 3, 1, 3, 3, pieceType::NONE, specialMove::NONE });
    return p;
}

// ──── TEST_BLACK_ENPASSANT 포지션 ────
// 중앙열에 백 폰(d4)과 흑 폰(e4) 배치. (행 4열 3, 행 4열 4)
// "bd4e3" 순서대로 로그에 넣으면 백이 d2→d4 두 칸 전진했을 때 설정
static testP make_test_black_enpassant_p() {
    testP p;
    int rank = 4;           // 0-based에서 5번째 행 (흑 관점 d4/e4)
    int white_file = 3;     // 'd'
    int black_file = 4;     // 'e'
    // 백 폰을 d4에 배치
    p.testboard(rank, white_file) = piece(pieceColor::WHITE, pieceType::PAWN, rank, white_file);
    // 흑 폰을 e4에 배치
    p.testboard(rank, black_file) = piece(pieceColor::BLACK, pieceType::PAWN, rank, black_file);

    // 백이 d2(d 파일 6행)에서 d4(4행)로 두 칸 전진했다는 로그 작성
    // fromFile=3, fromRank=6 → toFile=3, toRank=4
    p.log.push_back(PGN{ pieceColor::WHITE, pieceType::PAWN, 3, 6, 3, 4, pieceType::NONE, specialMove::NONE });
    return p;
}

// ──── TEST_WHITE_QUEEN_SIDE_CASTLING 포지션 ────
// 백 퀸사이드 캐슬링: 백 킹(e1,7행4열)과 백 퀸사이드 룩(a1,7행0열)만 배치
static testP make_test_white_queen_side_castling_p() {
    testP p;
    int king_rank = 7, king_file = 4;  // e1
    int rook_file = 0;                 // a1
    // 백 왕
    p.testboard(king_rank, king_file) = piece(pieceColor::WHITE, pieceType::KING, king_rank, king_file);
    // 백 퀸사이드 룩
    p.testboard(king_rank, rook_file) = piece(pieceColor::WHITE, pieceType::ROOK, king_rank, rook_file);

    p.log.clear();
    return p;
}

// ──── TEST_WHITE_KING_SIDE_CASTLING 포지션 ────
// 백 킹사이드 캐슬링: 백 킹(e1)과 백 킹사이드 룩(h1,7행7열)만 배치
static testP make_test_white_king_side_castling_p() {
    testP p;
    int king_rank = 7, king_file = 4;  // e1
    int rook_file = 7;                 // h1
    // 백 왕
    p.testboard(king_rank, king_file) = piece(pieceColor::WHITE, pieceType::KING, king_rank, king_file);
    // 백 킹사이드 룩
    p.testboard(king_rank, rook_file) = piece(pieceColor::WHITE, pieceType::ROOK, king_rank, rook_file);

    p.log.clear();
    return p;
}

// ──── TEST_BLACK_QUEEN_SIDE_CASTLING 포지션 ────
// 흑 퀸사이드 캐슬링: 흑 왕(e8,0행4열)과 흑 퀸사이드 룩(a8,0행0열)만 배치
static testP make_test_black_queen_side_castling_p() {
    testP p;
    int king_rank = 0, king_file = 4;  // e8
    int rook_file = 0;                 // a8
    // 흑 왕
    p.testboard(king_rank, king_file) = piece(pieceColor::BLACK, pieceType::KING, king_rank, king_file);
    // 흑 퀸사이드 룩
    p.testboard(king_rank, rook_file) = piece(pieceColor::BLACK, pieceType::ROOK, king_rank, rook_file);

    p.log.clear();
    return p;
}

// ──── TEST_BLACK_KING_SIDE_CASTLING 포지션 ────
// 흑 킹사이드 캐슬링: 흑 왕(e8)과 흑 킹사이드 룩(h8,0행7열)만 배치
static testP make_test_black_king_side_castling_p() {
    testP p;
    int king_rank = 0, king_file = 4;  // e8
    int rook_file = 7;                 // h8
    // 흑 왕
    p.testboard(king_rank, king_file) = piece(pieceColor::BLACK, pieceType::KING, king_rank, king_file);
    // 흑 킹사이드 룩
    p.testboard(king_rank, rook_file) = piece(pieceColor::BLACK, pieceType::ROOK, king_rank, rook_file);

    p.log.clear();
    return p;
}

// ──── TEST_PIECE_QUEEN 포지션 ────
// 자유로운 위치에 퀸 한 개만 배치 (예: d4 → 4행,3열), 이동 로그는 없음
static testP make_test_piece_queen_p() {
    testP p;
    int rank = 4, file = 3;  // 'd4'
    p.testboard(rank, file) = piece(pieceColor::WHITE, pieceType::QUEEN, rank, file);
    p.log.clear();
    return p;
}

// ──── TEST_PIECE_KING 포지션 ────
// 자유로운 위치에 킹 한 개만 배치 (예: e5 → 3행,4열), 이동 로그는 없음
static testP make_test_piece_king_p() {
    testP p;
    int rank = 3, file = 4;  // 'e5'
    p.testboard(rank, file) = piece(pieceColor::WHITE, pieceType::KING, rank, file);
    p.log.clear();
    return p;
}

// ──── TEST_PIECE_KNIGHT 포지션 ────
// 자유로운 위치에 나이트 한 개만 배치 (예: g6 → 2행,6열), 이동 로그는 없음
static testP make_test_piece_knight_p() {
    testP p;
    int rank = 2, file = 6;  // 'g6'
    p.testboard(rank, file) = piece(pieceColor::WHITE, pieceType::KNIGHT, rank, file);
    p.log.clear();
    return p;
}

// ──── TEST_PIECE_ROOK 포지션 ────
// 자유로운 위치에 룩 한 개만 배치 (예: a5 → 3행,0열), 이동 로그는 없음
static testP make_test_piece_rook_p() {
    testP p;
    int rank = 3, file = 0;  // 'a5'
    p.testboard(rank, file) = piece(pieceColor::WHITE, pieceType::ROOK, rank, file);
    p.log.clear();
    return p;
}

// ──── TEST_PIECE_BISHOP 포지션 ────
// 자유로운 위치에 비숍 한 개만 배치 (예: f3 → 5행,5열), 이동 로그는 없음
static testP make_test_piece_bishop_p() {
    testP p;
    int rank = 5, file = 5;  // 'f3'
    p.testboard(rank, file) = piece(pieceColor::WHITE, pieceType::BISHOP, rank, file);
    p.log.clear();
    return p;
}

// ──── TEST_PIECE_PAWN 포지션 ────
// 자유로운 위치에 폰 한 개만 배치 (예: h2 → 6행,7열), 이동 로그는 없음
static testP make_test_piece_pawn_p() {
    testP p;
    int rank = 6, file = 7;  // 'h2'
    p.testboard(rank, file) = piece(pieceColor::WHITE, pieceType::PAWN, rank, file);
    p.log.clear();
    return p;
}

static testP make_test_white_check_p(){
    testP p;
    p.testboard(0, 1) = piece(pieceColor::WHITE, pieceType::KING, 0, 1);
    p.testboard(7, 7) = piece(pieceColor::BLACK, pieceType::QUEEN, 7, 7);
    p.log.clear();
    return p;
}

static testP make_test_black_check_p(){
    testP p;
    p.testboard(0, 1) = piece(pieceColor::BLACK, pieceType::KING, 0, 1);
    p.testboard(7, 7) = piece(pieceColor::WHITE, pieceType::QUEEN, 7, 7);
    p.log.clear();
    return p;
}

// 전역 변수 정의
testP default_p                    = make_default_p();
testP test_white_enpassant_p       = make_test_white_enpassant_p();
testP test_black_enpassant_p       = make_test_black_enpassant_p();
testP test_white_queen_side_castling_p = make_test_white_queen_side_castling_p();
testP test_white_king_side_castling_p  = make_test_white_king_side_castling_p();
testP test_black_queen_side_castling_p = make_test_black_queen_side_castling_p();
testP test_black_king_side_castling_p  = make_test_black_king_side_castling_p();
testP test_piece_queen_p           = make_test_piece_queen_p();
testP test_piece_king_p            = make_test_piece_king_p();
testP test_piece_knight_p          = make_test_piece_knight_p();
testP test_piece_rook_p            = make_test_piece_rook_p();
testP test_piece_bishop_p          = make_test_piece_bishop_p();
testP test_piece_pawn_p            = make_test_piece_pawn_p();
testP test_white_check_p           = make_test_white_check_p();
testP test_black_check_p           = make_test_black_check_p();

// positionType → testPosition 맵 초기화
std::unordered_map<positionType, testP> positionTableForDebug = {
    { positionType::DEFAULT,                     default_p },
    { positionType::TEST_WHITE_ENPASSANT,         test_white_enpassant_p },
    { positionType::TEST_BLACK_ENPASSANT,         test_black_enpassant_p },
    { positionType::TEST_WHITE_QUEEN_SIDE_CASTLING, test_white_queen_side_castling_p },
    { positionType::TEST_WHITE_KING_SIDE_CASTLING,  test_white_king_side_castling_p },
    { positionType::TEST_BLACK_QUEEN_SIDE_CASTLING, test_black_queen_side_castling_p },
    { positionType::TEST_BLACK_KING_SIDE_CASTLING,  test_black_king_side_castling_p },
    { positionType::TEST_PIECE_QUEEN,             test_piece_queen_p },
    { positionType::TEST_PIECE_KING,              test_piece_king_p },
    { positionType::TEST_PIECE_KNIGHT,            test_piece_knight_p },
    { positionType::TEST_PIECE_ROOK,              test_piece_rook_p },
    { positionType::TEST_PIECE_BISHOP,            test_piece_bishop_p },
    { positionType::TEST_PIECE_PAWN,              test_piece_pawn_p },
    { positionType::TEST_WHITE_CHECK,             test_white_check_p },
    { positionType::TEST_BLACK_CHECK,             test_black_check_p }
};

piece::piece(pieceColor c, pieceType t, int r, int f)
{
    C = c;
    T = t;
    rank = r;
    file = f;
}

chessboard::chessboard(positionType pT) : board(8, 8)
{
    ptype = pT;
    auto &pos = positionTableForDebug[ptype];
    loadPosition(pos);
}

void chessboard::loadPosition(const testP &t)
{
    board = t.testboard;
    log = t.log;
    board.cpyToDev();
}



bool chessboard::isBlackChacked()
{
    for(int i = 0; i < 8; i++){
        for(int j = 0; j < 8; j++){
            if(board(i, j).T == pieceType::KING && board(i, j).C == pieceColor::BLACK && (board(i, j).sT == squareType::WHITE_THREAT || board(i, j).sT == squareType::BOTH_THREAT)) return true;
        }
    }
    return false;
}

bool chessboard::isWhiteChacked()
{
    for(int i = 0; i < 8; i++){
        for(int j = 0; j < 8; j++){
            if(board(i, j).T == pieceType::KING && board(i, j).C == pieceColor::WHITE && (board(i, j).sT == squareType::BLACK_THREAT || board(i, j).sT == squareType::BOTH_THREAT)) return true; 
        }
    }
    return false;
}

// Kernel: 각 스레드가 한 칸(board index) 담당
__global__ void genLegalMovesKernel(
    const piece* __restrict__ board_ptr,
    PGN*         __restrict__ moveBuf,
    int*         __restrict__ moveCount,
    PGN*         __restrict__ log,
    int*         __restrict__ logsize,
    PGN*         __restrict__ blackThreat,
    int*         __restrict__ blackThreatSize,
    PGN*         __restrict__ whiteThreat,
    int*         __restrict__ whiteThreatSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 64) return;

    __shared__ PGN threadLocalBuf[32][MAX_MOVES_PER_PIECE];

    const piece p = board_ptr[idx];
    if (p.T == pieceType::NONE || p.C == pieceColor::NONE) {
        return;
    }

    const int l_size = *logsize;
    PGN last_Log;

    if(l_size > 0){
        last_Log = log[l_size - 1];
    }else{
        last_Log = PGN{};
    }

    int r0 = p.rank;
    int f0 = p.file;
    int localCnt = 0;

    switch (p.T) {
        case pieceType::KING: {
            // --- (A) 주변 한 칸 이동 가능해서 생기는 일반 King 이동
            for (int dr = -1; dr <= 1; ++dr) {
                for (int df = -1; df <= 1; ++df) {
                    if (dr == 0 && df == 0) continue;
                    int r1 = r0 + dr;
                    int f1 = f0 + df;
                    if (r1 < 0 || r1 > 7 || f1 < 0 || f1 > 7) continue;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) continue;
                    threadLocalBuf[threadIdx.x][localCnt++] 
                       = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE, specialMove::NONE};
                }
            }
        
            // --- (B) 캐슬링 관련 사전 준비
            int checkRank = (p.C == pieceColor::WHITE ? 7 : 0);
        
            // (B1) 로그를 순회하며 같은 색 King/QueenSideRook/KingSideRook이 움직였는지 검사
            bool kingMoved   = false;
            bool rookA_moved = false; // 퀸사이드 룩 (a열) 이동 여부
            bool rookH_moved = false; // 킹사이드 룩 (h열) 이동 여부
            for (int j = 0; j < l_size; ++j) {
                // 같은 색 King이 한 번이라도 움직였는지
                if (log[j].type == pieceType::KING && log[j].color == p.C) {
                    kingMoved = true;
                }
                // 같은 색 퀸사이드 룩(=a열, checkRank)에 있었다가 움직였는지
                if (log[j].type == pieceType::ROOK
                    && log[j].color == p.C
                    && log[j].fromRank == checkRank 
                    && log[j].fromFile == 0)
                {
                    rookA_moved = true;
                }
                // 같은 색 킹사이드 룩(=h열, checkRank)에 있었다가 움직였는지
                if (log[j].type == pieceType::ROOK
                    && log[j].color == p.C
                    && log[j].fromRank == checkRank
                    && log[j].fromFile == 7)
                {
                    rookH_moved = true;
                }
            }
        
            // (B2) helper 람다들: threatened / occupied / same-color 룩 존재 여부
            auto isThreatened = [&](int r, int f) {
                PGN* threatArr    = (p.C == pieceColor::WHITE) ? blackThreat : whiteThreat;
                int* threatSize   = (p.C == pieceColor::WHITE) ? blackThreatSize : whiteThreatSize;
                for (int i = 0; i < *threatSize; ++i) {
                    if (threatArr[i].toRank == r && threatArr[i].toFile == f) {
                        return true;
                    }
                }
                return false;
            };
        
            auto isOccupied = [&](int r, int f) {
                if (r < 0 || r > 7 || f < 0 || f > 7) return false;
                const piece sq = board_ptr[r * 8 + f];
                return (sq.T != pieceType::NONE);
            };
        
            auto isThereRook = [&](int r, int f) {
                if (r < 0 || r > 7 || f < 0 || f > 7) return false;
                const piece sq = board_ptr[r * 8 + f];
                // 같은 색의 룩이어야 한다
                return (sq.T == pieceType::ROOK && sq.C == p.C);
            };
        
            // --- (C) 퀸사이드 캐슬링 조건 모두 점검
            bool canQueenSide = true;
            // (C1) 왕이나 퀸사이드 룩이 이미 움직였었는지
            if (kingMoved || rookA_moved) {
                canQueenSide = false;
            }
            // (C2) a열(0)에 같은 색 룩이 반드시 있어야 함
            if (!isThereRook(checkRank, 0)) {
                canQueenSide = false;
            }
            // (C3) b열(1), c열(2)는 빈칸이어야 함
            if (isOccupied(checkRank, 1) || isOccupied(checkRank, 2)) {
                canQueenSide = false;
            }
            // (C4) a열(0), c열(2), d열(3)은 공격받지 않아야 함
            if ( isThreatened(checkRank, 0)
              || isThreatened(checkRank, 2)
              || isThreatened(checkRank, 3) )
            {
                canQueenSide = false;
            }
            // (C5) 조건 통과 시 PGN에 추가
            if (canQueenSide) {
                threadLocalBuf[threadIdx.x][localCnt++] 
                    = PGN{ p.C, p.T, f0, r0, f0 - 2, r0,
                           pieceType::NONE, specialMove::QUEEN_SIDE_CASTLING };
            }
        
            // --- (D) 킹사이드 캐슬링 조건 모두 점검
            bool canKingSide = true;
            // (D1) 왕이나 킹사이드 룩이 이미 움직였었는지
            if (kingMoved || rookH_moved) {
                canKingSide = false;
            }
            // (D2) h열(7)에 같은 색 룩이 반드시 있어야 함
            if (!isThereRook(checkRank, 7)) {
                canKingSide = false;
            }
            // (D3) f열(5), g열(6)는 빈칸이어야 함
            if (isOccupied(checkRank, 5) || isOccupied(checkRank, 6)) {
                canKingSide = false;
            }
            // (D4) h열(7), f열(5), g열(6)은 공격받지 않아야 함
            if ( isThreatened(checkRank, 7)
              || isThreatened(checkRank, 5)
              || isThreatened(checkRank, 6) )
            {
                canKingSide = false;
            }
            // (D5) 조건 통과 시 PGN에 추가
            if (canKingSide) {
                threadLocalBuf[threadIdx.x][localCnt++] 
                    = PGN{ p.C, p.T, f0, r0, f0 + 2, r0,
                           pieceType::NONE, specialMove::KING_SIDE_CASTLING };
            }
        
            break;
        }        
        case pieceType::QUEEN:
            // 룩 슬라이딩
            for (int dir : { -1, +1 }) {
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0 + dir * i;
                    int f1 = f0;
                    if (r1 < 0 || r1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0;
                    int f1 = f0 + dir * i;
                    if (f1 < 0 || f1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
            }
            // 비숍 슬라이딩
            for (int dr : { -1, +1 }) for (int df : { -1, +1 }) {
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0 + dr * i;
                    int f1 = f0 + df * i;
                    if (r1 < 0 || r1 > 7 || f1 < 0 || f1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
            }
            break;

        case pieceType::ROOK:
            // 룩 슬라이딩
            for (int dir : { -1, +1 }) {
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0 + dir * i;
                    int f1 = f0;
                    if (r1 < 0 || r1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0;
                    int f1 = f0 + dir * i;
                    if (f1 < 0 || f1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
            }
            break;

        case pieceType::BISHOP:
            // 비숍 슬라이딩
            for (int dr : { -1, +1 }) for (int df : { -1, +1 }) {
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0 + dr * i;
                    int f1 = f0 + df * i;
                    if (r1 < 0 || r1 > 7 || f1 < 0 || f1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
            }
            break;

        case pieceType::KNIGHT:{
            // 맨해튼 반지름 3
            for (int dr = -2; dr <= 2; ++dr) {
                for (int df = -2; df <= 2; ++df) {
                    if (dr*dr + df*df != 5) continue;
                    int r1 = r0 + dr;
                    int f1 = f0 + df;
                    if (r1 < 0 || r1 > 7 || f1 < 0 || f1 > 7) continue;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) continue;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                }
            }
            break;
        }
        case pieceType::PAWN:{
            // 전진 & 캡쳐
                int forward = (p.C == pieceColor::WHITE) ? -1 : +1;
                int nr = r0 + forward;
                int nf = f0;
                bool home = (p.C == pieceColor::WHITE && r0 == 6) || (p.C == pieceColor::BLACK && r0 == 1);
                if (nr >= 0 && nr <= 7) {
                    const piece t = board_ptr[nr * 8 + nf];
                    if (t.T == pieceType::NONE) {
                        // 프로모션
                        if ((p.C == pieceColor::WHITE && nr == 0) || (p.C == pieceColor::BLACK && nr == 7)) {
                            threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, nf, nr, pieceType::QUEEN, specialMove::PROMOTION};
                            threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, nf, nr, pieceType::ROOK, specialMove::PROMOTION};
                            threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, nf, nr, pieceType::BISHOP, specialMove::PROMOTION};
                            threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, nf, nr, pieceType::KNIGHT, specialMove::PROMOTION};
                        } else {
                            threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, nf, nr, pieceType::NONE};
                        }
                        int nr2 = r0 + 2 * forward;
                        if (home && nr2 >= 0 && nr2 <=7) {
                            const piece t2 = board_ptr[nr2 * 8 + nf];
                            if (t2.T == pieceType::NONE) {
                                threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, nf, nr2, pieceType::NONE};
                            }
                        }
                    }
                }
                for (int df : { -1, +1 }) {
                    int r1 = r0 + forward;
                    int f1 = f0 + df;
                    if (r1 < 0 || r1 > 7 || f1 < 0 || f1 > 7) continue;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C != p.C && t.C != pieceColor::NONE) {
                        // 프로모션
                        if ((p.C == pieceColor::WHITE && r1 == 0) || (p.C == pieceColor::BLACK && r1 == 7)) {
                            threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::QUEEN, specialMove::PROMOTION};
                            threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::ROOK, specialMove::PROMOTION};
                            threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::BISHOP, specialMove::PROMOTION};
                            threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::KNIGHT, specialMove::PROMOTION};
                        } else {
                            threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                        }
                    }
                }

                //앙파상 코드
                int ep = p.C == pieceColor::WHITE ? 3 : 4;
                if(r0 == ep && last_Log.type == pieceType::PAWN && last_Log.color != p.C && abs(last_Log.toRank - last_Log.fromRank) == 2 && f0 - last_Log.toFile == -1){
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f0+1, r0+forward, pieceType::NONE, specialMove::ENPASSANT};
                }else if(r0 == ep && last_Log.type == pieceType::PAWN && last_Log.color != p.C && abs(last_Log.toRank - last_Log.fromRank) == 2 && f0 - last_Log.toFile == 1){
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f0-1, r0+forward, pieceType::NONE, specialMove::ENPASSANT};
                }
            break;
        }
        default:
            break;
    }

    

    if (localCnt > 0) {
        int start = atomicAdd(moveCount, localCnt);
        for (int i = 0; i < localCnt; ++i) {
            moveBuf[start + i] = threadLocalBuf[threadIdx.x][i];
        }
    }
}

__global__ void calculateTheatsSq(
    piece*                    board_ptr,
    PGN*         __restrict__ threatBuf,
    int*         __restrict__ threatCount,
    pieceColor color) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 64) return;

    __shared__ PGN threadLocalBuf[32][MAX_MOVES_PER_PIECE];

    const piece p = board_ptr[idx];
    if (p.T == pieceType::NONE || p.C == pieceColor::NONE || p.C != color) {
        return;
    }

    int r0 = p.rank;
    int f0 = p.file;
    int localCnt = 0;

    switch (p.T) {
        case pieceType::KING:
            for (int dr = -1; dr <= 1; ++dr) {
                for (int df = -1; df <= 1; ++df) {
                    if (dr == 0 && df == 0) continue;
                    int r1 = r0 + dr;
                    int f1 = f0 + df;
                    if (r1 < 0 || r1 > 7 || f1 < 0 || f1 > 7) continue;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) continue;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                }
            }
            break;

        case pieceType::QUEEN:
            // 룩 슬라이딩
            for (int dir : { -1, +1 }) {
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0 + dir * i;
                    int f1 = f0;
                    if (r1 < 0 || r1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0;
                    int f1 = f0 + dir * i;
                    if (f1 < 0 || f1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
            }
            // 비숍 슬라이딩
            for (int dr : { -1, +1 }) for (int df : { -1, +1 }) {
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0 + dr * i;
                    int f1 = f0 + df * i;
                    if (r1 < 0 || r1 > 7 || f1 < 0 || f1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
            }
            break;

        case pieceType::ROOK:
            // 룩 슬라이딩
            for (int dir : { -1, +1 }) {
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0 + dir * i;
                    int f1 = f0;
                    if (r1 < 0 || r1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0;
                    int f1 = f0 + dir * i;
                    if (f1 < 0 || f1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
            }
            break;

        case pieceType::BISHOP:
            // 비숍 슬라이딩
            for (int dr : { -1, +1 }) for (int df : { -1, +1 }) {
                for (int i = 1; i < 8; ++i) {
                    int r1 = r0 + dr * i;
                    int f1 = f0 + df * i;
                    if (r1 < 0 || r1 > 7 || f1 < 0 || f1 > 7) break;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) break;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                    if (t.T != pieceType::NONE) break;
                }
            }
            break;

        case pieceType::KNIGHT:{
            // 맨해튼 반지름 3
            for (int dr = -2; dr <= 2; ++dr) {
                for (int df = -2; df <= 2; ++df) {
                    if (dr*dr + df*df != 5) continue;
                    int r1 = r0 + dr;
                    int f1 = f0 + df;
                    if (r1 < 0 || r1 > 7 || f1 < 0 || f1 > 7) continue;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if (t.C == p.C) continue;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                }
            }
            break;
        }
        case pieceType::PAWN:{
            // 캡쳐
                int forward = (p.C == pieceColor::WHITE) ? -1 : +1;
                for (int df : { -1, +1 }) {
                    int r1 = r0 + forward;
                    int f1 = f0 + df;
                    if (r1 < 0 || r1 > 7 || f1 < 0 || f1 > 7) continue;
                    const piece t = board_ptr[r1 * 8 + f1];
                    if(t.C == p.C) continue;
                    threadLocalBuf[threadIdx.x][localCnt++] = PGN{p.C, p.T, f0, r0, f1, r1, pieceType::NONE};
                }
            break;
        }
        default:
            break;
    }

    if (localCnt > 0) {
        int start = atomicAdd(threatCount, localCnt);
        for (int i = 0; i < localCnt; ++i) {
            threatBuf[start + i] = threadLocalBuf[threadIdx.x][i];
        }
    }
}

void chessboard::calculateThreatSquare(pieceColor color)
{
    PGN* d_threatBuf;
    int* d_threatCnt;
    int maxTotal = 32 * MAX_MOVES_PER_PIECE;
    cudaMalloc(&d_threatBuf, sizeof(PGN)*maxTotal);
    cudaMalloc(&d_threatCnt, sizeof(int));
    cudaMemset(d_threatCnt, 0, sizeof(int));

    int threads = 32;
    int blocks  = 2;
    calculateTheatsSq<<<blocks, threads>>>(board.getDevPointer(), d_threatBuf, d_threatCnt, color);
    cudaDeviceSynchronize();

    int h_cnt;
    cudaMemcpy(&h_cnt, d_threatCnt, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<PGN>* targetThreat = (color == pieceColor::WHITE) ? &whiteThreatsq : &blackThreatsq;
    targetThreat->resize(h_cnt);
    cudaMemcpy(targetThreat->data(), d_threatBuf, sizeof(PGN)*h_cnt, cudaMemcpyDeviceToHost);

    // Host에서 board의 sT 갱신
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) board(i, j).sT = squareType::NONE;
    for (const auto& pgn : *targetThreat) {
        int r = pgn.toRank;
        int f = pgn.toFile;
        if (r < 0 || r > 7 || f < 0 || f > 7) continue;
        squareType& st = board(r, f).sT;
        if (st == squareType::NONE) {
            st = (color == pieceColor::WHITE ? squareType::WHITE_THREAT : squareType::BLACK_THREAT);
        } else if (st != (color == pieceColor::WHITE ? squareType::WHITE_THREAT : squareType::BLACK_THREAT)) {
            st = squareType::BOTH_THREAT;
        }
    }

    board.cpyToDev();
    cudaFree(d_threatBuf);
    cudaFree(d_threatCnt);
}

// Host wrapper: chessboard::genLegalMoves()
void chessboard::genLegalMoves() {

    PGN* d_moveBuf;
    int* d_moveCount;
    PGN* d_log;
    int* d_logsize;
    int  h_logData = static_cast<int>(log.size());
    PGN* d_wThreat;
    int* d_wThreatSize;
    int  h_wThreatSize = static_cast<int>(whiteThreatsq.size());
    PGN* d_bThreat;
    int* d_bThreatSize;
    int  h_bThreatSize = static_cast<int>(blackThreatsq.size());
    int maxTotal = 32 * MAX_MOVES_PER_PIECE;

    calculateThreatSquare(pieceColor::WHITE);
    calculateThreatSquare(pieceColor::BLACK);
    board.cpyToDev();

    cudaMalloc(&d_moveBuf, sizeof(PGN) * maxTotal);
    cudaMalloc(&d_moveCount, sizeof(int));
    cudaMemset(d_moveCount, 0, sizeof(int));

    cudaMalloc(&d_log, sizeof(PGN)*log.size());
    cudaMemcpy(d_log, log.data(), sizeof(PGN)*log.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_logsize, sizeof(int));
    cudaMemcpy(d_logsize, &h_logData, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_wThreat, sizeof(PGN)*whiteThreatsq.size());
    cudaMemcpy(d_wThreat, whiteThreatsq.data(), sizeof(PGN)*whiteThreatsq.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_wThreatSize, sizeof(int));
    cudaMemcpy(d_wThreatSize, &h_wThreatSize, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_bThreat, sizeof(PGN)*blackThreatsq.size());
    cudaMemcpy(d_bThreat, blackThreatsq.data(), sizeof(PGN)*blackThreatsq.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_bThreatSize, sizeof(int));
    cudaMemcpy(d_bThreatSize, &h_bThreatSize, sizeof(int), cudaMemcpyHostToDevice);

    int threads = 32;
    int blocks  = 2;
    genLegalMovesKernel<<<blocks, threads>>>(board.getDevPointer(), d_moveBuf, d_moveCount, d_log, d_logsize, d_bThreat, d_bThreatSize, d_wThreat, d_wThreatSize);
    cudaDeviceSynchronize();

    int h_count;
    cudaMemcpy(&h_count, d_moveCount, sizeof(int), cudaMemcpyDeviceToHost);
    moveList.resize(h_count);
    cudaMemcpy(moveList.data(), d_moveBuf, sizeof(PGN) * h_count, cudaMemcpyDeviceToHost);

    testP save_position;

    save_position.testboard = board;  // 보드 복원용 백업
    save_position.log = log;

    std::vector<PGN> legal_moves;
    bool whiteInCheck = isWhiteChacked();
    bool blackInCheck = isBlackChacked();

    // ── White 차례이고, 현재 White가 체크 중이라면 ──
    if (whiteInCheck) {
        for (auto mv : moveList) {
            if (mv.color != pieceColor::WHITE) continue;

            // (1) 한 수 두기
            updateBoard(mv);

            // (2) 둔 뒤의 위협 정보를 다시 계산하고
            calculateThreatSquare(pieceColor::WHITE);
            calculateThreatSquare(pieceColor::BLACK);

            // (3) 이제 진짜로 체크가 풀렸는지 확인
            if (!isWhiteChacked()) {
                legal_moves.push_back(mv);
            }

            // (4) 반드시 원본 보드로 복원
            loadPosition(save_position);
        }
    }
    // ── Black 차례이고, 현재 Black이 체크 중이라면 ──
    else if (blackInCheck) {
        for (auto mv : moveList) {
            if (mv.color != pieceColor::BLACK) continue;

            updateBoard(mv);
            calculateThreatSquare(pieceColor::WHITE);
            calculateThreatSquare(pieceColor::BLACK);

            if (!isBlackChacked()) {
                legal_moves.push_back(mv);
            }
            loadPosition(save_position);
        }
    }
    // ── 체크 상태가 아니라면, moveList 자체가 모두 합법 ──
    else {
        legal_moves = moveList;
    }

    calculateThreatSquare(pieceColor::WHITE);
    calculateThreatSquare(pieceColor::BLACK);

    moveList = std::move(legal_moves);

    cudaFree(d_moveBuf);
    cudaFree(d_moveCount);
    cudaFree(d_log);
    cudaFree(d_logsize);
    cudaFree(d_bThreat);
    cudaFree(d_bThreatSize);
    cudaFree(d_wThreat);
    cudaFree(d_wThreatSize);
}



void chessboard::updateBoard(const PGN &mv)
{
    bool KingSideRight = false;
    std::for_each(moveList.begin(), moveList.end(), [&KingSideRight](PGN i) { if(i.stype == specialMove::KING_SIDE_CASTLING){ KingSideRight = true; } });
    bool QueenSideRight = false;
    std::for_each(moveList.begin(), moveList.end(), [&QueenSideRight](PGN i) { if(i.stype == specialMove::QUEEN_SIDE_CASTLING){ QueenSideRight = true; } });

    if(std::find(moveList.begin(), moveList.end(), mv) == moveList.end()) return;

    if(mv.stype == specialMove::KING_SIDE_CASTLING && KingSideRight){
        int rank = (mv.color == pieceColor::WHITE) ? 7 : 0;
        // 킹 이동
        board(rank, 6) = board(rank, 4);
        board(rank, 4) = piece();
        board(rank, 6).file = 6;
        // 룩 이동
        board(rank, 5) = board(rank, 7);
        board(rank, 7) = piece();
        board(rank, 5).file = 5;
    } else if (mv.stype == specialMove::QUEEN_SIDE_CASTLING && QueenSideRight) {
        int rank = (mv.color == pieceColor::WHITE) ? 7 : 0;
        // 킹 이동
        board(rank, 2) = board(rank, 4);
        board(rank, 4) = piece();
        board(rank, 2).file = 2;
        // 룩 이동
        board(rank, 3) = board(rank, 0);
        board(rank, 0) = piece();
        board(rank, 3).file = 3;
    } else if (mv.stype == specialMove::ENPASSANT) {
        // 앙파상
        int capRank = (mv.color == pieceColor::WHITE) ? mv.toRank + 1 : mv.toRank - 1;
        board(mv.toRank, mv.toFile) = board(mv.fromRank, mv.fromFile);
        board(mv.fromRank, mv.fromFile) = piece();
        board(capRank, mv.toFile) = piece(); // 잡힌 폰 제거
        board(mv.toRank, mv.toFile).rank = mv.toRank;
        board(mv.toRank, mv.toFile).file = mv.toFile;
    } else {
        // 일반 이동 및 프로모션
        board(mv.toRank, mv.toFile) = board(mv.fromRank, mv.fromFile);
        board(mv.fromRank, mv.fromFile) = piece();
        board(mv.toRank, mv.toFile).rank = mv.toRank;
        board(mv.toRank, mv.toFile).file = mv.toFile;
        if (mv.promotion != pieceType::NONE)
            board(mv.toRank, mv.toFile).T = mv.promotion;
    }

    log.push_back(mv);
    board.cpyToDev();
}

void chessboard::printBoard()
{
    for(int i = 0; i < 8; i++){
        for(int j = 0; j < 8; j++){
            switch (board(i, j).T)
            {
            case pieceType::KING:
                if(board(i, j).C == pieceColor::WHITE) std::cout << "K";
                else if(board(i, j).C == pieceColor::BLACK) std::cout << "k";
                break;
            case pieceType::QUEEN:
                if(board(i, j).C == pieceColor::WHITE) std::cout << "Q";
                else if(board(i, j).C == pieceColor::BLACK) std::cout << "q";
                break;
            case pieceType::BISHOP:
                if(board(i, j).C == pieceColor::WHITE) std::cout << "B";
                else if(board(i, j).C == pieceColor::BLACK) std::cout << "b";
                break;
            case pieceType::KNIGHT:
                if(board(i, j).C == pieceColor::WHITE) std::cout << "N";
                else if(board(i, j).C == pieceColor::BLACK) std::cout << "n";
                break;
            case pieceType::ROOK:
                if(board(i, j).C == pieceColor::WHITE) std::cout << "R";
                else if(board(i, j).C == pieceColor::BLACK) std::cout << "r";
                break;
            case pieceType::PAWN:
                if(board(i, j).C == pieceColor::WHITE) std::cout << "P";
                else if(board(i, j).C == pieceColor::BLACK) std::cout << "p";
                break;
            case pieceType::NONE:
                std::cout << ".";
                break;
            default:
                break;
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }
}


std::string translateMove(const PGN& mv){
    std::string result;

    if (mv.toFile < 0 || mv.toFile > 7 || mv.toRank < 0 || mv.toRank > 7){
        std::cerr << "Error!" << std::endl;
        std::cerr << "  bad move: toFile=" << mv.toFile << ", toRank=" << mv.toRank << std::endl;
        return "0";
    }

    if(mv.color == pieceColor::WHITE){
        result += 'w';
    }else if(mv.color == pieceColor::BLACK){
        result += 'b';
    }

    if(mv.stype == specialMove::KING_SIDE_CASTLING){
        result += "O-O";
        return result;
    }else if(mv.stype == specialMove::QUEEN_SIDE_CASTLING){
        result += "O-O-O";
        return result;
    }

    if(mv.type == pieceType::KING){
        result += 'K';
    } else if(mv.type == pieceType::QUEEN){
        result += 'Q';
    } else if(mv.type == pieceType::BISHOP){
        result += 'B';
    } else if(mv.type == pieceType::KNIGHT){
        result += 'N';
    } else if(mv.type == pieceType::ROOK){
        result += 'R';
    }

    result += (char)('a' + mv.fromFile);
    result += (char)('8' - mv.fromRank);
    result += (char)('a' + mv.toFile);
    result += (char)('8' - mv.toRank);

    if(mv.promotion == pieceType::QUEEN){
        result += 'Q';
    } else if(mv.promotion == pieceType::BISHOP){
        result += 'B';
    } else if(mv.promotion == pieceType::KNIGHT){
        result += 'N';
    } else if(mv.promotion == pieceType::ROOK){
        result += 'R';
    }

    return result;
}


PGN translatePGN(const std::string& mv) {
    PGN result;
    // (1) 기본 필드 초기화
    result.color     = pieceColor::NONE;
    result.type      = pieceType::NONE;
    result.fromFile  = -1;
    result.fromRank  = -1;
    result.toFile    = -1;
    result.toRank    = -1;
    result.promotion = pieceType::NONE;
    result.stype     = specialMove::NONE;

    int idx = 0;
    int N = static_cast<int>(mv.size());

    // (2) 첫 글자: 색상 ("w" 또는 "b")
    if (idx < N) {
        if (mv[idx] == 'w' || mv[idx] == 'W') {
            result.color = pieceColor::WHITE;
        } else if (mv[idx] == 'b' || mv[idx] == 'B') {
            result.color = pieceColor::BLACK;
        } else {
            // 유효하지 않은 색상 문자
            // 필요에 따라 예외 처리할 수 있습니다.
            // throw std::invalid_argument("Invalid color in PGN string");
            result.color = pieceColor::NONE;
        }
    }
    idx++;

    // (3) 캐슬링 표기 검출: "O-O" 또는 "O-O-O"
    //    mv 예: "wO-O", "bO-O-O"
    if (idx + 2 < N && mv[idx] == 'O' && mv[idx+1] == '-' && mv[idx+2] == 'O') {
        // "O-O"로 시작하는 경우 (킹사이드 또는 퀸사이드)
        // 킹사이드: 정확히 "O-O"
        // 퀸사이드: "O-O-O"
        bool isQueenSide = false;
        if (idx + 4 < N && mv[idx+3] == '-' && mv[idx+4] == 'O') {
            // "O-O-O"인 경우
            isQueenSide = true;
        }

        // (3-1) 기물은 KING으로 고정
        result.type = pieceType::KING;

        // (3-2) 출발지와 도착지는 색상에 따라 미리 정해진 위치
        //     WHITE: e1 (4,7) → O-O → g1 (6,7), O-O-O → c1 (2,7)
        //     BLACK: e8 (4,0) → O-O → g8 (6,0), O-O-O → c8 (2,0)
        int baseRank = (result.color == pieceColor::WHITE ? 7 : 0);
        int fromFile = 4;
        int fromRank = baseRank;
        int toFile, toRank = baseRank;

        if (isQueenSide) {
            // 퀸사이드 캐슬링
            toFile        = 2;
            result.stype  = specialMove::QUEEN_SIDE_CASTLING;
            // 전체 길이는 "O-O-O" → 5글자
            idx += 5;
        } else {
            // 킹사이드 캐슬링
            toFile        = 6;
            result.stype  = specialMove::KING_SIDE_CASTLING;
            // 전체 길이는 "O-O" → 3글자
            idx += 3;
        }

        result.fromFile = fromFile;
        result.fromRank = fromRank;
        result.toFile   = toFile;
        result.toRank   = toRank;
        // 캐슬링이므로 promotion 없음

        return result;
    }

    // (4) 캐슬링이 아니면, 이어서 “기물문자 여부” 검사
    //     mv[idx]가 대문자면 KING, QUEEN, BISHOP, KNIGHT, ROOK 중 하나
    if (idx < N && std::isupper(static_cast<unsigned char>(mv[idx]))) {
        switch (mv[idx]) {
            case 'K': result.type = pieceType::KING;   break;
            case 'Q': result.type = pieceType::QUEEN;  break;
            case 'B': result.type = pieceType::BISHOP; break;
            case 'N': result.type = pieceType::KNIGHT; break;
            case 'R': result.type = pieceType::ROOK;   break;
            default:
                // 알 수 없는 대문자
                result.type = pieceType::NONE;
                break;
        }
        idx++;
    } else {
        // mv[idx]가 대문자가 아니면 pawn(PAWN) 이동
        result.type = pieceType::PAWN;
    }

    // (5) 출발칸(fromFile, fromRank)과 도착칸(toFile, toRank) 파싱
    //     남은 문자열에서 최소 4글자 필요: ex) "e2e4"
    if (idx + 3 < N) {
        char fChar  = mv[idx + 0]; // 예: 'e'
        char rChar  = mv[idx + 1]; // 예: '2'
        char fChar2 = mv[idx + 2]; // 예: 'e'
        char rChar2 = mv[idx + 3]; // 예: '4'

        // 파일 -> 0~7
        if (fChar >= 'a' && fChar <= 'h') {
            result.fromFile = fChar - 'a';
        }
        // 랭크 -> 0~7 ( '1'->7, '8'->0 )
        if (rChar >= '1' && rChar <= '8') {
            result.fromRank = '8' - rChar;
        }
        if (fChar2 >= 'a' && fChar2 <= 'h') {
            result.toFile = fChar2 - 'a';
        }
        if (rChar2 >= '1' && rChar2 <= '8') {
            result.toRank = '8' - rChar2;
        }
        idx += 4;
    }

    // (6) 남은 문자가 있으면 프로모션(Promotion) 처리
    //     ex) "a7a8Q" 처럼 끝에 'Q','R','B','N'이 붙을 수 있음
    if (idx < N) {
        char promo = mv[idx];
        switch (promo) {
            case 'Q':
                result.promotion = pieceType::QUEEN;
                result.stype     = specialMove::PROMOTION;
                break;
            case 'R':
                result.promotion = pieceType::ROOK;
                result.stype     = specialMove::PROMOTION;
                break;
            case 'B':
                result.promotion = pieceType::BISHOP;
                result.stype     = specialMove::PROMOTION;
                break;
            case 'N':
                result.promotion = pieceType::KNIGHT;
                result.stype     = specialMove::PROMOTION;
                break;
            default:
                // 그 외 문자는 무시
                break;
        }
        // idx++;
    }

    return result;
}


void debugChessboardvar(chessboard* board)
{
    board->genLegalMoves();

    auto moves = board->getMoveList();
    auto whiteThreat = board->getWhiteThreatSq();
    auto blackThreat = board->getBlackThreatSq();

    std::cout << "가능한 움직임:";
    for (int i = 0; i < moves.size(); i++)
    {
        std::cout << translateMove(moves[i]) << "," << " ";
    }
    std::cout << std::endl;
    std::cout << "백의 공격지점:";
    for (int i = 0; i < whiteThreat.size(); i++)
    {
        std::cout << translateMove(whiteThreat[i]) << "," << " ";
    }
    std::cout << std::endl;
    std::cout << "흑의 공격지점:";
    for (int i = 0; i < blackThreat.size(); i++)
    {
        std::cout << translateMove(blackThreat[i]) << "," << " ";
    }
    std::cout << std::endl;
}


