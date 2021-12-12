#pragma once

#include "field.h"
#include <random>

struct AlphaBetaResult
{
	int score;
	Move move;
	AlphaBetaResult(int score, Move move = 0) : score(score), move(move) {}
};

MoveList GeneratePossibleMoves2(const Field& field, MoveList& moves);
AlphaBetaResult AlphaBeta(Field& field, const MoveList& moves, int depth, int alpha = -INT_MAX, int beta = INT_MAX);
Move AlphaBeta(Field& field, std::mt19937 &gen, int depth);