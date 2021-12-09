#pragma once

#include "field.h"
#include <random>

struct AlphaBetaResult
{
	int score;
	Move move;
	AlphaBetaResult(int score, Move move = 0) : score(score), move(move) {}
};

Move AlphaBeta(Field& field, std::mt19937 &gen, int depth);