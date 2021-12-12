#include "bns.h"
#include "alpha_beta.h"
#include <algorithm>
#include <iostream>

int NextGuess(int alpha, int beta, int subtreeCount)
{
	return alpha + (beta - alpha) *(subtreeCount - 1) / subtreeCount;
}
#define FMT(x) #x << " = " << x
Move BestNodeSearch(Field& field, std::mt19937& gen, int depth)
{
	//MoveList moves = field.GetAllMoves();
	MoveList moves;
	MoveList tempBorder = GeneratePossibleMoves2(field, moves);
	shuffle(moves.begin(), moves.end(), gen);
	Move bestMove = -1;
	int subtreeCount = moves.size(), betterCount;
	int alpha = -100, beta = 100;
	do
	{
		int test = NextGuess(alpha, beta, subtreeCount);
		betterCount = 0;
		int maxValue = -INT_MAX;
		std::cout << FMT(alpha) << ", " << FMT(beta) << ", " << FMT(subtreeCount) << ", " << FMT(test) << ":\t";
		// TODO: Parallel
		for (Move move : moves)
		{
			field.MakeMove(move);
			// TODO: AlphaBetaWithMemory
			int bestVal = -AlphaBeta(field, moves, depth - 1, -test, -(test - 1)).score;
			//maxValue = std::max(maxValue, bestVal);
			if (bestVal > maxValue)
			{
				maxValue = bestVal;
				bestMove = move;
			}
			if (bestVal >= test)
			{
				betterCount++;
				//bestMove = move;
			}
			field.Undo();
		}
		std::cout << FMT(betterCount) << ", " << FMT(maxValue) << "\n";
		if (betterCount == 0)
		{
			//beta = test - 1;
			beta = maxValue;
		}
		else
		{
			//alpha = test;
			alpha = maxValue;
			subtreeCount = betterCount;
		}
	} while (beta - alpha > 1 && betterCount != 1);
	for (Move move : tempBorder)
		field.DelBorder(move);
	std::cout << "\n";
	std::cout << "BestNodeSearch " << alpha << " " << beta << "\n";
	return bestMove;
}