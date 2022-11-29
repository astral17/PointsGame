#include "alpha_beta.h"
#include <omp.h>
#include <algorithm>

// TODO: Удалить дублирование кода
MoveList GeneratePossibleMoves2(const Field& field, MoveList& moves)
{
	// TODO BUG: пустая далёкая зона в центре является выходом, что не есть правда
	MoveList tempBorder;
	unsigned char* r_field = new unsigned char[field.Length()];
	std::fill_n(r_field, field.Length(), 0);
	std::queue<Move> q;

	moves.clear();
	for (Move i = field.MinMove(); i <= field.MaxMove(); i++)
		if (field.IsPoint(i))
			q.push(i);

	while (!q.empty())
	{
		if (field.CouldMove(q.front()))
			moves.push_back(q.front());
		if (r_field[q.front()] < 2)
		{
			if (field.CouldMove(field.Up(q.front())) && r_field[field.Up(q.front())] == 0)
			{
				r_field[field.Up(q.front())] = r_field[q.front()] + 1;
				q.push(field.Up(q.front()));
			}
			if (field.CouldMove(field.Down(q.front())) && r_field[field.Down(q.front())] == 0)
			{
				r_field[field.Down(q.front())] = r_field[q.front()] + 1;
				q.push(field.Down(q.front()));
			}
			if (field.CouldMove(field.Left(q.front())) && r_field[field.Left(q.front())] == 0)
			{
				r_field[field.Left(q.front())] = r_field[q.front()] + 1;
				q.push(field.Left(q.front()));
			}
			if (field.CouldMove(field.Right(q.front())) && r_field[field.Right(q.front())] == 0)
			{
				r_field[field.Right(q.front())] = r_field[q.front()] + 1;
				q.push(field.Right(q.front()));
			}
		}
		else
		{
			if (!field.IsBorder(field.Up(q.front())) && r_field[field.Up(q.front())] == 0)
			{
				tempBorder.push_back(field.Up(q.front()));
			}
			if (!field.IsBorder(field.Down(q.front())) && r_field[field.Down(q.front())] == 0)
			{
				tempBorder.push_back(field.Down(q.front()));
			}
			if (!field.IsBorder(field.Left(q.front())) && r_field[field.Left(q.front())] == 0)
			{
				tempBorder.push_back(field.Left(q.front()));
			}
			if (!field.IsBorder(field.Right(q.front())) && r_field[field.Right(q.front())] == 0)
			{
				tempBorder.push_back(field.Right(q.front()));
			}
		}
		q.pop();
	}
	if (moves.empty())
		moves.push_back(field.ToMove(field.width / 2, field.height / 2));
	delete[] r_field;
	return tempBorder;
}

AlphaBetaResult AlphaBeta(Field& field, const MoveList& moves, int depth, int alpha, int beta)
{
	if (depth == 0)
		return AlphaBetaResult(field.GetScore(field.GetPlayer()));
	AlphaBetaResult bestMove(-INT_MAX);
	bool moved = false;
	for (Move move : moves)
	{
		if (field.CouldMove(move))
		{
			moved = true;
			field.MakeMove(move);
			AlphaBetaResult curMove(-AlphaBeta(field, moves, depth - 1, -beta, -alpha).score, move);
			field.Undo();
			if (bestMove.score < curMove.score)
				bestMove = curMove;
			if (beta <= curMove.score)
				break;
			alpha = std::max(alpha, curMove.score);
		}
	}
	if (!moved)
		return AlphaBetaResult(field.GetScore(field.GetPlayer()));
	return bestMove;
}

Move AlphaBeta(Field& field, std::mt19937 &gen, int depth)
{
	//MoveList moves = field.GetAllMoves();
	MoveList moves;
	MoveList tempBorder = GeneratePossibleMoves2(field, moves);
	shuffle(moves.begin(), moves.end(), gen);
	AlphaBetaResult result(-INT_MAX);
	omp_set_num_threads(std::min((size_t)omp_get_max_threads(), moves.size()));
	//omp_set_num_threads(1);
	int alpha = -INT_MAX, beta = INT_MAX;
#pragma omp parallel
	{
		Field local_field(field);
		MoveList local_moves;
		for (auto i = moves.begin() + omp_get_thread_num(); i < moves.end(); i += omp_get_num_threads())
			if (alpha < beta)
			{
				if (local_field.CouldMove(*i))
				{
					local_field.MakeMove(*i);
					AlphaBetaResult curMove(-AlphaBeta(local_field, moves, depth - 1, -beta, -alpha).score, *i);
					local_field.Undo();
#pragma omp critical
					{
						if (result.score < curMove.score)
							result = curMove;
						alpha = std::max(alpha, curMove.score);
					}
				}
			}
		//AlphaBetaResult local_result = AlphaBeta(local_field, local_moves, depth);
	}
	for (Move move : tempBorder)
		field.DelBorder(move);
	//std::cout << "alpha beta score: " << result.score << "\n";
	if (!field.CouldMove(result.move))
		exit(123);
	return result.move;
	//return AlphaBeta(field, moves, depth).move;
}