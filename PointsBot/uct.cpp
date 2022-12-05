#include "uct.h"
//#include "field.h"
#include <limits>
#include <queue>
#include <vector>
#include <algorithm>
#include <omp.h>

#include <fstream>

constexpr int UCTK = 1;
constexpr int UCT_RADIUS = 2;

#define ORDER_BY_SCORE 1

using namespace std;

short PlayRandomGame(Field& field, mt19937& gen, MoveList moves)
{
	size_t putted = 0;
	short result;
	shuffle(moves.begin(), moves.end(), gen);
	for (auto i = moves.begin(); i < moves.end(); i++)
		if (field.CouldMove(*i))
		{
			field.MakeMove(*i);
			putted++;
		}
#ifdef ORDER_BY_SCORE
	result = field.GetScore(kPlayerRed);
#else
	if (field.GetScore(kPlayerRed) > 0)
		result = kPlayerRed;
	else if (field.GetScore(kPlayerBlack) > 0)
		result = kPlayerBlack;
	else
		result = -1;
#endif

	for (size_t i = 0; i < putted; i++)
		field.Undo();

	return result;
}

bool CreateChildren(Field& field, const MoveList& moves, UctNode* n)
{
	UctNode** cur_child = &n->child;

	for (Move move : moves)
		if (field.CouldMove(move))
		{
			*cur_child = new UctNode(n);
			(*cur_child)->move = move;
			cur_child = &(*cur_child)->sibling;
		}
	return n->child;
}

UctNode* UctSelect(mt19937& gen, UctNode* n)
{
	double bestUct = numeric_limits<double>::lowest(), winRate, uct, uctValue;
	UctNode* result = nullptr;
	for (UctNode* next = n->child; next; next = next->sibling)
	{
		if (next->visits > 0)
		{
#ifdef ORDER_BY_SCORE
			winRate = static_cast<double>(next->wins) / next->visits;
			uct = UCTK * sqrt(log(static_cast<double>(n->visits)) / next->visits);
#else
			winRate = static_cast<double>(next->wins) / (4 * next->visits);
			uct = UCTK * sqrt(log(static_cast<double>(n->visits)) / (5 * next->visits));
#endif
			uctValue = winRate + uct;
		}
		else
		{
			uctValue = (gen() | (1u << 31));
		}

		if (uctValue > bestUct)
		{
			bestUct = uctValue;
			result = next;
		}
	}

	return result;
}

short PlaySimulation(Field& field, mt19937& gen, const MoveList& possible_moves, UctNode* cur)
{
	// TODO: Избавиться от дублирования кода и сравнить рекурсивную и не рекурсивную реализации
	short result;
	int stkLen = 0;
	while (cur->visits)
	{
		if (!cur->child && !CreateChildren(field, possible_moves, cur))
		{
			// Детей нету и не будет - терминальное состояние
			cur->visits = numeric_limits<int>::max();
			if (field.GetScore(field.GetNextPlayer()) > 0)
				cur->wins = numeric_limits<int>::max();
			else if (field.GetScore(field.GetNextPlayer()) < 0)
				cur->wins = numeric_limits<int>::min();
#ifdef ORDER_BY_SCORE
			result = field.GetScore(kPlayerRed);
#else
			if (field.GetScore(kPlayerRed) > 0)
				result = kPlayerRed;
			else if (field.GetScore(kPlayerBlack) > 0)
				result = kPlayerBlack;
			else
				result = -1;
#endif
			break;
		}
		cur = UctSelect(gen, cur);
		field.MakeMove(cur->move);
	}
	if (!cur->visits)
	{
		result = PlayRandomGame(field, gen, possible_moves);
		cur->visits++;
#ifdef ORDER_BY_SCORE
		if (field.GetNextPlayer() == kPlayerRed)
			cur->wins += result;
		else
			cur->wins -= result;
#else
		if (result == field.GetNextPlayer())
			cur->wins += 4;
		else if (result == -1)
			cur->wins++;
#endif
	}

	while (cur->parent)
	{
		cur = cur->parent;
		field.Undo();
		cur->visits++;
#ifdef ORDER_BY_SCORE
		if (field.GetNextPlayer() == kPlayerRed)
			cur->wins += result;
		else
			cur->wins -= result;
#else
		if (result == field.GetNextPlayer())
			cur->wins += 4;
		else if (result == -1)
			cur->wins++;
#endif
	}

	return result;
}

template<typename _Cont>
MoveList GeneratePossibleMoves(Field& field, _Cont& moves)
{
	// TODO BUG: пустая далёкая зона в центре является выходом, что не есть правда
	MoveList tempBorder;
	unsigned char* r_field = new unsigned char[field.Length()];
	fill_n(r_field, field.Length(), 0);
	std::queue<Move> q;

	moves.clear();
	for (Move i = field.MinMove(); i <= field.MaxMove(); i++)
		if (field.IsPoint(i))
			q.push(i);

	while (!q.empty())
	{
		if (field.CouldMove(q.front()))
			moves.push_back(q.front());
		if (r_field[q.front()] < UCT_RADIUS)
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
	//field.astar.BuildDistance();
	return tempBorder;
}

Move Uct(Field& field, mt19937& gen, int simulations)
{
	// Список всех возможных ходов для UCT.
	MoveList moves;
	double best_estimate = numeric_limits<double>::lowest();
	Move result = -1;

	MoveList tempBorder = GeneratePossibleMoves(field, moves);
	for (Move move : tempBorder)
		field.SetBorder(move);

	omp_set_num_threads(min((size_t)omp_get_max_threads(), moves.size()));
	//omp_set_num_threads(1);


	//unsigned short* probs = new unsigned short[cur_field->length()];
	//fill_n(probs, cur_field->length(), 0);

#pragma omp parallel
	{
		UctNode n;
		Field local_field(field);
		unsigned int seed;
#pragma omp critical
		{
			seed = gen();
		}
		mt19937 local_gen(seed);

		UctNode** cur_child = &n.child;
		for (auto i = moves.begin() + omp_get_thread_num(); i < moves.end(); i += omp_get_num_threads())
		{
			*cur_child = new UctNode(&n);
			(*cur_child)->move = *i;
			cur_child = &(*cur_child)->sibling;
		}

#pragma omp for
		for (int i = 0; i < simulations; i++)
			PlaySimulation(local_field, local_gen, moves, &n);

#pragma omp critical
		{
			for (UctNode* next = n.child; next; next = next->sibling)
			{
				double cur_estimate = static_cast<double>(next->wins) / next->visits;
				//printf("%s %u (%d %d) %u %u\n", next->wins / (double)next->visits / 4 > 0.75 ? "@" : "!", next->move, cur_field->ToX(next->move), cur_field->ToY(next->move), next->wins, next->visits);
				//printf("! %u %u %u\n", next->move, next->wins, next->visits);
				//cout << "! " << next->move << " " << next->wins << " " << next->visits << "\n";
				//probs[next->move] = next->wins * 100 / 4 / next->visits;
				if (cur_estimate > best_estimate)
				{
					best_estimate = cur_estimate;
					result = next->move;
				}
			}
		}
	}

	//cout << "   :";
	//for (int i = 0; i < cur_field->width; i++)
	//	cout << std::setw(3) << i;
	//cout << "\n";
	//for (int i = cur_field->height - 1; i >= 0; i--)
	//{
	//	cout << std::setw(3) << i << ":";
	//	for (int j = 0; j < cur_field->width; j++)
	//	{
	//		cout << std::setw(3) << (int)probs[cur_field->ToMove(j, i)];
	//	}
	//	cout << "\n";
	//}

	for (Move move : tempBorder)
		field.DelBorder(move);

	return result;
}
