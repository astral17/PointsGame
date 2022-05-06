#include "field.h"

void Field::MakeMove(Move move)
{
	AddToBackup(move);
	Cell old = field[move];
	SetOwner(move, player);
	//ApplyFlag(move, player | kPointBit);
	ApplyFlag(move, kPointBit | (player << kColorShift));
	// Произошёл ли захват?, Если да, то обновить поле
	//MoveList visited;
	int ucnt = (Up(Left(move)) & kPointBit) + (Up(move) & kPointBit) + (Up(Right(move)) & kPointBit);
	int dcnt = (Down(Left(move)) & kPointBit) + (Down(move) & kPointBit) + (Down(Right(move)) & kPointBit);
	int lcnt = (Left(Up(move)) & kPointBit) + (Left(move) & kPointBit) + (Left(Down(move)) & kPointBit);
	int rcnt = (Right(Up(move)) & kPointBit) + (Right(move) & kPointBit) + (Right(Down(move)) & kPointBit);
	if (ucnt && dcnt || lcnt && rcnt)
	{
		//TryCapture(Up(move), visited);
		//TryCapture(Down(move), visited);
		//TryCapture(Left(move), visited);
		//TryCapture(Right(move), visited);
		// TODO: сделать очистку внутри TryCapture
		TryCapture(Up(move));
		for (int i = 0; i < q.r; i++)
			field[q[i]] &= ~kVisitedBit;
		TryCapture(Down(move));
		for (int i = 0; i < q.r; i++)
			field[q[i]] &= ~kVisitedBit;
		TryCapture(Left(move));
		for (int i = 0; i < q.r; i++)
			field[q[i]] &= ~kVisitedBit;
		TryCapture(Right(move));
		for (int i = 0; i < q.r; i++)
			field[q[i]] &= ~kVisitedBit;
	}
	player = NextPlayer(player);
	// Ход сейчас в пустой вражеской зоне?, Получи и захватись
	if (old & kEmptyBaseBit)
	{
		// Если зона принадлежит врагу
		if ((old & kPlayerBit) == player)
		{
			// Если не было захвата, то точка захватывается
			if (score == scores.back())
			{
				//TryCapture(move, visited);
				CaptureEmpty(move);
				score += player == 0 ? 1 : -1;
			}
			else
			{
				// Иначе нужно разрушить базу
				DestroyEmpty(move);
			}
		}
		else
		{
			// Своя пустая зона, просто убрать пометку пустоты
			AddToBackup(move);
			field[move] &= ~kEmptyBaseBit;
		}
	}
	//for (int i = 0; i < q.r; i++)
	//	field[ q[i] ] &= ~kVisitedBit;
	//for (Move move : visited)
	//	field[move] &= ~kVisitedBit;
	CommitChanges();
}

void Field::Undo()
{
	history.pop_back();
	scores.pop_back();
	score = scores.back();
	while (changes.size() > history.back())
	{
		field[changes.back().first] = changes.back().second;
		changes.pop_back();
	}
	player = NextPlayer(player);
}

MoveList Field::GetAllMoves(Player player) const
{
	MoveList list;
	for (int i = 0; i < Length(); i++)
		if (CouldMove(i))
			list.push_back(i);
	return list;
}

int Field::GetScore(Player player) const
{
	return player == 0 ? score : -score;
}

bool Field::TryCapture(Move start)
{
	// TODO: A*
	if (IsBorder(start) || IsOwner(start, player))
		return false;
	//std::queue<Move> q;
	q.clear();
	q.push(start);
	int captured = 0;
	int cnt = 0;

	if (IsPoint(start) && IsOwner(start, NextPlayer(player)))
		captured++;
	field[start] |= kVisitedBit;

	while (!q.empty())
	{
		Move move = q.front();
		q.pop();
		// Если сосед граница (возможно искусственно установленная эвристикой для быстрого выхода), то окружить нельзя
		// В посещённые клетки ходить нет смысла, а также мы не можем проходить сквозь свои клетки
		if (IsBorder(Up(move)))
			return false;
		if (!IsVisited(Up(move)) && !IsOwner(Up(move), player))
		{
			// Т.к. это гарантированно не наша клетка, то раз это точка, то захватываем
			if (IsPoint(Up(move)))
				captured++;
			field[Up(move)] |= kVisitedBit;
			q.push(Up(move));
		}
		if (IsBorder(Down(move)))
			return false;
		if (!IsVisited(Down(move)) && !IsOwner(Down(move), player))
		{
			if (IsPoint(Down(move)))
				captured++;
			field[Down(move)] |= kVisitedBit;
			q.push(Down(move));
		}
		if (IsBorder(Left(move)))
			return false;
		if (!IsVisited(Left(move)) && !IsOwner(Left(move), player))
		{
			if (IsPoint(Left(move)))
				captured++;
			field[Left(move)] |= kVisitedBit;
			q.push(Left(move));
		}
		if (IsBorder(Right(move)))
			return false;
		if (!IsVisited(Right(move)) && !IsOwner(Right(move), player))
		{
			if (IsPoint(Right(move)))
				captured++;
			field[Right(move)] |= kVisitedBit;
			q.push(Right(move));
		}
	}
	Cell addFlag = captured ? kBaseBit : kEmptyBaseBit;
	for (int i = 0; i < q.r; i++)
	{
		AddToBackup(q[i]);
		SetOwner(q[i], player);
		ApplyFlag(q[i], addFlag);
	}

	score += player == 0 ? captured : -captured;
	return captured;
}

void Field::CaptureEmpty(Move start)
{
	//std::queue<Move> q;
	q.clear();
	q.push(start);

	AddToBackup(start);
	SetOwner(start, player);
	field[start] &= ~kEmptyBaseBit;
	field[start] |= kBaseBit;

	while (!q.empty())
	{
		Move move = q.front();
		q.pop();
		if (IsEmptyBase(Up(move)))
		{
			AddToBackup(Up(move));
			field[Up(move)] &= ~kEmptyBaseBit;
			field[Up(move)] |= kBaseBit;
			q.push(Up(move));
		}
		if (IsEmptyBase(Down(move)))
		{
			AddToBackup(Down(move));
			field[Down(move)] &= ~kEmptyBaseBit;
			field[Down(move)] |= kBaseBit;
			q.push(Down(move));
		}
		if (IsEmptyBase(Left(move)))
		{
			AddToBackup(Left(move));
			field[Left(move)] &= ~kEmptyBaseBit;
			field[Left(move)] |= kBaseBit;
			q.push(Left(move));
		}
		if (IsEmptyBase(Right(move)))
		{
			AddToBackup(Right(move));
			field[Right(move)] &= ~kEmptyBaseBit;
			field[Right(move)] |= kBaseBit;
			q.push(Right(move));
		}
	}
}

void Field::DestroyEmpty(Move start)
{
	//std::queue<Move> q;
	q.clear();
	q.push(start);

	AddToBackup(start);
	field[start] &= ~kEmptyBaseBit;

	while (!q.empty())
	{
		Move move = q.front();
		q.pop();
		if (IsEmptyBase(Up(move)))
		{
			AddToBackup(Up(move));
			field[Up(move)] &= ~kEmptyBaseBit;
			q.push(Up(move));
		}
		if (IsEmptyBase(Down(move)))
		{
			AddToBackup(Down(move));
			field[Down(move)] &= ~kEmptyBaseBit;
			q.push(Down(move));
		}
		if (IsEmptyBase(Left(move)))
		{
			AddToBackup(Left(move));
			field[Left(move)] &= ~kEmptyBaseBit;
			q.push(Left(move));
		}
		if (IsEmptyBase(Right(move)))
		{
			AddToBackup(Right(move));
			field[Right(move)] &= ~kEmptyBaseBit;
			q.push(Right(move));
		}
	}
}
