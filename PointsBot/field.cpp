#include "field.h"

Field::Field(short height, short width, Player player) : field(new Cell[(width + 2) * (height + 2)]), q((width + 2)* (height + 2)), width(width), height(height), player(player), astar(*this)
{
	for (int i = 0; i < Length(); i++)
		field[i] = 0;
	for (int i = -1; i <= height; i++)
		field[ToMove(-1, i)] = field[ToMove(width, i)] = kBorderBit;
	for (int i = -1; i <= width; i++)
		field[ToMove(i, -1)] = field[ToMove(i, height)] = kBorderBit;
}

Field::Field(const Field& other) : q((other.width + 2)* (other.height + 2)), field(new Cell[(other.width + 2) * (other.height + 2)]), width(other.width), height(other.height), score(other.score), player(other.player), astar(*this)
{
	std::copy_n(other.field.get(), (width + 2) * (height + 2), field.get());
	scores.back() = score;
}

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
		if ((Left(Up(move)) & kPointBit) + (Left(move) & kPointBit))
			TryCapture(Up(move));
		if ((Right(move) & kPointBit) + (Right(Down(move)) & kPointBit))
			TryCapture(Down(move));
		if ((Down(Left(move)) & kPointBit) + (Down(move) & kPointBit))
			TryCapture(Left(move));
		if ((Up(move) & kPointBit) + (Up(Right(move)) & kPointBit))
			TryCapture(Right(move));
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

inline void Field::UnTagQueue()
{
	for (int i = 0; i < q.r; i++)
		field[q[i]] &= ~kVisitedBit;
}

bool Field::TryCapture(Move start)
{
	if (IsBorder(start) || IsOwner(start, player))
		return false;
	q.clear();
	if (astar.HaveExit(start))
	{
		for (int i = 0; i < q.r; i++)
			field[q[i]] &= ~kVisitedBit;
		return false;
	}
	int captured = 0;
	for (int i = 0; i < q.r; i++)
		captured += IsPoint(q[i]);
	Cell addFlag = captured ? kBaseBit : kEmptyBaseBit;
	for (int i = 0; i < q.r; i++)
	{
		field[q[i]] &= ~kVisitedBit;
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
