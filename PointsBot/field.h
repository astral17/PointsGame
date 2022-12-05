#pragma once
#include <vector>
#include <queue>
#include <memory>

#include <iostream>
#include <iomanip>

#include "fast_queue.h"

using Move = uint16_t;
using Cell = uint8_t;
using Player = uint8_t;
using MoveList = std::vector<Move>;

constexpr Player kPlayerRed = 0;
constexpr Player kPlayerBlack = 1;

class Field
{
public:
	Field(short height, short width, Player player = kPlayerRed) : field(new Cell[(width + 2) * (height + 2)]), q((width + 2) * (height + 2)), width(width), height(height), player(player)
	{
		for (int i = 0; i < Length(); i++)
			field[i] = 0;
		for (int i = -1; i <= height; i++)
			field[ToMove(-1, i)] = field[ToMove(width, i)] = kBorderBit;
		for (int i = -1; i <= width; i++)
			field[ToMove(i, -1)] = field[ToMove(i, height)] = kBorderBit;
	}
	//Field(const Field& other) : Field(other.height, other.width, other.player)
	//{
	//	for (int i = 0; i < Length(); i++)
	//		field[i] = other.field[i];
	//	scores.back() = score = other.score;
	//}

	Field(const Field& other) : q((other.width + 2)* (other.height + 2)), field(new Cell[(other.width + 2) * (other.height + 2)]), width(other.width), height(other.height), score(other.score), player(other.player)
	{
		std::copy_n(other.field.get(), (width + 2) * (height + 2), field.get());
		scores.back() = score;
	}

	inline bool CouldMove(Move move) const { return !(field[move] & kCouldMove); }
	inline bool IsPoint(Move move) const { return field[move] & kPointBit; }
	inline bool IsPlayerPoint(const Move move, const Player player) const
	{
		return (field[move] & (kPlayerBit | kPointBit)) == (kPointBit | player); 
	}

	inline bool IsRedPoint(const Move move) const { return IsPoint(move) && !(field[move] & kColorBit); }
	inline bool IsBlackPoint(const Move move) const { return IsPoint(move) && (field[move] & kColorBit); }
	
	inline bool IsRedArea(const Move move) const { return (field[move] & kBaseBit) && GetOwner(move) == kPlayerRed; }
	inline bool IsBlackArea(const Move move) const { return (field[move] & kBaseBit) && GetOwner(move) == kPlayerBlack; }

	inline bool IsRedEmpty(const Move move) const { return IsEmptyBase(move) && GetOwner(move) == kPlayerRed; }
	inline bool IsBlackEmpty(const Move move) const { return IsEmptyBase(move) && GetOwner(move) == kPlayerBlack; }

	inline bool IsBorder(const Move move) const { return field[move] & kBorderBit; }
	inline bool HaveOwner(const Move move) const { return field[move] & kHaveOwner; }
	inline bool IsVisited(const Move move) const { return field[move] & kVisitedBit; }
	// !!! ≈сли клетка не имеет владельца, то поведение вообще говор€ неопределено
	inline Player GetOwner(const Move move) const { return field[move] & kPlayerBit; }

	inline bool IsOwner(const Move move, const Player player) const { return HaveOwner(move) && GetOwner(move) == player; }

	inline bool IsEmptyBase(const Move move) const { return field[move] & kEmptyBaseBit; }

	inline void SetBorder(const Move move) { field[move] |= kBorderBit; }
	inline void DelBorder(const Move move) { field[move] &= ~kBorderBit; }

	// —делать ход в указанную клетку (без проверок)
	void MakeMove(Move move);
	// ќтменить последний сделанный ход
	void Undo();
	// ѕолучить все ходы, которые может сделать указанный игрок
	MoveList GetAllMoves(Player player) const;
	// ѕолучить все ходы, которые может сделать текущий игрок
	MoveList GetAllMoves() const { return GetAllMoves(player); }

	inline Move Up(const Move move) const { return move - width - 2; }
	inline Move Down(const Move move) const { return move + width + 2; };
	inline Move Left(const Move move) const { return move - 1; };
	inline Move Right(const Move move) const { return move + 1; };

	Player NextPlayer(Player player) const { return 1 - player; }
	Player GetPlayer() const { return player; }
	Player GetNextPlayer() const { return NextPlayer(player); }

	int GetScore(Player player) const;

	Move Length() const { return (width + 2) * (height + 2); }
	Move ToMove(int x, int y) const { return (y + 1) * (width + 2) + x + 1; }
	Move MinMove() const { return ToMove(0, 0); }
	inline Move MaxMove() const { return ToMove(width - 1, height - 1); }

	inline static short ToX(Move move, short width) { return move % (width + 2) - 1; }
	inline static short ToY(Move move, short width) { return move / (width + 2) - 1; }
	inline short ToX(Move move) const { return ToX(move, width); }
	inline short ToY(Move move) const { return ToY(move, width); }

	void DebugPrint(std::ostream &out, const std::string s = "", bool axis = true, bool colors = false) const
	{
		out << s << "\n";
		if (axis)
		{
			out << "   :";
			for (int i = 0; i < width; i++)
				out << std::setw(3) << i;
			out << "\n";
		}
		for (int i = height - 1; i >= 0; i--)
		{
			if (axis)
				out << std::setw(3) << i << ":";
			for (int j = 0; j < width; j++)
			{
				if (colors && !CouldMove(ToMove(j, i)))
				{
					out << "\x1B[" << (GetOwner(ToMove(j, i)) ? 34 : 31) << "m";
				}
				out << std::setw(2 + axis) << (int)field[ToMove(j, i)] << (axis ? "" : ",");
				if (colors && !CouldMove(ToMove(j, i)))
				{
					out << "\033[0m";
				}
			}
			out << "\n";
		}
	}
private:
	inline void ApplyFlag(Move move, Cell flag) { field[move] |= flag; }
	inline void SetOwner(Move move, Player owner) { field[move] = (field[move] & ~kPlayerBit) | owner; }
	// —охранить клетку дл€ возможности отката ходов
	inline void AddToBackup(Move move)
	{
		changes.emplace_back(move, field[move]);
	}
	// «афиксировать изменени€, т.е. контрольна€ точка дл€ отката хода
	inline void CommitChanges()
	{
		history.push_back(changes.size());
		scores.push_back(score);
	}
	inline void UnTagQueue();
	//bool TryCapture(Move start, MoveList &visited);
	bool TryCapture(Move start);
	void CaptureEmpty(Move start);
	void DestroyEmpty(Move start);
	enum Masks : Cell
	{
		// Cell owner
		kPlayerBit = 0x1,
		kPointBit = 0x2,
		kBaseBit = 0x4,
		kEmptyBaseBit = 0x8,
		kBorderBit = 0x10,
		kVisitedBit = 0x20,
		// Original points color
		kColorBit = 0x40,
		kColorShift = 6,
		kCouldMove = kPointBit | kBaseBit | kBorderBit,
		kHaveOwner = kPointBit | kBaseBit | kEmptyBaseBit,
	};
	// TODO: Private
public:
	std::vector< std::pair<Move, Cell> > changes;
	std::vector<int> history{ 0 };
	std::vector<int> scores{ 0 };
	FastQueue<Move> q;
	std::unique_ptr<Cell[]> field;
	// Zobrist or BCH Hash

	int score = 0;
	const short width, height;
	Player player;
};