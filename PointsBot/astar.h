#pragma once
#include <memory>
#include "fast_stack.h"

struct AStar
{
    AStar(Field& field);
    void BuildDistance();
	bool HaveExit(Move move);
	FastStack<Move> cur_stk, next_stk;
#ifdef ASTAR_EXTENDED_FEATURE
	std::unique_ptr<short[]> distRed, distBlack;
#else
	std::unique_ptr<short[]> dist;
#endif
	Field& field;
};