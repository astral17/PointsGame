#pragma once
#include <memory>
#include "fast_stack.h"

struct AStar
{
    AStar(Field& field);
    void BuildDistance();
	bool HaveExit(Move move);
	FastStack<Move> cur_stk, next_stk;
	std::unique_ptr<short[]> distRed, distBlack;
	Field& field;
};