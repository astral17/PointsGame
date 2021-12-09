#pragma once

#include "field.h"
#include <random>
#include <list>
#include <stack>

using namespace std;

#define NOMINMAX 1
#define WINDOWS 1

struct UctNode
{
	uint32_t wins;
	uint32_t visits;
	Move move;
	UctNode* parent;
	UctNode* child;
	UctNode* sibling;

	UctNode(UctNode* parent = nullptr) : wins(0), visits(0), move(0), parent(parent), child(nullptr), sibling(nullptr) {}
	~UctNode()
	{
		delete child;
		delete sibling;
	}
};

Move Uct(Field& field, mt19937& gen, int simulations);