#pragma once

#include "field.h"
#include <random>

#define DLLEXPORT extern "C" __declspec(dllexport)

struct Bot
{
	Field field;
	std::mt19937 gen;
	Bot(int width, int height, int seed) : field(width, height), gen(seed) {}
};

DLLEXPORT Bot* Init(int width, int height, int seed);
DLLEXPORT void Finish(Bot* bot);
DLLEXPORT void MakeMove(Bot* bot, Move move);
DLLEXPORT void Undo(Bot* bot);
DLLEXPORT Move GenWithStrength(Bot* bot, int strength);
DLLEXPORT Move GenWithTime(Bot* bot, int time);
DLLEXPORT void GetField(Bot* bot, char* result);
DLLEXPORT int GetScore(Bot* bot, Player player);
DLLEXPORT bool IsFinished(Bot* bot);