// dllmain.cpp : Определяет точку входа для приложения DLL.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "dllmain.h"
#include "uct.h"
#include "bns.h"
#include <sstream>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

DLLEXPORT Bot* Init(int width, int height, int seed)
{
    return new Bot(width, height, seed);
}

DLLEXPORT void Finish(Bot* bot)
{
    delete bot;
}

DLLEXPORT void MakeMove(Bot* bot, Move move)
{
    bot->field.MakeMove(move);
}

DLLEXPORT void Undo(Bot* bot)
{
    bot->field.Undo();
}

DLLEXPORT Move GenWithStrength(Bot* bot, int strength)
{
    return BestNodeSearch(bot->field, bot->gen, strength);
    //return Uct(bot->field, bot->gen, strength);

    //MoveList moves = bot->field.GetAllMoves();
    //return moves[bot->gen() % moves.size()];
}

DLLEXPORT Move GenWithTime(Bot* bot, int time)
{
    return -1;
}

DLLEXPORT void GetField(Bot* bot, char* result)
{
    stringstream stream;
    bot->field.DebugPrint(stream);
    strcpy(result, stream.str().c_str());
}

DLLEXPORT int GetScore(Bot* bot, Player player)
{
    return bot->field.GetScore(player);
}

DLLEXPORT bool IsFinished(Bot* bot)
{
    return bot->field.GetAllMoves().size() == 0;
}
