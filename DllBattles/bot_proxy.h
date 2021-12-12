#pragma once

#include <iostream>

typedef uint16_t Move;
typedef uint8_t Player;
typedef void* (__cdecl* InitFunc)(int, int, int);
typedef void(__cdecl* FinishFunc)(void*);
typedef void(__cdecl* MakeMoveFunc)(void*, Move);
typedef void(__cdecl* UndoFunc)(void*);
typedef Move(__cdecl* GenWithStrengthFunc)(void*, int);
typedef Move(__cdecl* GenWithTimeFunc)(void*, int);
typedef void(__cdecl* GetFieldFunc)(void*, char*);
typedef int(__cdecl* GetScoreFunc)(void*, Player);
typedef bool(__cdecl* IsFinishedFunc)(void*);

//DLLEXPORT Bot* Init(int width, int height, int seed);
//DLLEXPORT void Finish(Bot* bot);
//DLLEXPORT void MakeMove(Bot* bot, Move move);
//DLLEXPORT void Undo(Bot* bot);
//DLLEXPORT Move GenWithStrength(Bot* bot, int strength);
//DLLEXPORT Move GenWithTime(Bot* bot, int time);
//DLLEXPORT void GetField(Bot* bot, char* result);

class BotProxy
{
    HMODULE hLib;
    InitFunc init;
    FinishFunc finish;
    MakeMoveFunc makeMove;
    UndoFunc undo;
    GenWithStrengthFunc genWithStrength;
    GenWithTimeFunc genWithTime;
    GetFieldFunc getField;
    GetScoreFunc getScore;
    IsFinishedFunc isFinished;
    void* bot = nullptr;
    int width, height;
public:
    BotProxy(const std::string name)
    {
        hLib = LoadLibraryA(name.c_str());
        if (!hLib)
        {
            std::cerr << "PointsBot.dll load failed with: " << GetLastError() << "\n";
            exit(-1);
        }
        init = (InitFunc)GetProcAddress(hLib, "Init");
        finish = (FinishFunc)GetProcAddress(hLib, "Finish");
        makeMove = (MakeMoveFunc)GetProcAddress(hLib, "MakeMove");
        undo = (UndoFunc)GetProcAddress(hLib, "Undo");
        genWithStrength = (GenWithStrengthFunc)GetProcAddress(hLib, "GenWithStrength");
        genWithTime = (GenWithTimeFunc)GetProcAddress(hLib, "GenWithTime");
        getField = (GetFieldFunc)GetProcAddress(hLib, "GetField");
        getScore = (GetScoreFunc)GetProcAddress(hLib, "GetScore");
        isFinished = (IsFinishedFunc)GetProcAddress(hLib, "IsFinished");
    }
    ~BotProxy()
    {
        if (bot)
            Finish();
        if (hLib)
            FreeLibrary(hLib);
    }

    void Init(int width, int height, int seed)
    {
        this->width = width;
        this->height = height;
        bot = init(width, height, seed);
    }
    void Finish()
    {
        finish(bot);
        bot = nullptr;
    }
    void MakeMove(Move move)
    {
        makeMove(bot, move);
    }
    void MakeMove(short x, short y)
    {
        makeMove(bot, ToMove(x, y));
    }
    void Undo()
    {
        undo(bot);
    }
    Move GenWithStrength(int strength)
    {
        return genWithStrength(bot, strength);
    }
    Move GenWithTime(int time)
    {
        return genWithTime(bot, time);
    }
    std::string GetField()
    {
        // TODO: Dynamic size
        char str[4096];
        getField(bot, str);
        return str;
    }
    int GetScore(Player player)
    {
        return getScore(bot, player);
    }
    bool IsFinished()
    {
        return isFinished(bot);
    }

    Move ToMove(int x, int y) const { return (y + 1) * (width + 2) + x + 1; }

    short ToX(const short cur_pos) const { return cur_pos % (width + 2) - 1; }
    short ToY(const short cur_pos) const { return cur_pos / (width + 2) - 1; }
};