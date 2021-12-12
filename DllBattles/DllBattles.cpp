#include <iostream>
#include <Windows.h>
#include "bot_proxy.h"
#include <memory>
#include <random>
#include <assert.h>

using namespace std;

//enum BattleResult
//{
//    FirstWin,
//    SecondWin,
//    Draw,
//};

mt19937 mt;
int GenRandom()
{
    return mt();
}

int Battle(unique_ptr<BotProxy> &bot1, int str1, unique_ptr<BotProxy> &bot2, int str2)
{
    int width = 10, height = 10;
    bot1->Init(width, height, GenRandom());
    bot2->Init(width, height, GenRandom());

    Move move;
    while (true)
    {
        if (bot1->IsFinished())
            break;
        move = bot1->GenWithStrength(str1);
        // TODO: Fix bug
        if (move == (Move)-1)
            break;
        bot1->MakeMove(move);
        bot2->MakeMove(move);
        if (bot2->IsFinished())
            break;
        move = bot2->GenWithStrength(str2);
        // TODO: Fix bug
        if (move == (Move)-1)
            break;
        bot1->MakeMove(move);
        bot2->MakeMove(move);
        //cout << bot1->GetField() << "\n";
    }
    int score1 = bot1->GetScore(0);
    int score2 = bot2->GetScore(0);
    assert(score1 == score2);
    bot1->Finish();
    bot2->Finish();
    return score1;
    //if (score1 > 0)
    //    return FirstWin;
    //if (score1 < 0)
    //    return SecondWin;
    //return Draw;
}

int main()
{
    mt.seed(time(0));
    constexpr int kBotCount = 2;
    unique_ptr<BotProxy> bots[kBotCount];
    int bstr[kBotCount];
    //bots[0] = unique_ptr<BotProxy>(new BotProxy("PointsBot_Uct_2_Score"));
    //bots[1] = unique_ptr<BotProxy>(new BotProxy("PointsBot_Uct_3_Score"));
    //bots[2] = unique_ptr<BotProxy>(new BotProxy("PointsBot_Uct_4_Score"));
    //bots[3] = unique_ptr<BotProxy>(new BotProxy("PointsBot_Random"));

    //bots[0] = unique_ptr<BotProxy>(new BotProxy("PointsBot_Uct_2_Score"));
    //bots[1] = unique_ptr<BotProxy>(new BotProxy("PointsBot_Uct_2_W4"));

    //bots[0] = unique_ptr<BotProxy>(new BotProxy("PointsBot_Uct_2_Score"));
    //bstr[0] = 100;
    //bots[1] = unique_ptr<BotProxy>(new BotProxy("PointsBot_Uct_2_Score"));
    //bstr[1] = 1000;
    //bots[2] = unique_ptr<BotProxy>(new BotProxy("PointsBot_Uct_2_Score"));
    //bstr[2] = 10000;

    bots[0] = unique_ptr<BotProxy>(new BotProxy("PointsBot_Uct_2_Score"));
    bstr[0] = 1000;
    bots[1] = unique_ptr<BotProxy>(new BotProxy("PointsBot_Bns_2"));
    bstr[1] = 4;

    for (int i = 0; i < kBotCount; i++)
        for (int j = 0; j < kBotCount; j++)
            if (i != j)
            //if (i != j && (i == 3 || j == 3))
            {
                int total = 0;
                cout << "Battle " << i << " " << j << ":";
                for (int k = 0; k < 10; k++)
                {
                    int score = Battle(bots[i], bstr[i], bots[j], bstr[j]);
                    cout << " " << score;
                    total += score;
                }
                cout << " = " << total / 10.0 << "\n";

            }
    // OpenMP Crash OMP_WAIT_POLICY
    cout << "Finished, wait OpenMP\n";
    Sleep(3000);
}
