#include "train.h"
#include "mcts.h"

bool PvP(Strategy& a, Strategy& b, Player first, MoveStorage* storage)
{
    Strategy* cur_player = &a;
    Strategy* next_player = &b;
    if (first != kPlayerRed)
        std::swap(cur_player, next_player);
    Field field(FIELD_HEIGHT, FIELD_WIDTH);
    std::vector<int> pos;
    while (!field.GetAllMoves().empty())
    {
        if (storage)
            pos.push_back(storage->r);
        cur_player->MakeMove(field, storage);
        std::swap(cur_player, next_player);
    }
    if (storage)
    {
        float score = field.GetScore(first);
        if (score > 0)
            score = 1;
        if (score < 0)
            score = -1;
        for (int x : pos)
        {
            storage->operator[](x).value = score;
            score *= -1;
        }
    }
    //std::cout << "Score: " << field.GetScore(first) << "\n";
    return field.GetScore(first) > 0;
}

void Trainer()
{
    // Цель создать сеть, которая сможет работать с MCTS и Field предсказывать результаты
    // Есть набор сохранённых ходов: вход - состояние доски, выход - вероятности ходов, тег победа/поражение
    // 

    //StrategyContainer best_strategy(new MctsStrategy(10));
    StrategyContainer test_strategy(new MctsStrategy(100));
    StrategyContainer best_strategy(new NeuralStrategy());
    StrategyContainer cur_strategy(new NeuralStrategy());

    MoveStorage storage(20000);
    std::mt19937 gen(1921);
    best_strategy.strategy().Randomize(gen());
    //((NeuralStrategy*)best_strategy.strategy_.get())->net.LoadWeights("weights_last.bwf");
    //((NeuralStrategy*)cur_strategy.strategy_.get())->net.LoadWeights("dense_only/weights_5216_1.bwf");

    //StrategyContainer rnd_strategy(new RandomStrategy());
    //int wins = 0, total = 10;
    //for (int i = 0; i < total; i++)
    //    if (PvP(best_strategy.strategy(), test_strategy.strategy(), gen() & 1))
    //        wins++;
    //std::cout << "Result: " << wins << "/" << total << "\n";
    //return;
    //for (int i = 0; i < 100; i++)
    //    PvP(best_strategy.strategy(), best_strategy.strategy(), gen() & 1, &storage);
    for (size_t run = 0;; run++)
    {
        //cur_strategy.strategy().Randomize(gen());
        //cur_strategy.strategy().Train(storage);
        best_strategy.strategy().Train(storage);
        std::cout << "Iteration: " << run << "\n";
        //int wins = 0, total = 10;
        //for (int i = 0; i < total; i++)
        //{
        //    // bestNet vs curNet
        //    if (PvP(cur_strategy.strategy(), best_strategy.strategy(), i & 1))
        //        wins++;
        //}
        //if (wins / (double)total > 0.55)
        //{
        //    std::cout << "WINNER: ";
        //    best_strategy = std::move(cur_strategy);
        //    //best_strategy.strategy_.reset(cur_strategy.strategy_.release());
        //    cur_strategy.strategy_.reset(new NeuralStrategy());
        //}
        //else
        //    std::cout << "loser:  ";
        //std::cout << wins << "/" << total << "\n";
        for (int i = 0; i < 25; i++)
        {
            PvP(best_strategy.strategy(), best_strategy.strategy(), gen() & 1, &storage);
            // bestNet vs bestNet
            // save moves to storage
        }
        if (run % 16 == 0)
        {
            int wins = 0, total = 10;
            for (int i = 0; i < total; i++)
            {
                if (PvP(best_strategy.strategy(), test_strategy.strategy(), i & 1))
                    wins++;
            }
            std::cout << "Test: " << wins << "/" << total << "\n";
            if (wins > 1)
                ((NeuralStrategy*)best_strategy.strategy_.get())->net.SaveWeights("weights_" + std::to_string(run) + "_" + std::to_string(wins) + ".bwf");
        }
        if (run % 32 == 0)
            ((NeuralStrategy*)best_strategy.strategy_.get())->net.SaveWeights("weights_last.bwf");
    }
}


// Базовая структура
// 400х400х401
// 200 ходов, 10s вся игра
// 20 игр, 55% винрейт замена сети
// 100 игр сам с собой, 10 ходов из каждой игры
// 10000 ходов тренировка сети 100 батчей по 128
