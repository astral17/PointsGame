#include "train.h"
#include "mcts.h"

dnnl::engine NeuralStrategy::eng(dnnl::engine::kind::cpu, 0);

void RightRotateTo(const MoveRecord& src, MoveRecord& dst)
{
    dst.value = src.value;
    auto src_position = (float(*)[FIELD_HEIGHT][FIELD_WIDTH])src.position.data();
    dst.position.resize(src.position.size());
    auto dst_position = (float(*)[FIELD_HEIGHT][FIELD_WIDTH])dst.position.data();
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < FIELD_HEIGHT; j++)
            for (int k = 0; k < FIELD_WIDTH; k++)
                dst_position[i][k][FIELD_WIDTH - j - 1] = src_position[i][j][k];

    auto src_policy = (float(*)[FIELD_WIDTH])src.policy.data();
    dst.policy.resize(src.policy.size());
    auto dst_policy = (float(*)[FIELD_WIDTH])dst.policy.data();
    for (int i = 0; i < FIELD_HEIGHT; i++)
        for (int j = 0; j < FIELD_WIDTH; j++)
            dst_policy[j][FIELD_WIDTH - i - 1] = src_policy[i][j];
}

void HorizontalMirrorTo(const MoveRecord& src, MoveRecord& dst)
{
    dst.value = src.value;
    auto src_position = (float(*)[FIELD_HEIGHT][FIELD_WIDTH])src.position.data();
    dst.position.resize(src.position.size());
    auto dst_position = (float(*)[FIELD_HEIGHT][FIELD_WIDTH])dst.position.data();
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < FIELD_HEIGHT; j++)
            for (int k = 0; k < FIELD_WIDTH; k++)
                dst_position[i][j][FIELD_WIDTH - k - 1] = src_position[i][j][k];

    auto src_policy = (float(*)[FIELD_WIDTH])src.policy.data();
    dst.policy.resize(src.policy.size());
    auto dst_policy = (float(*)[FIELD_WIDTH])dst.policy.data();
    for (int i = 0; i < FIELD_HEIGHT; i++)
        for (int j = 0; j < FIELD_WIDTH; j++)
            dst_policy[i][FIELD_WIDTH - j - 1] = src_policy[i][j];
}

void AugmentLast(MoveStorage* storage, std::vector<int>& pos)
{
    assert(FIELD_HEIGHT == FIELD_WIDTH);
    assert(storage->capacity >= 8);
    int src;
    for (int i = 0; i < 3; i++)
    {
        src = pos.back();
        pos.push_back(storage->r);
        RightRotateTo((*storage)[src], (*storage)[pos.back()]);
        storage->next();
    }
    src = pos.back();
    pos.push_back(storage->r);
    HorizontalMirrorTo((*storage)[src], (*storage)[pos.back()]);
    storage->next();
    for (int i = 0; i < 3; i++)
    {
        src = pos.back();
        pos.push_back(storage->r);
        RightRotateTo((*storage)[src], (*storage)[pos.back()]);
        storage->next();
    }
}

bool PvP(Strategy& a, Strategy& b, Player first, MoveStorage* storage, bool train)
{
    Strategy* cur_player = &a;
    Strategy* next_player = &b;
    if (first != kPlayerRed)
        std::swap(cur_player, next_player);
    Field field(FIELD_HEIGHT, FIELD_WIDTH);
    std::vector<int> pos[2];
    while (!field.GetAllMoves().empty())
    {
        if (storage)
            pos[field.GetPlayer()].push_back(storage->r);
        cur_player->MakeMove(field, storage, train);
        if (storage)
            AugmentLast(storage, pos[field.GetNextPlayer()]);
        std::swap(cur_player, next_player);
    }
    if (storage)
    {
        float score = field.GetScore(kPlayerRed);
        if (score > 0)
            score = 1;
        if (score < 0)
            score = -1;
        for (int x : pos[kPlayerRed])
            (*storage)[x].value = score;
        for (int x : pos[kPlayerBlack])
            (*storage)[x].value = -score;
    }
    if (!storage)
        std::cout << "Score: " << field.GetScore(first) << "\n";
    return field.GetScore(first) > 0;
}

void Trainer()
{
    // Цель создать сеть, которая сможет работать с MCTS и Field предсказывать результаты
    // Есть набор сохранённых ходов: вход - состояние доски, выход - вероятности ходов, тег победа/поражение
    // 

    //StrategyContainer best_strategy(new MctsStrategy(10));
    StrategyContainer test_strategy(new MctsStrategy(100));
    //StrategyContainer test_strategy(new NeuralStrategy(500));
    //StrategyContainer test_strategy(new RandomStrategy());
    StrategyContainer train_strategy(new MctsStrategy(10000));
    StrategyContainer best_strategy(new NeuralStrategy(100));
    //StrategyContainer best_strategy(new RandomStrategy());
    //StrategyContainer cur_strategy(new NeuralStrategy(100));

    MoveStorage storage(FIELD_HEIGHT * FIELD_WIDTH * 50 * 8 * 4);
    //std::mt19937 gen(1351925);
    std::mt19937 gen(time(0));
    train_strategy.strategy().Randomize(gen());
    //best_strategy.strategy().Randomize(gen());
    ((NeuralStrategy*)best_strategy.strategy_.get())->net.LoadWeights("weights_last.bwf");
    //((NeuralStrategy*)cur_strategy.strategy_.get())->net.LoadWeights("weights_last_cnn_1.bwf");

    //StrategyContainer rnd_strategy(new RandomStrategy());
    //rnd_strategy.strategy().Randomize(gen());
    //int wins = 0, total = 100;
    //for (int i = 0; i < total; i++)
    //    //if (PvP(best_strategy.strategy(), test_strategy.strategy(), gen() & 1))
    //    //if (PvP(test_strategy.strategy(), test_strategy.strategy(), i & 1))
    //    //if (PvP(best_strategy.strategy(), best_strategy.strategy(), i & 1))
    //    //if (PvP(best_strategy.strategy(), cur_strategy.strategy(), i & 1))
    //    //if (PvP(best_strategy.strategy(), test_strategy.strategy(), i & 1))
    //    if (PvP(best_strategy.strategy(), rnd_strategy.strategy(), i & 1))
    //    //if (PvP(rnd_strategy.strategy(), rnd_strategy.strategy(), i & 1))
    //    //if (PvP(test_strategy.strategy(), rnd_strategy.strategy(), i & 1))
    //        wins++;
    //std::cout << "Result: " << wins << "/" << total << "\n";
    //return;
    //std::cout << (int)PvP(best_strategy.strategy(), train_strategy.strategy(), gen() & 1, &storage) << "\n";
    //for (int m = 0; m < storage.r; m++)
    //{
    //    std::cout << "--- MOVE: " << m / 8 << " ---\n";
    //    auto s = storage[m];
    //    float(*s_input)[FIELD_HEIGHT][FIELD_WIDTH] = (float(*)[FIELD_HEIGHT][FIELD_WIDTH])s.position.data();
    //    for (int i = 0; i < 6; i++)
    //    {
    //        for (int j = 4; j >= 0; j--)
    //        {
    //            for (int k = 0; k < 5; k++)
    //                std::cout << s_input[i][j][k] << " ";
    //            std::cout << "\n";
    //        }
    //        std::cout << "\n";
    //    }
    //    NNetwork& net = ((NeuralStrategy*)best_strategy.strategy_.get())->net;
    //    net.input << s_input;
    //    dnnl::stream st(net.engine);
    //    net.Forward(st);
    //    std::vector<float> tmp;
    //    net.output(0) >> tmp;
    //    float(*o_policy)[FIELD_WIDTH] = (float(*)[FIELD_WIDTH])tmp.data();
    //    for (int i = 4; i >= 0; i--)
    //    {
    //        for (int j = 0; j < 5; j++)
    //            std::cout << o_policy[i][j] << " ";
    //        std::cout << "\n";
    //    }
    //    net.output(1) >> tmp;
    //    std::cout << tmp[0] << "\n";
    //    float(*s_policy)[FIELD_WIDTH] = (float(*)[FIELD_WIDTH])s.policy.data();
    //    for (int i = 4; i >= 0; i--)
    //    {
    //        for (int j = 0; j < 5; j++)
    //            std::cout << s_policy[i][j] << " ";
    //        std::cout << "\n";
    //    }

    //    std::cout << s.value << "\n";
    //    std::getchar();
    //}
    //return;
    //for (int i = 0; i < 100; i++)
    //    //    //PvP(best_strategy.strategy(), best_strategy.strategy(), gen() & 1, &storage);
    //    PvP(train_strategy.strategy(), train_strategy.strategy(), gen() & 1, &storage, true);
    //best_strategy.strategy().Train(storage);
    //for (int i = 0; i < 10; i++)
    //    PvP(best_strategy.strategy(), test_strategy.strategy(), gen() & 1);
    //return;
    for (size_t run = 0;; run++)
    {
        //cur_strategy.strategy().Randomize(gen());
        //cur_strategy.strategy().Train(storage);
        std::cout << "Iteration: " << run << ".";
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
        for (int i = 0; i < 50; i++)
            PvP(best_strategy.strategy(), best_strategy.strategy(), gen() & 1, &storage, true);
        //for (int i = 100; i < 100; i++)
        //{
        //    /*PvP(best_strategy.strategy(), best_strategy.strategy(), gen() & 1, &storage, true);*/
        //    PvP(train_strategy.strategy(), train_strategy.strategy(), gen() & 1, &storage, true);
        //    //PvP(best_strategy.strategy(), train_strategy.strategy(), gen() & 1, &storage, true);
        //    // bestNet vs bestNet
        //    // save moves to storage
        //}
        std::cout << ".";
        best_strategy.strategy().Train(storage);
        std::cout << ".\n";
        if (run % 2 == 0)
        {
            int wins = 0, total = 10;
            for (int i = 0; i < total; i++)
            {
                if (PvP(best_strategy.strategy(), test_strategy.strategy(), i & 1))
                    wins++;
            }
            std::cout << "Test: " << wins << "/" << total << "\n";
            if (wins > 3)
                ((NeuralStrategy*)best_strategy.strategy_.get())->net.SaveWeights("weights_" + std::to_string(run) + "_" + std::to_string(wins) + ".bwf");
        }
        if (run % 2 == 0)
            ((NeuralStrategy*)best_strategy.strategy_.get())->net.SaveWeights("weights_last.bwf");
    }
}


// Базовая структура
// 400х400х401
// 200 ходов, 10s вся игра
// 20 игр, 55% винрейт замена сети
// 100 игр сам с собой, 10 ходов из каждой игры
// 10000 ходов тренировка сети 100 батчей по 128
