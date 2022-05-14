#pragma once
#include "field.h"
#include "layers.h"
#include "uct.h"
#include <typeinfo>
#include <algorithm>
#include <iostream>
#include <memory>
#include "mcts.h"

struct MoveRecord
{
    std::vector<float> position;
    std::vector<float> policy;
    float value = 0;
    MoveRecord() {}
    MoveRecord(const std::vector<float>& position, const std::vector<float>& policy, float value)
        : position(position), policy(policy), value(value) {}
};

struct MoveStorage
{
    //MoveStorage(int capacity) {}
    
public:
    MoveStorage(int capacity) : q(new MoveRecord[capacity]), capacity(capacity), size(0), r(0) {}

    inline MoveRecord& operator[](const int i) const { return q[i]; }
    inline void clear() { size = r = 0; }
    inline bool empty() const { return size == 0; }
    inline void next()
    {
        r++;
        size = std::max(size, r);
        if (r >= capacity)
            r = 0;
    }
    inline void push(const MoveRecord& x)
    {
        q[r] = x;
        next();
    }
public:
    std::unique_ptr<MoveRecord[]> q;
    int capacity, size, r;
};

struct Strategy
{
    virtual void Train(const MoveStorage& storage) {}
    virtual void Randomize(int seed) {}
    virtual void CopyFrom(Strategy &other) {}
    virtual void MakeMove(Field& field, MoveStorage* storage = nullptr, bool train = false) = 0;
};

struct StrategyContainer
{
    std::unique_ptr<Strategy> strategy_;

    StrategyContainer(Strategy* strategy) : strategy_(strategy) {}

    Strategy& strategy() { return *strategy_; }
    StrategyContainer& operator=(StrategyContainer&& other) noexcept
    {
        strategy_.release();
        std::swap(strategy_, other.strategy_);
        //if (typeid(strategy()) == typeid(other.strategy()))
        //{
        //    strategy().CopyFrom(other.strategy());
        //}
        //else
        //{
        //    // CLONE?
        //}
        return *this;
    }
};

struct NeuralStrategy : Strategy
{
    const int batch_size = 128;
    dnnl::engine eng;
    NNetwork train_net, net;
    std::mt19937 gen;
    int strength;
    //static dnnl::engine eng;
    virtual NNetwork Build(const dnnl::engine& eng, dnnl::prop_kind kind, int batch)
    {
        NNetwork net(eng, kind);

        constexpr int kBlocks = 2;
        constexpr int kFilters = 32;
        constexpr int kPolicyFilters = 16;
        constexpr int kValueFilters = 16;
        constexpr int kSeSize = 8;
        constexpr int kReluAlpha = 0;
        std::vector<std::unique_ptr<Layer>> layers;
        layers.emplace_back(new InputLayer(net, { batch, 6, FIELD_HEIGHT, FIELD_WIDTH }));
        layers.emplace_back(new ConvLayer(*layers.back(), { kFilters, 6, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }));
        layers.emplace_back(new BatchNormLayer(*layers.back()));

        for (int i = 0; i < kBlocks; i++)
        {
            Layer& residual = *layers.back();
            layers.emplace_back(new ConvLayer(residual, { kFilters, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }));
            layers.emplace_back(new BatchNormLayer(*layers.back()));
            layers.emplace_back(new ConvLayer(*layers.back(), { kFilters, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }));
            layers.emplace_back(new BatchNormLayer(*layers.back()));
            layers.emplace_back(new SELayer(*layers.back(), kSeSize));
            layers.emplace_back(new LayerAdder(residual, *layers.back()));
            layers.emplace_back(new ReluLayer(*layers.back(), kReluAlpha));
        }
        Layer& tower_last = *layers.back();
        layers.emplace_back(new ConvLayer(tower_last, { kPolicyFilters, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }));
        layers.emplace_back(new BatchNormLayer(*layers.back()));
        layers.emplace_back(new ReluLayer(*layers.back(), kReluAlpha));
        layers.emplace_back(new DenseLayer(*layers.back(), FIELD_HEIGHT * FIELD_WIDTH));
        layers.emplace_back(new LogSoftMaxLayer(*layers.back()));
        Layer& policy_layer = *layers.back();

        layers.emplace_back(new ConvLayer(tower_last, { kValueFilters, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }));
        layers.emplace_back(new BatchNormLayer(*layers.back()));
        layers.emplace_back(new ReluLayer(*layers.back(), kReluAlpha));
        layers.emplace_back(new DenseLayer(*layers.back(), 3)); // win, draw, lose
        layers.emplace_back(new LogSoftMaxLayer(*layers.back()));
        Layer& value_layer = *layers.back();

        NLLLoss loss_policy(net);
        NLLLoss loss_value(net);
        net.Build({ &policy_layer, &value_layer }, 0.01, { &loss_policy, &loss_value });
        return net;
    }
    NeuralStrategy(int strength = 100, dnnl::engine::kind eng_kind = dnnl::engine::kind::cpu)
        : eng(eng_kind, 0),
        strength(strength),
        train_net(Build(eng, dnnl::prop_kind::forward_training, batch_size)),
        net(Build(eng, dnnl::prop_kind::forward_inference, 1))
    {
    }
    virtual void Train(const MoveStorage& storage) override
    {
        if (storage.empty())
            return;

        for (int i = 0; i < train_net.weights.size(); i++)
        {
            std::vector<float> tmp;
            net.weights[i] >> tmp;
            train_net.weights[i] << tmp;
        }

        dnnl::stream s(train_net.engine);
        int batch_count = storage.size / batch_size;
        for (int i = 0; i < batch_count; i++)
        {
            float* input = (float*)MemoryMapData(train_net.input);
            float* answer = (float*)MemoryMapData(train_net.answer(0));
            float* answer_value = (float*)MemoryMapData(train_net.answer(1));
            float* input_pos = input;
            float* answer_pos = answer;
            float* answer_value_pos = answer_value;
            std::uniform_int_distribution<int> dist(0, storage.size - 1);
            for (int i = 0; i < batch_size; i++)
            {
                auto& cur = storage[dist(gen)];
                input_pos = std::copy(cur.position.begin(), cur.position.end(), input_pos);
                answer_pos = std::copy(cur.policy.begin(), cur.policy.end(), answer_pos);
                *answer_value_pos++ = cur.value > 0;
                *answer_value_pos++ = cur.value == 0;
                *answer_value_pos++ = cur.value < 0;
            }
            MemoryUnmapData(train_net.input, input);
            MemoryUnmapData(train_net.answer(0), answer);
            MemoryUnmapData(train_net.answer(1), answer_value);
            train_net.Forward(s);
            train_net.Backward(s);
            s.wait();
        }
        for (int i = 0; i < train_net.weights.size(); i++)
        {
            std::vector<float> tmp;
            train_net.weights[i] >> tmp;
            net.weights[i] << tmp;
        }
    }
    virtual void Randomize(int seed) override
    {
        gen = std::mt19937(seed);
        net.RandomWeights(gen);
    }
    virtual void MakeMove(Field& field, MoveStorage* storage = nullptr, bool train = false) override
    {
        MctsNode node;
        Move move = Mcts(field, node, &net, gen, strength, train);
        // Save predictions
        if (storage)
        {
            std::vector<float> i_v(6 * FIELD_HEIGHT * FIELD_WIDTH), p_v(FIELD_HEIGHT * FIELD_WIDTH, 0);
            FieldToNNInput(field, i_v.data());
            MctsNode** cur_child = &node.child;
            for (MctsNode** cur_child = &node.child; *cur_child; cur_child = &(*cur_child)->sibling)
            {
                p_v[field.ToY((*cur_child)->move) * field.width + field.ToX((*cur_child)->move)] = (*cur_child)->visits / (float)node.visits;
            }
            storage->push({ i_v, p_v, node.value });
        }
        field.MakeMove(move);
    }
};

struct MctsStrategy : Strategy
{
    std::mt19937 gen;
    int strength;
    MctsStrategy(int strength) : strength(strength) {}
    virtual void Randomize(int seed) override
    {
        gen = std::mt19937(seed);
    }
    virtual void MakeMove(Field& field, MoveStorage* storage = nullptr, bool train = false) override
    {
        //field.MakeMove(Uct(field, gen, strength));
        MctsNode node;
        Move move = Mcts(field, node, nullptr, gen, strength, train);
        // Save predictions
        if (storage)
        {
            std::vector<float> i_v(6 * FIELD_HEIGHT * FIELD_WIDTH), p_v(FIELD_HEIGHT * FIELD_WIDTH, 0);
            FieldToNNInput(field, i_v.data());
            MctsNode** cur_child = &node.child;
            for (MctsNode** cur_child = &node.child; *cur_child; cur_child = &(*cur_child)->sibling)
            {
                p_v[field.ToY((*cur_child)->move) * field.width + field.ToX((*cur_child)->move)] = (*cur_child)->visits / (float)node.visits;
            }
            storage->push({ i_v, p_v, node.value });
        }
        field.MakeMove(move);
    }
};

struct RandomStrategy : Strategy
{
    std::mt19937 gen;
    virtual void Randomize(int seed) override
    {
        gen = std::mt19937(seed);
    }
    virtual void MakeMove(Field& field, MoveStorage* storage = nullptr, bool train = false) override
    {
        MoveList moves = field.GetAllMoves();
        std::uniform_int_distribution<int> dist(0, moves.size() - 1);
        field.MakeMove(moves[dist(gen)]);
    }
};

struct HumanStrategy : Strategy
{
    virtual void MakeMove(Field& field, MoveStorage* storage = nullptr, bool train = false) override
    {
        field.DebugPrint(std::cout, std::to_string(field.GetScore(field.GetPlayer())), true, true);
        std::cout << "Move: ";
        Move move;
        do
        {
            short x, y;
            std::cin >> x >> y;
            move = field.ToMove(x, y);
        } while (!field.CouldMove(move));
        field.MakeMove(move);
    }
};

// Is first strategy win
bool PvP(Strategy& a, Strategy& b, Player first, MoveStorage* storage = nullptr, bool train = false);

void Trainer();
