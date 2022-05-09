#pragma once
#include "field.h"
#include "layers.h"
#include "uct.h"
#include <typeinfo>
#include <algorithm>
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
    inline void push(const MoveRecord& x)
    {
        q[r++] = x;
        size = std::max(size, r);
        if (r >= capacity)
            r = 0;
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
    const int batch_size = 64;
    static dnnl::engine eng;
    NNetwork train_net, net;
    std::mt19937 gen;
    int strength;
    //static dnnl::engine eng;
    NNetwork Build(const dnnl::engine& eng, dnnl::prop_kind kind, int batch)
    {
        NNetwork net(eng, kind);
        InputLayer in(net, { batch, 6 * FIELD_HEIGHT * FIELD_WIDTH });
        DenseLayer dense(in, 12 * FIELD_HEIGHT * FIELD_WIDTH);
        ReluLayer relu(dense);
        DenseLayer dense_policy(relu, FIELD_HEIGHT * FIELD_WIDTH);
        SigmoidLayer sigm(dense_policy);
        DenseLayer dense_value(relu, 1);
        TanhLayer tanh(dense_value);
        CrossEntropyLoss loss_policy(net);
        MeanSquaredLoss loss_value(net);
        net.Build({ &sigm, &tanh }, 0.1, { &loss_policy, &loss_value});

        //constexpr int kFilters = 16;
        //vector<unique_ptr<Layer>> layers;
        //layers.emplace_back(new InputLayer(net, { batch, 6, FIELD_HEIGHT, FIELD_WIDTH }));
        //layers.emplace_back(new ConvLayer(*layers.back(), { kFilters, 6, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }));
        //layers.emplace_back(new BatchNormLayer(*layers.back()));
        ////layers.emplace_back(new TanhLayer(*layers.back())); // TODO: del

        //for (int i = 0; i < 8; i++)
        //{
        //    Layer& residual = *layers.back();
        //    layers.emplace_back(new ConvLayer(residual, { kFilters, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }));
        //    //layers.emplace_back(new TanhLayer(*layers.back())); // TODO: del
        //    layers.emplace_back(new BatchNormLayer(*layers.back()));
        //    layers.emplace_back(new ConvLayer(*layers.back(), { kFilters, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }));
        //    //layers.emplace_back(new TanhLayer(*layers.back())); // TODO: del
        //    layers.emplace_back(new BatchNormLayer(*layers.back()));
        //    layers.emplace_back(new SELayer(*layers.back(), kFilters));
        //    layers.emplace_back(new LayerAdder(residual, *layers.back()));
        //    layers.emplace_back(new ReluLayer(*layers.back(), 0.3));
        //    //layers.emplace_back(new TanhLayer(*layers.back()));
        //}
        //Layer& tower_last = *layers.back();
        //layers.emplace_back(new ConvLayer(tower_last, { 2, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }));
        ////layers.emplace_back(new TanhLayer(*layers.back())); // TODO: del
        //layers.emplace_back(new BatchNormLayer(*layers.back()));
        //layers.emplace_back(new ReluLayer(*layers.back(), 0.3));
        //layers.emplace_back(new DenseLayer(*layers.back(), FIELD_HEIGHT * FIELD_WIDTH));
        ////layers.emplace_back(new TanhLayer(*layers.back())); // TODO: del
        //Layer& policy_layer = *layers.back();

        //layers.emplace_back(new ConvLayer(tower_last, { 1, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }));
        ////layers.emplace_back(new TanhLayer(*layers.back())); // TODO: del
        //layers.emplace_back(new BatchNormLayer(*layers.back()));
        //layers.emplace_back(new ReluLayer(*layers.back(), 0.3));
        //layers.emplace_back(new DenseLayer(*layers.back(), 1));
        ////layers.emplace_back(new TanhLayer(*layers.back())); // TODO: del
        //Layer& value_layer = *layers.back();

        //net.Build({ &policy_layer, &value_layer }, 1e-3);
        return net;
    }
    NeuralStrategy(int strength = 100)
        : //eng(dnnl::engine::kind::cpu, 0),
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
                *answer_value_pos++ = cur.value;
            }
            //if (input_pos - input > GetMemoryCount(train_net.input) || answer_pos - answer > GetMemoryCount(train_net.answer))
            //{
            //    std::cout << input_pos - input << " " << GetMemoryCount(train_net.input) << "\n";
            //    std::cout << answer_pos - answer << " " << GetMemoryCount(train_net.answer) << "\n";
            //    std::cout << "BAD FAIL\n";
            //}
            MemoryUnmapData(train_net.input, input);
            MemoryUnmapData(train_net.answer(0), answer);
            MemoryUnmapData(train_net.answer(1), answer_value);
            train_net.Forward(s);
            train_net.Backward(s);
            s.wait();
            //std::vector<float> tmp2;
            //train_net.grad(0) >> tmp2;
            //float max_value_2 = -123;
            //for (int j = 0; j < tmp2.size(); j++)
            //    max_value_2 = std::max(max_value_2, std::abs(tmp2[j]));
            //std::cout << max_value_2 << "\n";
            //for (int i = 0; i < train_net.weights.size(); i++)
            //{
            //    std::vector<float> tmp;
            //    std::vector<float> tmp2;
            //    train_net.weights[i] >> tmp;
            //    net.weights[i] >> tmp2;
            //    for (int j = 0; j < tmp.size(); j++)
            //        max_value_2 = std::max(max_value_2, std::abs(tmp[j] - tmp2[j]));
            //}
            //std::cout << max_value_2 << "\n";
        }
        //float max_value = -123;
        for (int i = 0; i < train_net.weights.size(); i++)
        {
            std::vector<float> tmp;
            //std::vector<float> tmp2;
            //net.weights[i] >> tmp2;
            //if (GetMemoryByteSize(train_net.weights[i]) != GetMemoryByteSize(net.weights[i]))
            //{
            //    std::cout << "EPIC FAIL\n";
            //}
            train_net.weights[i] >> tmp;
            net.weights[i] << tmp;

            //assert(tmp.size() == tmp2.size());
            //for (int j = 0; j < tmp.size(); j++)
            //    max_value = std::max(max_value, std::abs(tmp[j] - tmp2[j]));
        }
        //std::cout << max_value << "\n";
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

dnnl::engine NeuralStrategy::eng(dnnl::engine::kind::cpu, 0);

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

// Is first strategy win
bool PvP(Strategy& a, Strategy& b, Player first, MoveStorage* storage = nullptr, bool train = false);

void Trainer();
