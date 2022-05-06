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
    virtual void MakeMove(Field& field, MoveStorage* storage = nullptr) = 0;
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
    static dnnl::engine eng;
    NNetwork train_net, net;
    std::mt19937 gen;
    int strength;
    //static dnnl::engine eng;
    NNetwork Build(const dnnl::engine& eng, dnnl::prop_kind kind, int batch)
    {
        NNetwork net(eng, kind);
        //InputLayer in(net, { batch, 6 * FIELD_HEIGHT * FIELD_WIDTH });
        //DenseLayer dense1(in, 100);
        //ReluLayer relu(dense1);
        //DenseLayer dense_policy(relu, 100);
        //SigmoidLayer sigm(dense_policy);
        //DenseLayer dense_value(relu, 1);
        //SigmoidLayer sigm2(dense_value);
        //net.Build({ &sigm, &sigm2 }, 1e-3);

        InputLayer in(net, { batch, 6, FIELD_HEIGHT, FIELD_WIDTH });
        constexpr int kFilters = 16;
        ConvLayer conv(in, { kFilters, 6, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 });

        ConvLayer conv1_1(conv, { kFilters, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 });
        ConvLayer conv1_2(conv1_1, { kFilters, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 });
        SELayer se1(conv1_2, kFilters);
        LayerAdder add1(conv, se1);
        ReluLayer relu1(add1);

        ConvLayer conv2_1(relu1, { kFilters, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 });
        ConvLayer conv2_2(conv2_1, { kFilters, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 });
        SELayer se2(conv2_2, kFilters);
        LayerAdder add2(relu1, se2);
        ReluLayer relu2(add2);

        ConvLayer conv_last(relu2, { kFilters, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 });
        ConvLayer conv_last_2(conv_last, { 1, kFilters, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 });
        SigmoidLayer sigm(conv_last_2);

        DenseLayer dense_value(relu2, 128);
        DenseLayer dense_value_2(relu2, 1);
        EltwiseLayer elt(dense_value_2, dnnl::algorithm::eltwise_tanh);

        net.Build({ &sigm, &elt }, 1e-3);
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
        }
        for (int i = 0; i < train_net.weights.size(); i++)
        {
            std::vector<float> tmp;
            //if (GetMemoryByteSize(train_net.weights[i]) != GetMemoryByteSize(net.weights[i]))
            //{
            //    std::cout << "EPIC FAIL\n";
            //}
            train_net.weights[i] >> tmp;
            net.weights[i] << tmp;
        }
    }
    virtual void Randomize(int seed) override
    {
        gen = std::mt19937(seed);
        net.RandomWeights(gen);
    }
    virtual void MakeMove(Field& field, MoveStorage* storage = nullptr) override
    {
        MctsNode node;
        Move move = Mcts(field, node, net, gen, 100);
        // Save predictions
        if (storage)
        {
            std::vector<float> i_v(6 * FIELD_HEIGHT * FIELD_WIDTH), p_v(FIELD_HEIGHT * FIELD_WIDTH, 0);
            FieldToNNInput(field, i_v.data());
            MctsNode** cur_child = &node.child;
            for (MctsNode** cur_child = &node.child; *cur_child; cur_child = &(*cur_child)->sibling)
            {
                p_v[field.ToY((*cur_child)->move) * field.width + field.ToX((*cur_child)->move)] = (*cur_child)->value;
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
    virtual void MakeMove(Field& field, MoveStorage* storage = nullptr) override
    {
        field.MakeMove(Uct(field, gen, strength));
    }
};

struct RandomStrategy : Strategy
{
    std::mt19937 gen;
    virtual void Randomize(int seed) override
    {
        gen = std::mt19937(seed);
    }
    virtual void MakeMove(Field& field, MoveStorage* storage = nullptr) override
    {
        MoveList moves = field.GetAllMoves();
        std::uniform_int_distribution<int> dist(0, moves.size() - 1);
        field.MakeMove(moves[dist(gen)]);
    }
};

// Is first strategy win
bool PvP(Strategy& a, Strategy& b, Player first, MoveStorage* storage = nullptr);

void Trainer();
