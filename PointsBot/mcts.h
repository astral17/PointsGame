#pragma once
#include "field.h"
#include "layers.h"
#include <random>

constexpr int FIELD_HEIGHT = 10;
constexpr int FIELD_WIDTH = 10;

struct MctsNode
{
    MctsNode* sibling = nullptr, * child = nullptr;
    float value = 0;
    int visits = 1;
    Move move = -1;
    float GetP() const
    {
        return value / visits;
    }
    float GetU(int total) const
    {
        return sqrtf(2 * logf(total) / visits);
    }
    MctsNode() {}
    MctsNode(Move move, float value) : move(move), value(value) {}
    ~MctsNode()
    {
        delete sibling;
        delete child;
    }
};

struct Policy
{
    std::vector<float> policy;
    int height, width;
    float& operator[](Move move)
    {
        return policy[Field::ToY(move, width) * width + Field::ToX(move, width)];
    }
    size_t size() const { return policy.size(); }
    Policy(int height, int width) : height(height), width(width), policy(height * width) {}
};

struct Prediction
{
    float value = 0;
    Policy policy;
    Prediction(int height, int width) : policy(height, width) {}
};

void FieldToNNInput(Field& field, void *input_)
{
    float(*input)[FIELD_HEIGHT][FIELD_WIDTH] = (float(*)[FIELD_HEIGHT][FIELD_WIDTH])input_;
    for (int i = 0; i < field.height; i++)
        for (int j = 0; j < field.width; j++)
        {
            // 0 - red point, 1 - black point
            // 2 - red area, 3 - black area
            // 4 - red empty, 5 - black empty
            input[0][i][j] = field.IsRedPoint(field.ToMove(j, i));
            input[1][i][j] = field.IsBlackPoint(field.ToMove(j, i));
            if (field.GetPlayer() != kPlayerRed)
                std::swap(input[0][i][j], input[1][i][j]);

            input[2][i][j] = field.IsRedArea(field.ToMove(j, i));
            input[3][i][j] = field.IsBlackArea(field.ToMove(j, i));
            if (field.GetPlayer() != kPlayerRed)
                std::swap(input[2][i][j], input[3][i][j]);

            input[4][i][j] = field.IsRedEmpty(field.ToMove(j, i));
            input[5][i][j] = field.IsBlackEmpty(field.ToMove(j, i));
            if (field.GetPlayer() != kPlayerRed)
                std::swap(input[4][i][j], input[5][i][j]);
        }
}

#define ORDER_BY_SCORE

float MctsRandomGame(Field& field, mt19937& gen, MoveList moves)
{
    Player player = field.GetPlayer();
    size_t putted = 0;
    float result;
    shuffle(moves.begin(), moves.end(), gen);
    for (auto i = moves.begin(); i < moves.end(); i++)
        if (field.CouldMove(*i))
        {
            field.MakeMove(*i);
            putted++;
        }

    result = std::max(-1.f, std::min(1.f, field.GetScore(player) / 10.f));

    for (size_t i = 0; i < putted; i++)
        field.Undo();

    return result;
}

Prediction Predict(Field& field, NNetwork& net, std::mt19937& gen)
{
    Prediction result(field.height, field.width);
    //std::uniform_real_distribution<float> dist(-1, 1);
    //result.value = dist(gen);
    //for (int i = 0; i < result.policy.size(); i++)
    //    result.policy[i] = dist(gen);
    
    void* input = MemoryMapData(net.input);
    FieldToNNInput(field, input);
    MemoryUnmapData(net.input, input);
    dnnl::stream s(net.engine);
    net.Forward(s);
    s.wait();
    net.output(0) >> result.policy.policy;
    //if (isnan(result.policy.policy[0]))
    //{
    //    for (int i = 0; i < net.weights.size(); i++)
    //    {
    //        vector<float> tmp;
    //        net.weights[i] >> tmp;
    //        std::cout << "hmm";
    //    }
    //}
    std::vector<float> value;
    net.output(1) >> value;
    result.value = value[0];
    
    //MoveList moves = field.GetAllMoves();
    //result.value = MctsRandomGame(field, gen, moves);
    //for (Move move : moves)
    //{
    //    field.MakeMove(move);
    //    result.policy[move] = MctsRandomGame(field, gen, moves);
    //    field.Undo();
    //}

    
    return result;
}

// Return score for field current player
float MctsSearch(Field& field, MctsNode* node, NNetwork& net, std::mt19937& gen)
{
    if (!node->child)
    {
        // Leaf, Finish state
        if (field.GetAllMoves().empty())
        {
            node->visits = std::numeric_limits<int>::max();
            return node->value = std::max(-1.f, std::min(1.f, field.GetScore(field.GetPlayer()) / 10.f));
        }
        // Leaf, Expand
        Prediction prediction = Predict(field, net, gen);
        MctsNode** cur_child = &node->child;
        MoveList moves = field.GetAllMoves();
        shuffle(moves.begin(), moves.end(), gen);
        for (Move move : moves)
        {
            *cur_child = new MctsNode(move, prediction.policy[move]);
            cur_child = &(*cur_child)->sibling;
        }
        // Backpropogation
        node->value += prediction.value;
        node->visits++;
        return prediction.value;
    }
    MctsNode* best = nullptr;
    float best_score = std::numeric_limits<float>::max();
    for (MctsNode** cur_child = &node->child; *cur_child; cur_child = &(*cur_child)->sibling)
    {
        // Select best child (min value because policy for enemy after move)
        float score = (*cur_child)->GetP() - (*cur_child)->GetU(node->visits);
        if (score < best_score)
        {
            best_score = score;
            best = *cur_child;
        }
    }
    // Go to best child
    field.MakeMove(best->move);
    float result = -MctsSearch(field, best, net, gen);
    // Backpropogation
    node->value += result;
    node->visits++;
    field.Undo();
    return result;
}

Move Mcts(Field& field, MctsNode& root, NNetwork& net, std::mt19937& gen, int simulations)
{
    for (int i = 0; i < simulations; i++)
        MctsSearch(field, &root, net, gen);

    MctsNode* best = nullptr;
    float best_score = std::numeric_limits<float>::max();
    for (MctsNode** cur_child = &root.child; *cur_child; cur_child = &(*cur_child)->sibling)
    {
        // Select best child (min value because policy for enemy after move)
        float score = (*cur_child)->GetP();// -(*cur_child)->GetU(root.visits);
        //std::cout << field.ToX((*cur_child)->move) << " " << field.ToY((*cur_child)->move) << " " << score << "\n";
        if (score < best_score)
        {
            best_score = score;
            best = *cur_child;
        }
    }
    return best ? best->move : -1;
}

Move Mcts(Field& field, NNetwork& net, std::mt19937& gen, int simulations)
{
    MctsNode root;
    return Mcts(field, root, net, gen, simulations);
}
