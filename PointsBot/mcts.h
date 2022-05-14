#pragma once
#include "field.h"
#include "layers.h"
#include <random>

constexpr int FIELD_HEIGHT = 6;
constexpr int FIELD_WIDTH = 6;

constexpr float kUct = 1;

struct MctsNode
{
    MctsNode* sibling = nullptr, * child = nullptr;
    float value = 0;
    int visits = 0;
    float prior_prob = 0;
    Move move = -1;
    // Mean action value
    float GetQ() const
    {
        return value / std::max(1, visits);
    }
    float GetP() const { return prior_prob; }
    float GetU(int total) const
    {
        return kUct * prior_prob * sqrtf(total) / (1 + visits);
    }
    MctsNode() {}
    MctsNode(Move move, float prior_prob) : move(move), prior_prob(prior_prob) {}
    ~MctsNode()
    {
        delete sibling;
        delete child;
    }
};

struct Prediction
{
    std::vector<float> policy;
    float value = 0;
    short height, width;
    size_t MoveToIndex(Move move) const
    {
        return Field::ToY(move, width)* width + Field::ToX(move, width);
    }
    Prediction(int height, int width, int size) : height(height), width(width), policy(size) {}
};

void FieldToNNInput(Field& field, void *input_) //, bool flip_h = false, bool flip_v = false, bool transpose = false)
{
    // input_[6][height][width]
    //float* input = (float*)input_;
    //short height = field.height, width = field.width;
    //auto get_pos =
    //    flip_h
    //    ? [height, width](int c, int i, int j) { return c * height * width + i * width + width - j - 1; }
    //    : [height, width](int c, int i, int j) { return c * height * width + i * width + j; };
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

float MctsRandomGame(Field& field, std::mt19937& gen, MoveList moves)
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

    //result = std::max(-1.f, std::min(1.f, field.GetScore(player) / 10.f));
    result = std::max(-1, std::min(1, field.GetScore(player)));

    for (size_t i = 0; i < putted; i++)
        field.Undo();

    return result;
}

Prediction Predict(Field& field, const MoveList& moves, std::mt19937& gen, NNetwork* net = nullptr)
{
    Prediction result(field.height, field.width, moves.size());
    //std::uniform_real_distribution<float> dist(-1, 1);
    //result.value = dist(gen);
    //for (int i = 0; i < result.policy.size(); i++)
    //    result.policy[i] = dist(gen);
    if (net)
    {
        void* input = MemoryMapData(net->input);
        //int flip = gen() & 7;
        FieldToNNInput(field, input);//, flip & 1, flip & 2, flip & 4);
        MemoryUnmapData(net->input, input);
        dnnl::stream s(net->engine);
        net->Forward(s);
        s.wait();
        float* policy = (float*)MemoryMapData(net->output(0));
        // softmax on available moves
        float max_p = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < moves.size(); i++)
        {
            float p = policy[result.MoveToIndex(moves[i])];
            result.policy[i] = p;
            max_p = std::max(max_p, p);
        }
        float total = 0;
        for (int i = 0; i < moves.size(); i++)
        {
            float p = std::exp(result.policy[i] - max_p);
            result.policy[i] = p;
            total += p;
        }
        MemoryUnmapData(net->output(0), policy);
        std::vector<float> value;
        net->output(1) >> value;
        result.value = value[0] - value[2];
    }
    else
    {
        result.value = MctsRandomGame(field, gen, moves);
        for (int i = 0; i < moves.size(); i++)
        {
            result.policy[i] = 1.f / moves.size();
        //    field.MakeMove(move);
        //    result.policy[move] = MctsRandomGame(field, gen, moves);
        //    field.Undo();
        }
    }
    
    return result;
}

// Return score for field current player
float MctsSearch(Field& field, MctsNode* node, std::mt19937& gen, NNetwork* net = nullptr)
{
    if (!node->child)
    {
        // Leaf, Finish state
        if (field.GetAllMoves().empty())
        {
            node->visits++; // std::numeric_limits<int>::max();
            //float score = std::max(-1.f, std::min(1.f, field.GetScore(field.GetPlayer()) / 10.f));
            float score = std::max(-1, std::min(1, field.GetScore(field.GetPlayer())));
            node->value = score * node->visits;
            // No uncertainty, so U = 0
            node->prior_prob = 0;
            return score;
        }
        // Leaf, Expand
        MoveList moves = field.GetAllMoves();
        shuffle(moves.begin(), moves.end(), gen);
        Prediction prediction = Predict(field, moves, gen, net);
        MctsNode** cur_child = &node->child;
        std::uniform_real_distribution<float> dist(0, 0.01);
        for (float& x : prediction.policy)
            x += dist(gen);
        for (int i = 0; i < moves.size(); i++)
        {
            *cur_child = new MctsNode(moves[i], prediction.policy[i]);
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
        // Select best child (min value because value for enemy after move)
        float score = (*cur_child)->GetQ() - (*cur_child)->GetU(node->visits);
        if (score < best_score)
        {
            best_score = score;
            best = *cur_child;
        }
    }
    // Go to best child
    if (!best)
    {
        std::cerr << "Mcts don't found any moves!\n";
        exit(-123);
    }
    field.MakeMove(best->move);
    float result = -MctsSearch(field, best, gen, net);
    // Backpropogation
    node->value += result;
    node->visits++;
    field.Undo();
    return result;
}

Move Mcts(Field& field, MctsNode& root, std::mt19937& gen, NNetwork* net, int simulations, bool train = false)
{
    for (int i = 0; i < simulations; i++)
        MctsSearch(field, &root, gen, net);

    MctsNode* best = nullptr;
    if (!train)
    {
        float best_score = -1;
        std::uniform_real_distribution<float> dist(0, 0.1);
        for (MctsNode** cur_child = &root.child; *cur_child; cur_child = &(*cur_child)->sibling)
        {
            float score = (*cur_child)->visits + dist(gen);
            //std::cout << field.ToX((*cur_child)->move) << " " << field.ToY((*cur_child)->move) << " " << score << "\n";
            if (score > best_score)
            {
                best_score = score;
                best = *cur_child;
            }
        }
    }
    else
    {
        float t = 1.f / 1;
        std::uniform_real_distribution<float> dist(0, 1 - 1e-6);
        float x = dist(gen);
        float cur = 0;
        float sum = 0;
        for (MctsNode** cur_child = &root.child; *cur_child; cur_child = &(*cur_child)->sibling)
            sum += std::powf((*cur_child)->visits, t);
        for (MctsNode** cur_child = &root.child; *cur_child; cur_child = &(*cur_child)->sibling)
        {
            cur += std::powf((*cur_child)->visits, t) / sum;
            if (cur >= x)
            {
                best = *cur_child;
                break;
            }
        }
    }
    return best ? best->move : -1;
}

Move Mcts(Field& field, std::mt19937& gen, NNetwork* net, int simulations, bool train = false)
{
    MctsNode root;
    return Mcts(field, root, gen, net, simulations, train);
}
