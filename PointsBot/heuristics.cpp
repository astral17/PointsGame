#include "heuristics.h"
#include <vector>

//struct PointDelta
//{
//    short dx, dy;
//    PointType type;
//    PointDelta(short dx, short dy, PointType type) : dx(dx), dy(dy), type(type) {}
//
//    inline void InvertType()
//    {
//        type = (PointType)(4 - type);
//    }
//    inline void FlipLR()
//    {
//        dx *= -1;
//    }
//    inline void Transpose()
//    {
//        std::swap(dx, dy);
//    }
//    inline void Rot90()
//    {
//        Transpose();
//        FlipLR();
//    }
//    inline int FromXY(int width, int offset = 0) const
//    {
//        return dx + dy * width + offset;
//    }
//};

int FromDeltaXY(int dx, int dy, int width, int offset)
{
    return dx + dy * width + offset;
}

//struct HeuristicPattern
//{
//    std::vector<PointDelta> points;
//    double score;
//    HeuristicPattern(const std::vector<PointDelta>& points, double score) : points(points), score(score) {}
//    inline void InvertType()
//    {
//        for (auto& point : points)
//            point.InvertType();
//    }
//    inline void FlipLR()
//    {
//        for (auto& point : points)
//            point.FlipLR();
//    }
//    inline void Transpose()
//    {
//        for (auto& point : points)
//            point.Transpose();
//    }
//    inline void Rot90()
//    {
//        for (auto& point : points)
//            point.Rot90();
//    }
//    std::pair<uint64_t, uint64_t> Hash() const
//    {
//        std::pair<uint64_t, uint64_t> result {0, 0};
//        for (const auto& point : points)
//        {
//            if (point.type != kPointWhite)
//                (point.type == kPointRed ? result.first : result.second) |= 1ll << point.FromXY(5, 12);
//        }
//        return result;
//    }
//};

const std::vector<HeuristicPattern> kBasicPatterns =
{
    {{
        {-1, 0, kPointRed},
        { 1, 0, kPointRed},
        {0, -1, kPointBlack},
        {0,  1, kPointBlack},
    }, 1.0},
    {{
        {-1, -1, kPointBlack},
        { 0, -1, kPointRedOrWhite},
        {-1,  0, kPointRed},
        { 1,  0, kPointRedOrWhite},
        { 0,  1, kPointBlack},
    }, 0.9},
    {{
        { 0, -1, kPointBlack},
        { 1, -1, kPointRedOrWhite},
        {-2,  0, kPointRed},
        {-1,  0, kPointRed},
        {-1,  1, kPointBlack},
        { 0,  1, kPointRedOrWhite},
    }, 0.9},
    {{
        {-1, -1, kPointBlack},
        { 1, -1, kPointRedOrWhite},
        {-1,  1, kPointRedOrWhite},
        { 1,  1, kPointBlack},
    }, 0.05},
    {{
        {-1, -1, kPointWhite},
        { 0, -1, kPointWhite},
        { 1, -1, kPointBlack},
        { 1,  0, kPointWhite},
        { 1,  1, kPointWhite},
        { 0, -2, kPointRed},
        { 2,  0, kPointRed},
    }, 0.8},

    {{
        {-1, -1, kPointRed},
        { 0, -1, kPointBlack},
        { 1, -1, kPointWhite},
        {-1,  0, kPointRedOrWhite},
        { 1,  0, kPointRedOrWhite},
        { 0,  1, kPointBlack},
        { 0, -2, kPointRed},
    }, 0.7},
    {{
        { 0, -2, kPointRed},
        {-1, -1, kPointRed},
        { 0, -1, kPointBlack},
        { 1, -1, kPointWhite},
    }, 0.2},
    {{
        {-1, -1, kPointWhite},
        {-1,  0, kPointBlack},
        {-1,  1, kPointRed},
        { 0,  1, kPointBlack},
        { 1,  1, kPointWhite},
        {-2,  0, kPointRed},
        { 0,  2, kPointRed},
    }, 1.0},
    {{
        {-1, -1, kPointRed},
        {-1,  0, kPointBlack},
        {-1,  1, kPointRed},
        {-2,  0, kPointRedOrWhite},
    }, 0.2},
    {{
        { 0, -1, kPointBlack},
        {-1,  0, kPointRedOrWhite},
        { 1,  0, kPointRedOrWhite},
        { 0,  1, kPointBlack},
    }, 0.05},

    {{
        {-2,  0, kPointRedOrWhite},
        {-1,  0, kPointRedOrWhite},
        {-2,  1, kPointWhite},
        {-1,  1, kPointWhite},
        { 0,  1, kPointWhite},
        {-2,  2, kPointRedOrWhite},
        {-1,  2, kPointBlackOrWhite},
        { 0,  2, kPointRedOrWhite},
        {-1,  3, kPointBlackOrWhite},
    }, 0.01},
    {{
        {-1, -1, kPointRedOrWhite},
        {-2,  0, kPointRed},
        {-1,  0, kPointWhite},
        {-2,  1, kPointRedOrWhite},
        {-1,  1, kPointBlack},
        { 0,  1, kPointWhite},
        {-1,  2, kPointBlack},
    }, 0.05},
    {{
        { 0, -1, kPointRed},
        {-1,  0, kPointRed},
        { 1,  0, kPointRed},
    }, -0.5},
    {{
        { 0, -2, kPointBlack},
        { 0, -1, kPointWhite},
        {-1,  0, kPointRedOrWhite},
        { 1,  0, kPointRedOrWhite},
        { 0,  1, kPointBlack},
    }, 0.05},
    {{
        { 0, -2, kPointBlack},
        { 0, -1, kPointWhite},
        {-1,  0, kPointRedOrWhite},
        { 1,  0, kPointRedOrWhite},
        { 0,  1, kPointWhite},
        { 0,  2, kPointBlack},
    }, 0.05},

    {{
        {-1,  0, kPointBlack},
        {-1,  1, kPointBlack},
        { 0,  1, kPointBlack},
    }, -0.5},
    {{
        { 0, -1, kPointRed},
        { 1, -1, kPointBlack},
        { 1,  0, kPointRed},
        { 0,  1, kPointBlack},
    }, 0.9},
};

//void GeneratePatterns(std::vector<HeuristicPattern>& patterns, HeuristicPattern &pattern, int pos = 0)
//{
//    if (pos >= pattern.points.size())
//    {
//        patterns.push_back(pattern);
//        return;
//    }
//    if (pattern.points[pos].type == kPointBlackOrWhite || pattern.points[pos].type == kPointRedOrWhite)
//    {
//        PointType old = pattern.points[pos].type;
//        pattern.points[pos].type = (PointType)((pattern.points[pos].type + kPointWhite) / 2);
//        GeneratePatterns(patterns, pattern, pos + 1);
//        pattern.points[pos].type = kPointWhite;
//        GeneratePatterns(patterns, pattern, pos + 1);
//        pattern.points[pos].type = old;
//        return;
//    }
//    GeneratePatterns(patterns, pattern, pos + 1);
//}

HeuristicStrategy::HeuristicStrategy()
{
    for (HeuristicPattern pattern : kBasicPatterns)
    {
        for (int i = 0; i < 2; i++)
        {
            all_patterns_.push_back(pattern);
            for (int i = 0; i < 3; i++)
            {
                pattern.Rot90();
                all_patterns_.push_back(pattern);
            }
            pattern.FlipLR();
            all_patterns_.push_back(pattern);
            for (int i = 0; i < 3; i++)
            {
                pattern.Rot90();
                all_patterns_.push_back(pattern);
            }
            if (i == 0)
                pattern.InvertType();
        }
    }
}

Move HeuristicStrategy::MakeMove(Field& field, std::mt19937& gen)
{
    std::vector<double> probs;
    std::vector<Move> moves;
    probs.reserve(field.height * field.width);
    moves.reserve(field.height * field.width);
    for (int j = 0; j < field.height; j++)
        for (int i = 0; i < field.width; i++)
        {
            Move move = field.ToMove(i, j);
            moves.push_back(move);
            if (!field.CouldMove(move))
            {
                probs.push_back(0);
                continue;
            }
            double score = 0;
            for (const auto& pattern : all_patterns_)
            {
                bool match = true;
                for (const auto& point : pattern.points)
                {
                    if (!field.IsValid(i + point.dx, j + point.dy))
                    {
                        match = false;
                        break;
                    }
                    move = field.ToMove(i + point.dx, j + point.dy);
                    switch (point.type)
                    {
                    case kPointRedOrWhite:
                        if (field.CouldMove(move))
                            break;
                        [[fallthrough]];
                    case kPointRed:
                        if (field.IsRedPoint(move) || field.IsRedArea(move))
                            break;
                        match = false;
                        break;
                    case kPointWhite:
                        if (field.CouldMove(move))
                            break;
                        match = false;
                        break;
                    case kPointBlackOrWhite:
                        if (field.CouldMove(move))
                            break;
                        [[fallthrough]];
                    case kPointBlack:
                        if (field.IsBlackPoint(move) || field.IsBlackArea(move))
                            break;
                        match = false;
                        break;
                    }
                    if (!match)
                        break;
                }
                if (match)
                {
                    score = pattern.score;
                    break;
                }
            }
            probs.push_back(std::exp(score * 100));
            //probs.push_back(std::max(0.0001, score));

            //std::pair<uint64_t, uint64_t> hash{ 0, 0 };
            //for (int dj = -2; dj <= 2; dj++)
            //    for (int di = -2; di <= 2; di++)
            //    {
            //        if (!field.IsValid(i + di, j + dj))
            //        {
            //            hash.first |= 1ll << FromDeltaXY(di, dj, 5, 12);
            //            hash.second |= 1ll << FromDeltaXY(di, dj, 5, 12);
            //            continue;
            //        }
            //        Move move = field.ToMove(i + di, j + dj);
            //        if (field.IsPoint(move))
            //        {
            //            (field.IsRedPoint(move) ? hash.first : hash.second) |= 1ll << FromDeltaXY(di, dj, 5, 12);
            //        }
            //    }
            //double score = (score_table_.count(hash) ? score_table_[hash] : -1);
        }
    std::discrete_distribution<> d(probs.begin(), probs.end());
    //int r = d(gen);
    //std::cout << std::log(probs[r]) << "\n";
    //std::cout << probs[r] << "\n";
    //return moves[r];
    return moves[d(gen)];
}
