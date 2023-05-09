#pragma once

#include "field.h"
#include <random>
#include <vector>

//template<class A, class B>
//struct std::hash<std::pair<A, B>>
//{
//	size_t operator() (const pair<A, B>& p) const
//	{
//		return std::hash<A>()(p.first) ^ std::hash<B>()(p.second);
//	}
//};

enum PointType : short
{
    kPointRedOrWhite = 0,
    kPointRed,
    kPointWhite,
    kPointBlack,
    kPointBlackOrWhite,
};

struct PointDelta
{
    short dx, dy;
    PointType type;
    PointDelta(short dx, short dy, PointType type) : dx(dx), dy(dy), type(type) {}

    inline void InvertType()
    {
        type = (PointType)(4 - type);
    }
    inline void FlipLR()
    {
        dx *= -1;
    }
    inline void Transpose()
    {
        std::swap(dx, dy);
    }
    inline void Rot90()
    {
        Transpose();
        FlipLR();
    }
    inline int FromXY(int width, int offset = 0) const
    {
        return dx + dy * width + offset;
    }
};

struct HeuristicPattern
{
    std::vector<PointDelta> points;
    double score;
    HeuristicPattern(const std::vector<PointDelta>& points, double score) : points(points), score(score) {}
    inline void InvertType()
    {
        for (auto& point : points)
            point.InvertType();
    }
    inline void FlipLR()
    {
        for (auto& point : points)
            point.FlipLR();
    }
    inline void Transpose()
    {
        for (auto& point : points)
            point.Transpose();
    }
    inline void Rot90()
    {
        for (auto& point : points)
            point.Rot90();
    }
    //std::pair<uint64_t, uint64_t> Hash() const
    //{
    //    std::pair<uint64_t, uint64_t> result{ 0, 0 };
    //    for (const auto& point : points)
    //    {
    //        if (point.type != kPointWhite)
    //            (point.type == kPointRed ? result.first : result.second) |= 1ll << point.FromXY(5, 12);
    //    }
    //    return result;
    //}
};

class HeuristicStrategy
{
	//std::unordered_map<std::pair<uint64_t, uint64_t>, double> score_table_;
	std::vector<HeuristicPattern> all_patterns_;
public:
	HeuristicStrategy();
	Move MakeMove(Field& field, std::mt19937& gen);
};