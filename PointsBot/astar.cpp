#include "field.h"
#include <tuple>

AStar::AStar(Field& field)
	: cur_stk((field.width + 2) * (field.height + 2)),
	next_stk((field.width + 2) * (field.height + 2)),
#ifdef ASTAR_EXTENDED_FEATURE
	distRed(new short[(field.width + 2) * (field.height + 2)]),
	distBlack(new short[(field.width + 2) * (field.height + 2)]),
#else
    dist(new short[(field.width + 2) * (field.height + 2)]),
#endif
	field(field)
{
    for (Move i = field.MinMove(); i <= field.MaxMove(); i++)
#ifdef ASTAR_EXTENDED_FEATURE
        distRed[i] = distBlack[i]
#else
    dist[i]
#endif
            = std::min(std::min<short>(field.ToX(i), field.width - field.ToX(i)), std::min<short>(field.ToY(i), field.width - field.ToY(i)));
}

void AStar::BuildDistance()
{
#ifdef ASTAR_EXTENDED_FEATURE
    struct BfsState
    {
        Move move;
        short dist;
        BfsState() {}
        BfsState(Move move, short dist) : move(move), dist(dist) {}
    };
    FastQueue<BfsState> q((field.width + 2) * (field.height + 2));
    for (auto [dist, player] : { std::make_tuple(distRed.get(), kPlayerRed), std::make_tuple(distBlack.get(), kPlayerBlack) })
    {
        q.clear();
        //for (Move move : border)
        //{
        //    field.field[move] |= Field::Masks::kVisitedBit;
        //    q.emplace(move, 0);
        //}
        for (int i = 0; i < field.width; i++)
        {
            Move move = field.ToMove(i, 0);
            field.field[move] |= Field::Masks::kVisitedBit;
            q.emplace(move, 0);
        }
        for (int i = 1; i < field.height - 1; i++)
        {
            Move move = field.ToMove(0, i);
            field.field[move] |= Field::Masks::kVisitedBit;
            q.emplace(move, 0);
        }
        //for (int i = 0; i < field.width; i++)
        //    for (int j = 0; j < field.height; j++)
        //    {
        //        Move move = field.ToMove(i, j);
        //        if (field.IsBorder(field.Up(move))
        //            || field.IsBorder(field.Down(move))
        //            || field.IsBorder(field.Left(move))
        //            || field.IsBorder(field.Right(move)))
        //        {
        //            field.field[move] |= Field::Masks::kVisitedBit;
        //            q.emplace(move, 0);
        //        }
        //        else
        //            field.field[field.Down(move)] &= ~Field::Masks::kVisitedBit;
        //    }
        while (!q.empty())
        {
            BfsState state = q.front();
            q.pop();
            for (auto cur_move : { field.Up(state.move), field.Down(state.move), field.Left(state.move), field.Right(state.move) })
            {
                if (!field.IsBorder(cur_move) && !field.IsVisited(cur_move) && !field.IsOwner(cur_move, player))
                {
                    field.field[cur_move] |= Field::Masks::kVisitedBit;
                    q.emplace(cur_move, state.dist + 1);
                }
            }
        }
        for (int i = 0; i < q.r; i++)
            field.field[q[i].move] &= ~Field::Masks::kVisitedBit;
    }
#endif
}

bool AStar::HaveExit(Move move)
{
#ifdef ASTAR_EXTENDED_FEATURE
    auto dist = field.player == kPlayerRed ? distRed.get() : distBlack.get();
#endif
    cur_stk.clear();
    next_stk.clear();
    cur_stk.push(move);
    field.field[move] |= Field::Masks::kVisitedBit;
    field.q.push(move);
    do
    {
        while (!cur_stk.empty())
        {
            Move move = cur_stk.top();
            cur_stk.pop();

            // Если сосед граница (возможно искусственно установленная эвристикой для быстрого выхода), то окружить нельзя
            // В посещённые клетки ходить нет смысла, а также мы не можем проходить сквозь свои клетки
            if (field.IsBorder(field.Up(move)))
                return true;
            if (!field.IsVisited(field.Up(move)) && !field.IsOwner(field.Up(move), field.player))
            {
                field.field[field.Up(move)] |= Field::Masks::kVisitedBit;
                field.q.push(field.Up(move));
                (dist[field.Up(move)] - dist[move] < 0 ? cur_stk : next_stk).push(field.Up(move));
            }

            if (field.IsBorder(field.Down(move)))
                return true;
            if (!field.IsVisited(field.Down(move)) && !field.IsOwner(field.Down(move), field.player))
            {
                field.field[field.Down(move)] |= Field::Masks::kVisitedBit;
                field.q.push(field.Down(move));
                (dist[field.Down(move)] - dist[move] < 0 ? cur_stk : next_stk).push(field.Down(move));
            }

            if (field.IsBorder(field.Left(move)))
                return true;
            if (!field.IsVisited(field.Left(move)) && !field.IsOwner(field.Left(move), field.player))
            {
                field.field[field.Left(move)] |= Field::Masks::kVisitedBit;
                field.q.push(field.Left(move));
                (dist[field.Left(move)] - dist[move] < 0 ? cur_stk : next_stk).push(field.Left(move));
            }

            if (field.IsBorder(field.Right(move)))
                return true;
            if (!field.IsVisited(field.Right(move)) && !field.IsOwner(field.Right(move), field.player))
            {
                field.field[field.Right(move)] |= Field::Masks::kVisitedBit;
                field.q.push(field.Right(move));
                (dist[field.Right(move)] - dist[move] < 0 ? cur_stk : next_stk).push(field.Right(move));
            }
        }
        cur_stk.swap(next_stk);
    } while (!cur_stk.empty());
    return false;
}