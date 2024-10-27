#include "simulator.hh"

#include <deque>

namespace great_risks
{
    Field::Field()
    {
        goals[0].x = 1;
        goals[0].y = 5;
        goals[1].x = 5;
        goals[1].y = 1;
        goals[2].x = 5;
        goals[2].y = 5;
        goals[3].x = 5;
        goals[3].y = 9;
        goals[4].x = 9;
        goals[4].y = 5;

        stakes[0].x = 0;
        stakes[0].y = 5;
        stakes[1].x = 10;
        stakes[1].y = 5;

        red_rings[0][0] = 2;
        red_rings[0][5] = 1;
        red_rings[0][10] = 2;
        red_rings[1][1] = 1;
        red_rings[1][3] = 1;
        red_rings[1][7] = 1;
        red_rings[1][9] = 1;
        red_rings[3][7] = 1;
        red_rings[5][0] = 1;
        red_rings[5][10] = 1;
        red_rings[7][7] = 1;
        red_rings[9][1] = 1;
        red_rings[9][3] = 1;
        red_rings[9][7] = 1;
        red_rings[9][9] = 1;
        red_rings[10][0] = 2;
        red_rings[10][5] = 1;
        red_rings[10][10] = 2;

        blue_rings[0][0] = 2;
        blue_rings[0][5] = 1;
        blue_rings[0][10] = 2;
        blue_rings[1][1] = 1;
        blue_rings[1][3] = 1;
        blue_rings[1][7] = 1;
        blue_rings[1][9] = 1;
        blue_rings[3][3] = 1;
        blue_rings[5][0] = 1;
        blue_rings[5][10] = 1;
        blue_rings[7][3] = 1;
        blue_rings[9][1] = 1;
        blue_rings[9][3] = 1;
        blue_rings[9][7] = 1;
        blue_rings[9][9] = 1;
        blue_rings[10][0] = 2;
        blue_rings[10][5] = 1;
        blue_rings[10][10] = 2;
    }

    void Field::add_robot(Robot robot)
    {
        robots.emplace_back(robot);
    }

    bool legal_move(int x, int y, bool is_red, const Field &field)
    {
        if (x < 0 || x > 10)
        {
            return false;
        }
        if (y < 0 || y > 10)
        {
            return false;
        }
        if (field.time_remaining > 90)
        {
            if (is_red && y > 5)
            {
                return false;
            }
            if (!is_red && y < 5)
            {
                return false;
            }
        }
        if (x == 3 && y == 5)
        {
            return false;
        }
        if (x == 5 && y == 3)
        {
            return false;
        }
        if (x == 5 && y == 7)
        {
            return false;
        }
        if (x == 7 && y == 5)
        {
            return false;
        }
        for (const Robot &robot : field.robots)
        {
            if (x == robot.x && y == robot.y)
            {
                return false;
            }
        }
        return true;
    }

    bool in_protected_corner(int x, int y, int time_remaining)
    {
        return time_remaining <= 15 && x == 10 && (y == 0 || y == 10);
    }

    std::uint8_t can_interact_with_goal(int x, int y, const Field &field)
    {
        // cannot interact with positive corners during endgame
        if (in_protected_corner(x, y, field.time_remaining))
        {
            return NO_GOAL;
        }
        for (size_t i = 0; i < field.goals.size(); i++)
        {
            if (field.goals[i].x == x && field.goals[i].y == y)
            {
                return i;
            }
        }
        return NO_GOAL;
    }

    std::vector<Action> Field::legal_actions(std::uint8_t i) const
    {
        Robot robot = robots[i];
        std::vector<Action> result;
        if (legal_move(robot.x - 1, robot.y, robot.is_red, *this))
        {
            result.push_back(MOVE_NORTH);
        }
        if (legal_move(robot.x + 1, robot.y, robot.is_red, *this))
        {
            result.push_back(MOVE_SOUTH);
        }
        if (legal_move(robot.x, robot.y + 1, robot.is_red, *this))
        {
            result.push_back(MOVE_EAST);
        }
        if (legal_move(robot.x, robot.y - 1, robot.is_red, *this))
        {
            result.push_back(MOVE_WEST);
        }
        std::uint8_t goal = can_interact_with_goal(robot.x, robot.y, *this);
        if (robot.goal == NO_GOAL && goal != NO_GOAL)
        {
            if (goals[goal].tipped)
            {
                result.push_back(UNTIP_MOBILE_GOAL);
            }
            else
            {
                result.push_back(GRAB_MOBILE_GOAL);
                result.push_back(TIP_MOBILE_GOAL);
            }
        }
        else if (robot.goal != NO_GOAL)
        {
            if (goal == NO_GOAL && !in_protected_corner(robot.x, robot.y, time_remaining))
            {
                result.push_back(RELEASE_MOBILE_GOAL);
            }
            if (!robot.rings.empty() && goals[robot.goal].rings.size() < 6)
            {
                result.push_back(SCORE_MOBILE_GOAL);
            }
            if (!goals[robot.goal].rings.empty() && robot.rings.size() < 2)
            {
                result.push_back(DESCORE_MOBILE_GOAL);
            }
        }
        if (robot.rings.size() < 2)
        {
            if (red_rings[robot.x][robot.y])
            {
                result.push_back(PICK_UP_RED);
            }
            if (blue_rings[robot.x][robot.y])
            {
                result.push_back(PICK_UP_BLUE);
            }
        }
        if (!robot.rings.empty())
        {
            result.push_back(RELEASE_RING);
        }
        for (const WallStake &stake : stakes)
        {
            if (robot.x == stake.x && robot.y == stake.y)
            {
                if (!robot.rings.empty() && stake.rings.size() < 6)
                {
                    result.push_back(SCORE_WALL_STAKE);
                }
                if (!stake.rings.empty() && robot.rings.size() < 2)
                {
                    result.push_back(DESCORE_WALL_STAKE);
                }
            }
        }
        return result;
    }

    void Field::perform_action(std::uint8_t i, Action a)
    {
        Robot &robot = robots[i];
        switch (a)
        {
            case MOVE_NORTH:
                robot.x--;
                break;
            case MOVE_SOUTH:
                robot.x++;
                break;
            case MOVE_EAST:
                robot.y++;
                break;
            case MOVE_WEST:
                robot.y--;
                break;
            case GRAB_MOBILE_GOAL:
                for (size_t i = 0; i < goals.size(); i++)
                {
                    if (goals[i].x == robot.x && goals[i].y == robot.y)
                    {
                        robot.goal = i;
                        goals[i].x = ON_ROBOT;
                        goals[i].y = ON_ROBOT;
                    }
                }
                break;
            case RELEASE_MOBILE_GOAL:
                goals[robot.goal].x = robot.x;
                goals[robot.goal].y = robot.y;
                robot.goal = ON_ROBOT;
                break;
            case TIP_MOBILE_GOAL:
                for (size_t i = 0; i < goals.size(); i++)
                {
                    if (goals[i].x == robot.x && goals[i].y == robot.y)
                    {
                        goals[i].tipped = true;
                    }
                }
                break;
            case UNTIP_MOBILE_GOAL:
                for (size_t i = 0; i < goals.size(); i++)
                {
                    if (goals[i].x == robot.x && goals[i].y == robot.y)
                    {
                        goals[i].tipped = false;
                    }
                }
                break;
            case PICK_UP_RED:
                red_rings[robot.x][robot.y]--;
                robot.rings.push_back(RED);
                break;
            case PICK_UP_BLUE:
                blue_rings[robot.x][robot.y]--;
                robot.rings.push_back(BLUE);
                break;
            case RELEASE_RING:
                if (robot.rings.back() == RED)
                {
                    red_rings[robot.x][robot.y]++;
                }
                else
                {
                    blue_rings[robot.x][robot.y]++;
                }
                robot.rings.pop_back();
                break;
            case SCORE_MOBILE_GOAL:
                goals[robot.goal].rings.push_back(robot.rings.front());
                robot.rings.erase(robot.rings.begin());
                break;
            case SCORE_WALL_STAKE:
                for (WallStake &stake : stakes)
                {
                    if (stake.x == robot.x && stake.y == robot.y)
                    {
                        stake.rings.push_back(robot.rings.front());
                        robot.rings.erase(robot.rings.begin());
                    }
                }
                break;
            case DESCORE_MOBILE_GOAL:
                robot.rings.insert(robot.rings.begin(), goals[robot.goal].rings.back());
                goals[robot.goal].rings.pop_back();
                break;
            case DESCORE_WALL_STAKE:
                for (WallStake &stake : stakes)
                {
                    if (stake.x == robot.x && stake.y == robot.y)
                    {
                        robot.rings.insert(robot.rings.begin(), stake.rings.back());
                        stake.rings.pop_back();
                    }
                }
                break;
        }
    }

    std::array<int, 2> Field::calculate_scores() const
    {
        int red_score = 0;
        int blue_score = 0;
        for (const MobileGoal &goal : goals)
        {
            if (!goal.rings.empty())
            {
                int multiplier = 1;
                if (goal.x == 0 && (goal.y == 0 || goal.y == 10))
                {
                    multiplier = -1;
                }
                else if (goal.x == 10 && (goal.y == 0 || goal.y == 10))
                {
                    multiplier = 2;
                }
                for (auto &c : goal.rings)
                {
                    if (c == RED)
                    {
                        red_score += multiplier;
                    }
                    else
                    {
                        blue_score += multiplier;
                    }
                }
                if (goal.rings.back() == RED)
                {
                    red_score += 2 * multiplier;
                }
                else
                {
                    blue_score += 2 * multiplier;
                }
            }
        }
        for (const WallStake &stake : stakes)
        {
            if (!stake.rings.empty())
            {
                for (auto &c : stake.rings)
                {
                    if (c == RED)
                    {
                        red_score += 1;
                    }
                    else
                    {
                        blue_score += 1;
                    }
                }
                if (stake.rings.back() == RED)
                {
                    red_score += 2;
                }
                else
                {
                    blue_score += 2;
                }
            }
        }
        if (red_score < 0)
        {
            red_score = 0;
        }
        if (blue_score < 0)
        {
            blue_score = 0;
        }
        return {red_score, blue_score};
    }

    std::pair<std::array<std::uint8_t, 2>, std::vector<Action>> Field::shortest_path(
        std::array<std::uint8_t, 2> begin,
        std::unordered_set<std::array<std::uint8_t, 2>> targets,
        bool is_red) const
    {
        std::unordered_set<std::array<std::uint8_t, 2>> explored;
        std::deque<std::pair<std::array<std::uint8_t, 2>, std::vector<Action>>> queue;
        queue.push_back({begin, std::vector<Action>()});
        explored.insert(begin);
        while (!queue.empty())
        {
            auto v = queue.front();
            queue.pop_front();
            if (targets.find(v.first) != targets.end())
            {
                return v;
            }
            auto x = v.first[0];
            auto y = v.first[1];
            std::array<std::uint8_t, 2> north_pos = {static_cast<uint8_t>(x - 1), y};
            std::array<std::uint8_t, 2> south_pos = {static_cast<uint8_t>(x + 1), y};
            std::array<std::uint8_t, 2> east_pos = {x, static_cast<uint8_t>(y + 1)};
            std::array<std::uint8_t, 2> west_pos = {x, static_cast<uint8_t>(y - 1)};
            if (legal_move(x - 1, y, is_red, *this) && explored.find(north_pos) == explored.end())
            {
                auto moves = v.second;
                moves.push_back(MOVE_NORTH);
                queue.push_back({north_pos, moves});
                explored.insert(north_pos);
            }
            if (legal_move(x + 1, y, is_red, *this) && explored.find(south_pos) == explored.end())
            {
                auto moves = v.second;
                moves.push_back(MOVE_SOUTH);
                queue.push_back({south_pos, moves});
                explored.insert(south_pos);
            }
            if (legal_move(x, y + 1, is_red, *this) && explored.find(east_pos) == explored.end())
            {
                auto moves = v.second;
                moves.push_back(MOVE_EAST);
                queue.push_back({east_pos, moves});
                explored.insert(east_pos);
            }
            if (legal_move(x, y - 1, is_red, *this) && explored.find(west_pos) == explored.end())
            {
                auto moves = v.second;
                moves.push_back(MOVE_WEST);
                queue.push_back({west_pos, moves});
                explored.insert(west_pos);
            }
        }
        return {begin, std::vector<Action>()};
    }
}  // namespace great_risks
