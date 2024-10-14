#include "reduced_game.hh"

namespace great_risks {
    ReducedField::ReducedField() {
        goals[0].x = 1;
        goals[0].y = 2;
        goals[1].x = 2;
        goals[1].y = 2;
        goals[2].x = 3;
        goals[2].y = 2;

        stakes[0].x = 0;
        stakes[0].y = 2;
        stakes[1].x = 4;
        stakes[1].y = 2;

        red_rings[0][0] = 2;
        red_rings[0][2] = 1;
        red_rings[0][4] = 2;
        red_rings[1][1] = 1;
        red_rings[1][3] = 1;
        red_rings[2][1] = 1;
        red_rings[2][3] = 1;
        red_rings[3][1] = 1;
        red_rings[3][3] = 1;
        red_rings[4][0] = 2;
        red_rings[4][2] = 1;
        red_rings[4][4] = 2;

        blue_rings[0][0] = 2;
        blue_rings[0][2] = 1;
        blue_rings[0][4] = 2;
        blue_rings[1][1] = 1;
        blue_rings[1][3] = 1;
        blue_rings[2][1] = 1;
        blue_rings[2][3] = 1;
        blue_rings[3][1] = 1;
        blue_rings[3][3] = 1;
        blue_rings[4][0] = 2;
        blue_rings[4][2] = 1;
        blue_rings[4][4] = 2;

        robots[0].x = 2;
        robots[0].y = 0;
        robots[0].is_red = true;
        robots[0].goal = NO_GOAL;
        robots[1].x = 2;
        robots[1].y = 4;
        robots[1].is_red = false;
        robots[1].goal = NO_GOAL;
    }

    bool legal_move(int x, int y, ReducedField &field)
    {
        if (x < 0 || x > 4)
        {
            return false;
        }
        if (y < 0 || y > 4)
        {
            return false;
        }
        for (Robot &robot : field.robots)
        {
            if (x == robot.x && y == robot.y)
            {
                return false;
            }
        }
        return true;
    }

    std::vector<Action> ReducedField::legal_actions(uint8_t i) {
        Robot robot = robots[i];
        std::vector<Action> result;
        if (legal_move(robot.x - 1, robot.y, *this))
        {
            result.push_back(MOVE_NORTH);
        }
        if (legal_move(robot.x + 1, robot.y, *this))
        {
            result.push_back(MOVE_SOUTH);
        }
        if (legal_move(robot.x, robot.y + 1, *this))
        {
            result.push_back(MOVE_EAST);
        }
        if (legal_move(robot.x, robot.y - 1, *this))
        {
            result.push_back(MOVE_WEST);
        }
        uint8_t goal = NO_GOAL;
        for (uint8_t i = 0; i < 3; i++) {
            if (goals[i].x == robot.x && goals[i].y == robot.y) {
                goal = i;
                break;
            }
        }
        if (robot.goal == NO_GOAL && goal != NO_GOAL) {
            if (goals[goal].tipped)
            {
                result.push_back(UNTIP_MOBILE_GOAL);
            }
            else
            {
                result.push_back(GRAB_MOBILE_GOAL);
                result.push_back(TIP_MOBILE_GOAL);
            }
        } else if (robot.goal != NO_GOAL) {
            if (goal == NO_GOAL) {
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
        for (WallStake &stake : stakes)
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

    ReducedField ReducedField::perform_action(std::uint8_t i, Action a) {
        ReducedField result = *this;
        Robot &robot = result.robots[i];
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
            for (int i = 0; i < 5; i++) {
                if (result.goals[i].x == robot.x && result.goals[i].y == robot.y) {
                    robot.goal = i;
                    result.goals[i].x = ON_ROBOT;
                    result.goals[i].y = ON_ROBOT;
                }
            }
            break;
        case RELEASE_MOBILE_GOAL:
            result.goals[robot.goal].x = robot.x;
            result.goals[robot.goal].y = robot.y;
            robot.goal = ON_ROBOT;
            break;
        case TIP_MOBILE_GOAL:
            for (int i = 0; i < 5; i++) {
                if (result.goals[i].x == robot.x && result.goals[i].y == robot.y) {
                    result.goals[i].tipped = true;
                }
            }
            break;
        case UNTIP_MOBILE_GOAL:
            for (int i = 0; i < 5; i++) {
                if (result.goals[i].x == robot.x && result.goals[i].y == robot.y) {
                    result.goals[i].tipped = false;
                }
            }
            break;
        case PICK_UP_RED:
            result.red_rings[robot.x][robot.y]--;
            robot.rings.push_back(RED);
            break;
        case PICK_UP_BLUE:
            result.blue_rings[robot.x][robot.y]--;
            robot.rings.push_back(BLUE);
            break;
        case RELEASE_RING:
            if (robot.rings.back() == RED) {
                result.red_rings[robot.x][robot.y]++;
            } else {
                result.blue_rings[robot.x][robot.y]++;
            }
            robot.rings.pop_back();
            break;
        case SCORE_MOBILE_GOAL:
            result.goals[robot.goal].rings.push_back(robot.rings.front());
            robot.rings.pop_front();
            break;
        case SCORE_WALL_STAKE:
            for (WallStake &stake : result.stakes) {
                if (stake.x == robot.x && stake.y == robot.y) {
                    stake.rings.push_back(robot.rings.front());
                    robot.rings.pop_front();
                }
            }
            break;
        case DESCORE_MOBILE_GOAL:
            robot.rings.push_front(result.goals[robot.goal].rings.back());
            result.goals[robot.goal].rings.pop_back();
            break;
        case DESCORE_WALL_STAKE:
            for (WallStake &stake : result.stakes) {
                if (stake.x == robot.x && stake.y == robot.y) {
                    robot.rings.push_front(stake.rings.back());
                    stake.rings.pop_back();
                }
            }
            break;
        }
        return result;
    }

    std::array<int, 2> ReducedField::calculate_scores() {
        int red_score = 0;
        int blue_score = 0;
        for (MobileGoal &goal: goals) {
            if (!goal.rings.empty()) {
                int multiplier = 1;
                if (goal.x == 0 && (goal.y == 0 || goal.y == 4)) {
                    multiplier = -1;
                } else if (goal.x == 4 && (goal.y == 0 || goal.y == 4)) {
                    multiplier = 2;
                }
                for (auto &c : goal.rings) {
                    if (c == RED) {
                        red_score += multiplier;
                    } else {
                        blue_score += multiplier;
                    }
                }
                if (goal.rings.back() == RED) {
                    red_score += 2 * multiplier;
                } else {
                    blue_score += 2 * multiplier;
                }
            }
        }
        for (WallStake &stake : stakes) {
            if (!stake.rings.empty()) {
                for (auto &c : stake.rings) {
                    if (c == RED) {
                        red_score += 1;
                    } else {
                        blue_score += 1;
                    }
                }
                if (stake.rings.back() == RED) {
                    red_score += 2;
                } else {
                    blue_score += 2;
                }
            }
        }
        if (red_score < 0) red_score = 0;
        if (blue_score < 0) blue_score = 0;
        return {red_score, blue_score};
    }

    std::pair<std::array<std::uint8_t, 2>, std::vector<Action>> ReducedField::shortest_path(std::array<std::uint8_t, 2> begin, std::unordered_set<std::array<std::uint8_t, 2>> targets, bool is_red) {
        std::unordered_set<std::array<std::uint8_t, 2>> explored;
        std::deque<std::pair<std::array<std::uint8_t, 2>, std::vector<Action>>> queue;
        queue.push_back({begin, std::vector<Action>()});
        explored.insert(begin);
        while (!queue.empty()) {
            auto v = queue.front();
            queue.pop_front();
            if (targets.find(v.first) != targets.end()) {
                return v;
            }
            auto x = v.first[0];
            auto y = v.first[1];
            std::array<std::uint8_t, 2> north_pos = {static_cast<uint8_t>(x - 1), y};
            std::array<std::uint8_t, 2> south_pos = {static_cast<uint8_t>(x + 1), y};
            std::array<std::uint8_t, 2> east_pos = {x, static_cast<uint8_t>(y + 1)};
            std::array<std::uint8_t, 2> west_pos = {x, static_cast<uint8_t>(y - 1)};
            if (legal_move(x - 1, y, *this) && explored.find(north_pos) == explored.end())
            {
                auto moves = v.second;
                moves.push_back(MOVE_NORTH);
                queue.push_back({north_pos, moves});
                explored.insert(north_pos);
            }
            if (legal_move(x + 1, y, *this) && explored.find(south_pos) == explored.end())
            {
                auto moves = v.second;
                moves.push_back(MOVE_SOUTH);
                queue.push_back({south_pos, moves});
                explored.insert(south_pos);
            }
            if (legal_move(x, y + 1, *this) && explored.find(east_pos) == explored.end())
            {
                auto moves = v.second;
                moves.push_back(MOVE_EAST);
                queue.push_back({east_pos, moves});
                explored.insert(east_pos);
            }
            if (legal_move(x, y - 1, *this) && explored.find(west_pos) == explored.end())
            {
                auto moves = v.second;
                moves.push_back(MOVE_WEST);
                queue.push_back({west_pos, moves});
                explored.insert(west_pos);
            }
        }
        return {begin, std::vector<Action>()};
    }
}