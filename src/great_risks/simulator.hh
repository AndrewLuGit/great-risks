#pragma once


#include <array>
#include <string>
#include <vector>
#include <unordered_set>

namespace great_risks
{
    struct MobileGoal
    {
        int x;
        int y;
        std::string rings;
        bool tipped;
    };

    struct WallStake
    {
        int x;
        int y;
        std::string rings;
    };

    struct Robot
    {
        int x;
        int y;
        int goal = -1;
        std::string rings;
        bool is_red;
    };

    enum Action
    {
        MOVE_NORTH,
        MOVE_SOUTH,
        MOVE_EAST,
        MOVE_WEST,
        GRAB_MOBILE_GOAL,
        RELEASE_MOBILE_GOAL,
        TIP_MOBILE_GOAL,
        UNTIP_MOBILE_GOAL,
        PICK_UP_RED,
        PICK_UP_BLUE,
        RELEASE_RING,
        SCORE_MOBILE_GOAL,
        SCORE_WALL_STAKE,
        DESCORE_MOBILE_GOAL,
        DESCORE_WALL_STAKE
    };

    class Field
    {
    public:
        MobileGoal goals[5];
        WallStake stakes[2];
        int red_rings[11][11] = {};
        int blue_rings[11][11] = {};
        std::vector<Robot> robots;
        int time_remaining = 120;
        Field();
        void add_robot(Robot robot);
        std::unordered_set<Action> legal_actions(int i);
        Field perform_action(int i, Action a);
        std::array<int, 2> calculate_scores();
    };
}  // namespace great_risks
