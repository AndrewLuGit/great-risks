#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#define ON_ROBOT 255
#define NO_GOAL 255

namespace great_risks
{
    enum Ring
    {
        RED,
        BLUE
    };

    struct MobileGoal
    {
        std::uint8_t x;
        std::uint8_t y;
        std::deque<Ring> rings;
        bool tipped;
    };

    struct WallStake
    {
        std::uint8_t x;
        std::uint8_t y;
        std::deque<Ring> rings;
    };

    struct Robot
    {
        std::uint8_t x;
        std::uint8_t y;
        std::uint8_t goal = NO_GOAL;
        std::deque<Ring> rings;
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
        std::uint8_t red_rings[11][11] = {};
        std::uint8_t blue_rings[11][11] = {};
        std::vector<Robot> robots;
        std::uint8_t time_remaining = 120;
        Field();
        void add_robot(Robot robot);
        std::vector<Action> legal_actions(int i);
        Field perform_action(int i, Action a);
        std::array<int, 2> calculate_scores();
    };
}  // namespace great_risks
