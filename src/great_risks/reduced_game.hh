#pragma once

#include "simulator.hh"

namespace great_risks {
    class ReducedField {
        public:
        MobileGoal goals[3];
        WallStake stakes[2];
        uint8_t red_rings[5][5] = {};
        uint8_t blue_rings[5][5] = {};
        Robot robots[2];
        uint8_t time_remaining = 30;
        ReducedField();
        std::vector<Action> legal_actions(uint8_t i);
        ReducedField perform_action(uint8_t i, Action a);
        std::array<int, 2> calculate_scores();
        std::pair<std::array<uint8_t, 2>, std::vector<Action>> shortest_path(std::array<uint8_t, 2> begin, std::unordered_set<std::array<uint8_t, 2>> targets, bool is_red);
    };

    class ReducedAgent
    {
    protected:
        std::uint8_t robot_index;
    public:
        ReducedAgent(std::uint8_t robot_index): robot_index(robot_index) {};
        virtual Action next_action(ReducedField field) = 0;
    };
}