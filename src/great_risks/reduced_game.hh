#pragma once

#include "simulator.hh"

namespace great_risks {
    class ReducedField {
        public:
        std::array<MobileGoal, 3> goals;
        std::array<WallStake, 2> stakes;
        std::array<std::array<uint8_t, 5>, 5> red_rings = {};
        std::array<std::array<uint8_t, 5>, 5> blue_rings = {};
        std::array<Robot, 2> robots;
        uint8_t time_remaining = 30;
        ReducedField();
        std::vector<Action> legal_actions(uint8_t i);
        void perform_action(uint8_t i, Action a);
        std::array<int, 2> calculate_scores();
        std::pair<std::array<uint8_t, 2>, std::vector<Action>> shortest_path(std::array<uint8_t, 2> begin, std::unordered_set<std::array<uint8_t, 2>> targets, bool is_red);
        bool operator==(const ReducedField& other) const
        {
            return (time_remaining == other.time_remaining && std::equal(goals.begin(), goals.end(), other.goals.begin()) && std::equal(stakes.begin(), stakes.end(), other.stakes.begin()) && std::equal(robots.begin(), robots.end(), other.robots.begin()) && std::equal(red_rings.begin(), red_rings.end(), other.red_rings.begin()) && std::equal(blue_rings.begin(), blue_rings.end(), other.blue_rings.begin()));
        }
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

template<>
struct std::hash<great_risks::ReducedField>
{
    size_t operator()(const great_risks::ReducedField& field) const noexcept
    {
        size_t seed = 0;
        hash<great_risks::MobileGoal> goal_hash;
        for (auto &goal : field.goals) {
            seed ^= goal_hash(goal) + (seed << 6) + (seed >> 2);
        }
        hash<great_risks::WallStake> stake_hash;
        for (auto &stake : field.stakes) {
            seed ^= stake_hash(stake) + (seed << 6) + (seed >> 2);
        }
        hash<great_risks::Robot> robot_hash;
        for (auto &robot : field.robots) {
            seed ^= robot_hash(robot) + (seed << 6) + (seed >> 2);
        }
        seed = field.time_remaining + (seed << 6) + (seed >> 2);
        return seed;
    }
};