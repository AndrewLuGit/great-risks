#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace std
{
    template <typename T, size_t N>
    struct hash<array<T, N>>
    {
        typedef array<T, N> argument_type;
        typedef size_t result_type;

        result_type operator()(const argument_type &a) const
        {
            hash<T> hasher;
            result_type h = 0;
            for (result_type i = 0; i < N; ++i)
            {
                h = h * 31 + hasher(a[i]);
            }
            return h;
        }
    };
}  // namespace std

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
        std::vector<Ring> rings;
        bool tipped;

        bool operator==(const MobileGoal &other) const
        {
            return (x == other.x && y == other.y && rings == other.rings && tipped == other.tipped);
        }
    };

    struct WallStake
    {
        std::uint8_t x;
        std::uint8_t y;
        std::vector<Ring> rings;

        bool operator==(const WallStake &other) const
        {
            return (rings == other.rings && x == other.x && y == other.y);
        }
    };

    struct Robot
    {
        std::uint8_t x;
        std::uint8_t y;
        std::uint8_t goal = NO_GOAL;
        std::vector<Ring> rings;
        bool is_red;

        bool operator==(const Robot &other) const
        {
            return (
                x == other.x && y == other.y && goal == other.goal && rings == other.rings &&
                is_red == other.is_red);
        }
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
        DESCORE_WALL_STAKE,
    };

    class Field
    {
    public:
        std::array<MobileGoal, 5> goals;
        std::array<WallStake, 2> stakes;
        std::array<std::array<uint8_t, 11>, 11> red_rings = {};
        std::array<std::array<uint8_t, 11>, 11> blue_rings = {};
        std::vector<Robot> robots;
        std::uint8_t time_remaining = 120;
        Field();
        void add_robot(Robot robot);
        std::vector<Action> legal_actions(std::uint8_t i) const;
        void perform_action(std::uint8_t i, Action a);
        std::array<int, 2> calculate_scores() const;
        std::pair<std::array<std::uint8_t, 2>, std::vector<Action>> shortest_path(
            std::array<std::uint8_t, 2> begin,
            std::unordered_set<std::array<std::uint8_t, 2>> targets,
            bool is_red) const;

        bool operator==(const Field &other) const
        {
            return (
                time_remaining == other.time_remaining &&
                std::equal(goals.begin(), goals.end(), other.goals.begin()) &&
                std::equal(stakes.begin(), stakes.end(), other.stakes.begin()) &&
                std::equal(robots.begin(), robots.end(), other.robots.begin()) &&
                std::equal(red_rings.begin(), red_rings.end(), other.red_rings.begin()) &&
                std::equal(blue_rings.begin(), blue_rings.end(), other.blue_rings.begin()));
        }
    };

    bool in_protected_corner(int x, int y, int time_remaining);
}  // namespace great_risks

namespace std
{
    template <>
    struct hash<great_risks::MobileGoal>
    {
        size_t operator()(const great_risks::MobileGoal &goal) const noexcept
        {
            size_t hash = goal.x;
            hash = (hash << 4) + goal.y;
            for (auto &ring : goal.rings)
            {
                hash = (hash << 4) + ring;
            }
            return hash;
        }
    };

    template <>
    struct hash<great_risks::WallStake>
    {
        size_t operator()(const great_risks::WallStake &stake) const noexcept
        {
            size_t hash = stake.x;
            hash = (hash << 4) + stake.y;
            for (auto &ring : stake.rings)
            {
                hash = (hash << 4) + ring;
            }
            return hash;
        }
    };

    template <>
    struct hash<great_risks::Robot>
    {
        size_t operator()(const great_risks::Robot &robot) const noexcept
        {
            size_t hash = robot.x;
            hash = (hash << 4) + robot.y;
            hash = (hash << 4) + robot.goal;
            for (auto &ring : robot.rings)
            {
                hash = (hash << 4) + ring;
            }
            return hash;
        }
    };

    template <>
    struct hash<great_risks::Field>
    {
        size_t operator()(const great_risks::Field &field) const noexcept
        {
            size_t seed = 0;
            hash<great_risks::MobileGoal> goal_hash;
            for (auto &goal : field.goals)
            {
                seed ^= goal_hash(goal) + (seed << 6) + (seed >> 2);
            }
            hash<great_risks::WallStake> stake_hash;
            for (auto &stake : field.stakes)
            {
                seed ^= stake_hash(stake) + (seed << 6) + (seed >> 2);
            }
            hash<great_risks::Robot> robot_hash;
            for (auto &robot : field.robots)
            {
                seed ^= robot_hash(robot) + (seed << 6) + (seed >> 2);
            }
            seed = field.time_remaining + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}  // namespace std
