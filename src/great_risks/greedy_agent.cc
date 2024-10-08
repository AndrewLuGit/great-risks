#include "greedy_agent.hh"

#include <algorithm>

namespace great_risks {
    Action GreedyAgent::next_action(Field field) {
        Robot robot_state = field.robots[robot_index];
        auto legal_actions = field.legal_actions(robot_index);
        // if there is no goal, go towards goal
        if (robot_state.goal == NO_GOAL) {
            std::unordered_set<std::array<std::uint8_t, 2>> grabbable_goals;
            for (auto &goal : field.goals) {
                if (goal.x != ON_ROBOT && !in_protected_corner(goal.x, goal.y, field.time_remaining) && !goal.tipped && goal.rings.size() < 6) {
                    grabbable_goals.insert({goal.x, goal.y});
                }
            }
            auto search_result = field.shortest_path({robot_state.x, robot_state.y}, grabbable_goals, robot_state.is_red);
            // if we can grab a goal, grab it
            if (search_result.second.empty() && std::find(legal_actions.begin(), legal_actions.end(), GRAB_MOBILE_GOAL) != legal_actions.end()) {
                return GRAB_MOBILE_GOAL;
            }
            if (!search_result.second.empty()) {
                return search_result.second[0];
            }
        }
        else if (robot_state.goal != NO_GOAL) {
            auto goal = field.goals[robot_state.goal];
            // if we can score more rings on the goal
            if (std::find(legal_actions.begin(), legal_actions.end(), SCORE_MOBILE_GOAL) != legal_actions.end()) {
                return SCORE_MOBILE_GOAL;
            }
            // otherwise if goal can fit more rings pick up rings
            if (goal.rings.size() < 6) {
                if (robot_state.is_red && std::find(legal_actions.begin(), legal_actions.end(), PICK_UP_RED) != legal_actions.end()) {
                    return PICK_UP_RED;
                }
                if (!robot_state.is_red && std::find(legal_actions.begin(), legal_actions.end(), PICK_UP_BLUE) != legal_actions.end()) {
                    return PICK_UP_BLUE;
                }
                std::unordered_set<std::array<std::uint8_t, 2>> grabbable_rings;
                for (uint8_t i = 0; i < 11; i++) {
                    for (uint8_t j = 0; j < 11; j++) {
                        std::array<std::uint8_t, 2> ring_pos = {i, j};
                        if (robot_state.is_red && field.red_rings[i][j]) {
                            grabbable_rings.insert(ring_pos);
                        }
                        else if (!robot_state.is_red && field.blue_rings[i][j]) {
                            grabbable_rings.insert(ring_pos);
                        }
                    }
                }
                if (!grabbable_rings.empty()) {
                    auto search_result = field.shortest_path({robot_state.x, robot_state.y}, grabbable_rings, robot_state.is_red);
                    return search_result.second[0];
                }
            }
            // otherwise try to release goal in positive corner
            bool left_corner_occupied = false;
            bool right_corner_occupied = false;
            for (auto &goal : field.goals) {
                if (goal.x == 10 && goal.y == 0) {
                    left_corner_occupied = true;
                }
                if (goal.x == 10 && goal.y == 10) {
                    right_corner_occupied = true;
                }
            }
            std::unordered_set<std::array<std::uint8_t, 2>> positive_corners;
            if (!left_corner_occupied) {
                positive_corners.insert({10, 0});
            }
            if (!right_corner_occupied) {
                positive_corners.insert({10, 10});
            }
            if (!positive_corners.empty()) {
                auto search_result = field.shortest_path({robot_state.x, robot_state.y}, positive_corners, robot_state.is_red);
                if (!search_result.second.empty()) {
                    return search_result.second[0];
                } else if (std::find(legal_actions.begin(), legal_actions.end(), RELEASE_MOBILE_GOAL) != legal_actions.end()) {
                    return RELEASE_MOBILE_GOAL;
                }
            }
        }
        return legal_actions[0];
    }
}