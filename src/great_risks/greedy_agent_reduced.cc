#include "greedy_agent_reduced.hh"

#include <algorithm>

namespace great_risks {
    Action GreedyAgentReduced::next_action(ReducedField field) {
        Robot robot_state = field.robots[robot_index];
        auto legal_actions = field.legal_actions(robot_index);
        // if there is no goal, go towards goal
        if (robot_state.goal == NO_GOAL) {
            std::unordered_set<std::array<std::uint8_t, 2>> grabbable_goals;
            for (auto &goal : field.goals) {
                if (goal.x != ON_ROBOT && !goal.tipped && goal.rings.size() < 6) {
                    grabbable_goals.insert({goal.x, goal.y});
                }
            }
            auto search_result = field.shortest_path({robot_state.x, robot_state.y}, grabbable_goals, robot_state.is_red);
            // if we can grab a goal, grab it
            if (search_result.second.empty() && !grabbable_goals.empty() && std::find(legal_actions.begin(), legal_actions.end(), GRAB_MOBILE_GOAL) != legal_actions.end()) {
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
            bool left_corner_occupied = false;
            bool right_corner_occupied = false;
            for (auto &goal : field.goals) {
                if (goal.x == 4 && goal.y == 0) {
                    left_corner_occupied = true;
                }
                if (goal.x == 4 && goal.y == 4) {
                    right_corner_occupied = true;
                }
            }
            // get state of positive corners
            std::unordered_set<std::array<std::uint8_t, 2>> positive_corners;
            if (!left_corner_occupied) {
                positive_corners.insert({4, 0});
            }
            if (!right_corner_occupied) {
                positive_corners.insert({4, 4});
            }
            // otherwise try to release goal in positive corner
            if (goal.rings.size() == 6 && !positive_corners.empty()) {
                auto search_result = field.shortest_path({robot_state.x, robot_state.y}, positive_corners, robot_state.is_red);
                if (!search_result.second.empty()) {
                    return search_result.second[0];
                } else if (positive_corners.find(search_result.first) != positive_corners.end() && std::find(legal_actions.begin(), legal_actions.end(), RELEASE_MOBILE_GOAL) != legal_actions.end()) {
                    // only release goal in positive corner, otherwise keep goal on robot
                    return RELEASE_MOBILE_GOAL;
                }
            }
        }
        if (robot_state.rings.size() > 0) {
            std::unordered_set<std::array<std::uint8_t, 2>> stakes;
            if (field.stakes[0].rings.size() < 6) {
                stakes.insert({0, 2});
            }
            if (field.stakes[1].rings.size() < 6) {
                stakes.insert({4, 2});
            }
            auto search_result = field.shortest_path({robot_state.x, robot_state.y}, stakes, robot_state.is_red);
            if (!search_result.second.empty()) {
                return search_result.second[0];
            }
        }
        // if we can score on wall stake
        if (std::find(legal_actions.begin(), legal_actions.end(), SCORE_WALL_STAKE) != legal_actions.end()) {
            return SCORE_WALL_STAKE;
        }
        if (robot_state.is_red && std::find(legal_actions.begin(), legal_actions.end(), PICK_UP_RED) != legal_actions.end()) {
            return PICK_UP_RED;
        }
        if (!robot_state.is_red && std::find(legal_actions.begin(), legal_actions.end(), PICK_UP_BLUE) != legal_actions.end()) {
            return PICK_UP_BLUE;
        }
        std::unordered_set<std::array<std::uint8_t, 2>> grabbable_rings;
        for (uint8_t i = 0; i < 5; i++) {
            for (uint8_t j = 0; j < 5; j++) {
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
            if (!search_result.second.empty()) {
                return search_result.second[0];
            }
        }
        return legal_actions[0];
    }
}