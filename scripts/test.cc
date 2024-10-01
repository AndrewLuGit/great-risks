#include <great_risks/simulator.hh>
#include <nlohmann/json.hpp>
#include <iostream>

using namespace great_risks;
using json = nlohmann::json;

Field field;

void print_state() {
    json j;
    j["goals"] = json::array();
    for (MobileGoal &goal : field.goals) {
        j["goals"].push_back({
            {"x", goal.x},
            {"y", goal.y},
            {"rings", goal.rings},
            {"tipped", goal.tipped}
        });
    }
    j["stakes"] = json::array();
    for (WallStake &stake : field.stakes) {
        j["stakes"].push_back({
            {"rings", stake.rings}
        });
    }
    for (Robot &robot : field.robots) {
        j["robots"].push_back({
            {"x", robot.x},
            {"y", robot.y},
            {"goal", robot.goal},
            {"is_red", robot.is_red},
            {"rings", robot.rings}
        });
    }
    j["time_remaining"] = field.time_remaining;
    j["legal_actions"] = json::array();
    for (const Action &a : field.legal_actions(0)) {
        j["legal_actions"].push_back(a);
    }

    auto [red_score, blue_score] = field.calculate_scores();
    j["scores"] = {
        {"red", red_score},
        {"blue", blue_score}
    };
    std::cout << j.dump() << std::endl;
}

auto main(int argc, char**argv) -> int
{
    Robot robot;
    robot.x = 1;
    robot.y = 0;
    robot.is_red = true;
    field.add_robot(robot);
    while (true) {
        print_state();
        json j;
        std::cin >> j;
        auto legal_actions = field.legal_actions(0);
        if (legal_actions.find(j["action"]) != legal_actions.end()) {
            field = field.perform_action(0, j["action"]);
        }
    }
}
