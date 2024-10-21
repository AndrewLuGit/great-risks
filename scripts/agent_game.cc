#include <great_risks/simulator.hh>
#include <great_risks/greedy_agent.hh>
#include <great_risks/random_agent.hh>
#include <great_risks/mcts_agent_greedy.hh>
#include <great_risks/mcts_agent_random.hh>
#include <nlohmann/json.hpp>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace great_risks;
using json = nlohmann::json;

Field field;

void print_state()
{
    json j;
    j["goals"] = json::array();
    for (MobileGoal &goal : field.goals)
    {
        j["goals"].push_back({{"x", goal.x}, {"y", goal.y}, {"rings", goal.rings}, {"tipped", goal.tipped}});
    }
    j["stakes"] = json::array();
    for (WallStake &stake : field.stakes)
    {
        j["stakes"].push_back({{"rings", stake.rings}});
    }
    for (Robot &robot : field.robots)
    {
        j["robots"].push_back(
            {{"x", robot.x},
             {"y", robot.y},
             {"goal", robot.goal},
             {"is_red", robot.is_red},
             {"rings", robot.rings}});
    }
    j["time_remaining"] = field.time_remaining;
    j["legal_actions"] = field.legal_actions(0);

    auto [red_score, blue_score] = field.calculate_scores();
    j["scores"] = {{"red", red_score}, {"blue", blue_score}};
    j["red_rings"] = field.red_rings;
    j["blue_rings"] = field.blue_rings;
    std::cout << j.dump() << std::endl;
}

auto main(int argc, char **argv) -> int
{
    Robot robot_1;
    robot_1.x = 1;
    robot_1.y = 0;
    robot_1.is_red = true;
    field.add_robot(robot_1);
    // Robot robot_2;
    // robot_2.x = 9;
    // robot_2.y = 0;
    // robot_1.is_red = true;
    // field.add_robot(robot_2);
    // Robot robot_3;
    // robot_3.x = 1;
    // robot_3.y = 10;
    // robot_3.is_red = false;
    // field.add_robot(robot_3);
    Robot robot_4;
    robot_4.x = 9;
    robot_4.y = 10;
    robot_4.is_red = false;
    field.add_robot(robot_4);
    std::vector<Agent *> agents;
    agents.push_back(new MCTSAgentGreedy(0, 1, time(NULL)));
    agents.push_back(new GreedyAgent(1));
    while (field.time_remaining > 0)
    {
        print_state();
        json j;
        std::cin >> j;
        for (size_t i = 0; i < agents.size(); i++)
        {
            auto action = agents[i]->next_action(field);
            field.perform_action(i, action);
        }
        field.time_remaining--;
    }
    print_state();
}
