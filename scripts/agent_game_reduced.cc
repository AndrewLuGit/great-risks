#include <great_risks/reduced_game.hh>
#include <great_risks/greedy_agent_reduced.hh>
#include <great_risks/mcts_agent_reduced.hh>
#include <nlohmann/json.hpp>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace great_risks;
using json = nlohmann::json;

ReducedField field;

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
    std::vector<ReducedAgent *> agents;
    agents.push_back(new GreedyAgentReduced(0));
    agents.push_back(new MCTSAgentReduced(1, 0, time(NULL)));
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
