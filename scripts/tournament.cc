#include <great_risks/greedy_agent.hh>
#include <great_risks/mcts_agent_greedy.hh>
#include <great_risks/mcts_agent_random.hh>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>

int red_wins = 0;
int blue_wins = 0;
int ties = 0;
std::mutex mtx;
using namespace great_risks;

void run_match() {
    Field field;
    Robot robot_1;
    robot_1.x = 1;
    robot_1.y = 0;
    robot_1.is_red = true;
    field.add_robot(robot_1);
    Robot robot_2;
    robot_2.x = 9;
    robot_2.y = 10;
    robot_2.is_red = false;
    field.add_robot(robot_2);
    std::vector<Agent *> agents;
    mtx.lock();
    agents.push_back(new MCTSAgentGreedy(0, 1, rand()));
    agents.push_back(new GreedyAgent(1));
    mtx.unlock();
    while (field.time_remaining > 0) {
        for (size_t i = 0; i < agents.size(); i++)
        {
            auto action = agents[i]->next_action(field);
            field.perform_action(i, action);
        }
        field.time_remaining--;
    }
    auto [red_score, blue_score] = field.calculate_scores();
    mtx.lock();
    //std::cout << "" << red_score << " " << blue_score << "\n";
    if (red_score > blue_score) {
        red_wins++;
    } else if (red_score < blue_score) {
        blue_wins++;
    } else {
        ties++;
    }
    mtx.unlock();
}

int main() {
    srand(time(NULL));
    for (int i = 0; i < 100; i+= 10) {
        std::cout << "running match " << i << "\n";
        std::vector<std::thread> threads;
        for (int j = 0; j < 10; j++) {
            threads.push_back(std::thread(run_match));
        }
        for (auto &thread : threads) {
            thread.join();
        }
    }
    std::cout << "red wins: " << red_wins << " blue wins: " << blue_wins << " ties: " << ties << "\n";
}