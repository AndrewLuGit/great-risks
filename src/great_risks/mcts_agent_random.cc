#include "mcts_agent_random.hh"

#include <algorithm>
#include <iostream>
#include <queue>

constexpr size_t NUM_ITERATIONS = 10000;
constexpr float EXPLORATION_PARAM = 1.41421;

namespace great_risks
{
    class Node
    {
    public:
        double wins;
        int total;
        Field state;
        Action action;
        uint8_t robot_index;
        Node *parent;
        std::vector<Node *> children;
        std::vector<Action> unexplored_actions;
/*
        ~Node()
        {
            for (auto &node : children)
            {
                delete node;
            }
        }
        */
    };

    Action MCTSAgentRandom::next_action(Field field)
    {
        std::array<Node, NUM_ITERATIONS + 1> nodes;
        Node *root = &nodes[0];
        root->wins = 0;
        root->total = 0;
        root->state = field;
        root->robot_index = robot_index;
        root->parent = nullptr;
        root->unexplored_actions = field.legal_actions(robot_index);
        std::shuffle(root->unexplored_actions.begin(), root->unexplored_actions.end(), rng);
        for (size_t i = 0; i < NUM_ITERATIONS; i++)
        {
            // selection: stop when node is not fully explored or it is terminal
            Node *node = root;
            while (node->unexplored_actions.empty() && node->state.time_remaining > 0)
            {
                double best_score = 0.0;
                Node *best_child = node->children.front();
                for (size_t i = 0; i < node->children.size(); i++)
                {
                    Node *child = node->children[i];
                    double score = child->wins / child->total +
                                   EXPLORATION_PARAM * sqrt(log(node->total) / child->total);
                    if (score > best_score)
                    {
                        best_score = score;
                        best_child = child;
                    }
                }
                node = best_child;
            }
            // expansion
            if (node->state.time_remaining > 0)
            {
                Node *child = &nodes[i + 1];
                child->wins = 0;
                child->total = 0;
                child->state = node->state;
                child->action = node->unexplored_actions.back();
                child->state.perform_action(node->robot_index, child->action);
                node->unexplored_actions.pop_back();
                child->robot_index = (node->robot_index + 1) % node->state.robots.size();
                child->unexplored_actions = child->state.legal_actions(child->robot_index);
                std::shuffle(child->unexplored_actions.begin(), child->unexplored_actions.end(), rng);
                if (child->robot_index == 0)
                {
                    child->state.time_remaining--;
                }
                child->parent = node;
                node->children.push_back(child);
                node = child;
            }
            // rollout
            Field rollout = node->state;
            uint8_t index = node->robot_index;
            while (rollout.time_remaining > 0)
            {
                auto legal_actions = rollout.legal_actions(index);
                std::vector<uint8_t> weights;
                uint8_t sum_weights = 0;
                for (auto &action : legal_actions) {
                    if (action == GRAB_MOBILE_GOAL || action == SCORE_MOBILE_GOAL || action == SCORE_WALL_STAKE) {
                        weights.push_back(2);
                    } else if ((rollout.robots[index].is_red && action == PICK_UP_RED) || (!rollout.robots[index].is_red && action == PICK_UP_BLUE)) {
                        weights.push_back(2);
                    } else {
                        weights.push_back(1);
                    }
                    sum_weights += weights.back();
                }
                std::uniform_int_distribution<uint32_t> uniform_dist(0, sum_weights - 1);
                auto rand = uniform_dist(rng);
                Action chosen_action = legal_actions.back();
                for (size_t i = 0; i < weights.size(); i++) {
                    if (rand < weights[i]) {
                        chosen_action = legal_actions[i];
                        break;
                    }
                    rand -= weights[i];
                }
                rollout.perform_action(index, chosen_action);
                index = (index + 1) % rollout.robots.size();
                if (index == 0)
                {
                    rollout.time_remaining--;
                }
            }
            auto [red_score, blue_score] = rollout.calculate_scores();
            double red_reward = 0;
            if (red_score > blue_score)
            {
                red_reward = 1;
            }
            else if (red_score == blue_score)
            {
                red_reward = 0.5;
            }
            // backpropagation
            while (node)
            {
                node->total++;
                if (node->state.robots[node->robot_index].is_red)
                {
                    node->wins += red_reward;
                }
                else
                {
                    node->wins += 1 - red_reward;
                }
                node = node->parent;
            }
        }
        Action selected_action = root->children[0]->action;
        double highest_win_rate = 0.0;
        for (Node *&child : root->children)
        {
            double win_rate = 1 - (child->wins / child->total);
            if (win_rate > highest_win_rate)
            {
                highest_win_rate = win_rate;
                selected_action = child->action;
            }
        }
        /*
        std::queue<Node*> bfs_queue;
        bfs_queue.push(root);
        while (!bfs_queue.empty()) {
            Node* node = bfs_queue.front();
            bfs_queue.pop();
            std::cerr << "wins: " << node->wins << " total: " << node->total << " Action: " << node->action << " index: " << static_cast<int>(node->robot_index) << "\n";
            for (auto &child : node->children) {
                bfs_queue.push(child);
            }
        }
        */
        return selected_action;
    }
}  // namespace great_risks