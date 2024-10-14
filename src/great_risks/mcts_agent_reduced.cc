#include "mcts_agent_reduced.hh"

#include <cmath>
#include <iostream>

#define NUM_ITERATIONS 225
#define EXPLORATION_PARAM 1.41421

namespace great_risks {
class Node {
public:
    double wins;
    int total;
    ReducedField state;
    Action action;
    bool is_opp;
    Node *parent;
    std::vector<Node*> children;
    std::vector<Action> unexplored_actions;
    ~Node() {
        for (Node *node : children) {
            delete node;
        }
    }
};
    Action MCTSAgentReduced::next_action(ReducedField field) {
        Node *root = new Node();
        root->wins = 0;
        root->total = 0;
        root->state = field;
        root->is_opp = false;
        root->parent = nullptr;
        root->unexplored_actions = field.legal_actions(robot_index);
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            // selection: stop when node is not fully explored or it is terminal
            Node *node = root;
            while (node->unexplored_actions.empty() && node->state.time_remaining > 0) {
                double best_score = 0.0;
                std::vector<size_t> best_score_indices;
                for (size_t i = 0; i < node->children.size(); i++) {
                    Node *child = node->children[i];
                    double score = child->wins / child->total + EXPLORATION_PARAM * sqrt(log(node->total) / child->total);
                    if (score > best_score) {
                        best_score_indices.clear();
                        best_score_indices.push_back(i);
                    } else if (score == best_score) {
                        best_score_indices.push_back(i);
                    }
                }
                std::uniform_int_distribution<uint32_t> select_index(0, node->children.size() - 1);
                node = node->children[select_index(rng)];
            }
            // expansion when non-terminal
            if (node->state.time_remaining > 0) {
                Node *child = new Node();
                child->wins = 0;
                child->total = 0;
                // do agent action
                std::uniform_int_distribution<uint32_t> uniform_dist(0, node->unexplored_actions.size() - 1);
                auto selected_index = uniform_dist(rng);
                child->action = node->unexplored_actions[selected_index];
                child->state = node->state.perform_action(robot_index, child->action);
                node->unexplored_actions.erase(node->unexplored_actions.begin() + selected_index);
                // do opponent action
                child->state = child->state.perform_action(opp_index, greedy.next_action(child->state));
                // decrement time
                child->state.time_remaining--;
                child->parent = node;
                node->children.push_back(child);
                child->unexplored_actions = child->state.legal_actions(robot_index);
                node = child;
            }
            // rollout
            ReducedField rollout = node->state;
            GreedyAgentReduced self_greedy(robot_index);
            while (rollout.time_remaining > 0) {
                rollout = rollout.perform_action(robot_index, self_greedy.next_action(rollout));
                rollout = rollout.perform_action(opp_index, greedy.next_action(rollout));
                rollout.time_remaining--;
            }
            auto [score, opp_score] = rollout.calculate_scores();
            double reward = 0;
            if (score > opp_score) {
                reward = 1;
            } else if (score == opp_score) {
                reward = 0.5;
            }
            // backpropagation
            while (node) {
                node->total++;
                node->wins += reward;
                node = node->parent;
            }
        }
        Action selected_action = root->children[0]->action;
        double highest_win_rate = 0.0;
        for (Node* &child : root->children) {
            double win_rate = child->wins / child->total;
            if (win_rate > highest_win_rate) {
                highest_win_rate = win_rate;
                selected_action = child->action;
            }
        }
        delete root;
        return selected_action;
    }
}