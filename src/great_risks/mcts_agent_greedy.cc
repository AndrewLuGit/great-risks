#include "mcts_agent_greedy.hh"

#include <algorithm>

constexpr int NUM_ITERATIONS = 10000;
const float EXPLORATION_PARAM = sqrt(2);

namespace great_risks
{
    class Node
    {
    public:
        float wins;
        int total;
        Field state;
        Action action;
        Node *parent;
        std::vector<Node *> children;
        std::vector<Action> unexplored_actions;
/*
        ~Node()
        {
            for (Node *node : children)
            {
                delete node;
            }
        }*/
    };

    Action MCTSAgentGreedy::next_action(Field field)
    {
        std::array<Node, NUM_ITERATIONS + 1> nodes;
        Node *root = &nodes[0];
        root->wins = 0;
        root->total = 0;
        root->state = field;
        bool is_red = field.robots[robot_index].is_red;
        root->parent = nullptr;
        root->unexplored_actions = field.legal_actions(robot_index);
        std::shuffle(root->unexplored_actions.begin(), root->unexplored_actions.end(), rng);
        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            // selection: stop when node is not fully explored or it is terminal
            Node *node = root;
            while (node->unexplored_actions.empty() && node->state.time_remaining > 0)
            {
                float best_score = 0.0;
                Node *best_child = node->children.front();
                for (size_t i = 0; i < node->children.size(); i++)
                {
                    Node *child = node->children[i];
                    float score = child->wins / child->total +
                                   EXPLORATION_PARAM * sqrt(log(node->total) / child->total);
                    if (score > best_score)
                    {
                        best_score = score;
                        best_child = child;
                    }
                }
                node = best_child;
            }
            // expansion when non-terminal
            if (node->state.time_remaining > 0)
            {
                Node *child = &nodes[i + 1];
                child->wins = 0;
                child->total = 0;
                // do agent action
                child->state = node->state;
                child->action = node->unexplored_actions.back();
                node->unexplored_actions.pop_back();
                child->state.perform_action(robot_index, child->action);
                // do opponent action
                Action opp_action = opp_greedy.next_action(child->state);
                child->state.perform_action(opp_index, opp_action);
                // decrement time
                child->state.time_remaining--;
                child->parent = node;
                node->children.push_back(child);
                child->unexplored_actions = child->state.legal_actions(robot_index);
                std::shuffle(child->unexplored_actions.begin(), child->unexplored_actions.end(), rng);
                node = child;
            }
            // rollout
            Field rollout = node->state;
            float reward = 0;
            if (rollout_cache.find(rollout) != rollout_cache.end())
            {
                reward = rollout_cache.at(rollout);
            }
            else
            {
                GreedyAgent self_greedy(robot_index);
                std::vector<Field> rollouts;
                rollouts.reserve(rollout.time_remaining + 1);
                rollouts.emplace_back(rollout);
                while (rollout.time_remaining > 0)
                {
                    Action self_action = self_greedy.next_action(rollout);
                    rollout.perform_action(robot_index, self_action);
                    Action opp_action = opp_greedy.next_action(rollout);
                    rollout.perform_action(opp_index, opp_action);
                    rollout.time_remaining--;
                    rollouts.emplace_back(rollout);
                }
                auto [red_score, blue_score] = rollout.calculate_scores();
                if (is_red) {
                    reward = 1 - exp(0.1 * (blue_score - red_score));
                    if (reward < 0) reward = 0;
                } else {
                    reward = 1 - exp(0.1 * (red_score - blue_score));
                    if (reward < 0) reward = 0;
                }
                for (const auto &rollout : rollouts) {
                    rollout_cache.insert_or_assign(rollout, reward);
                }
                //rollout_cache.insert_or_assign(node->state, reward);
            }
            // backpropagation
            while (node)
            {
                node->total++;
                node->wins += reward;
                node = node->parent;
            }
        }
        Action selected_action = root->children[0]->action;
        double highest_win_rate = 0.0;
        for (const Node *const&child : root->children)
        {
            double win_rate = child->wins / child->total;
            if (win_rate > highest_win_rate)
            {
                highest_win_rate = win_rate;
                selected_action = child->action;
            }
        }
        return selected_action;
    }
}  // namespace great_risks