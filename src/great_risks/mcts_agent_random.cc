#include "mcts_agent_random.hh"

#define NUM_ITERATIONS 100000
#define EXPLORATION_PARAM 1.41421

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

        ~Node()
        {
            for (auto &node : children)
            {
                delete node;
            }
        }
    };

    Action MCTSAgentRandom::next_action(Field field)
    {
        Node *root = new Node();
        root->wins = 0;
        root->total = 0;
        root->state = field;
        root->robot_index = robot_index;
        root->parent = nullptr;
        root->unexplored_actions = field.legal_actions(robot_index);
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
                Node *child = new Node();
                child->wins = 0;
                child->total = 0;
                std::uniform_int_distribution<uint32_t> uniform_dist(0, node->unexplored_actions.size() - 1);
                auto selected_index = uniform_dist(rng);
                child->state = node->state;
                child->action = node->unexplored_actions[selected_index];
                child->state.perform_action(node->robot_index, child->action);
                node->unexplored_actions.erase(node->unexplored_actions.begin() + selected_index);
                child->robot_index = (node->robot_index + 1) % node->state.robots.size();
                child->unexplored_actions = child->state.legal_actions(child->robot_index);
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
                std::uniform_int_distribution<uint32_t> uniform_dist(0, legal_actions.size() - 1);
                rollout.perform_action(index, legal_actions[uniform_dist(rng)]);
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
            double win_rate = child->wins / child->total;
            if (win_rate > highest_win_rate)
            {
                highest_win_rate = win_rate;
                selected_action = child->action;
            }
        }
        delete root;
        return selected_action;
    }
}  // namespace great_risks