#include "mcts_agent_greedy.hh"

#define NUM_ITERATIONS 10000
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
        Node *parent;
        std::vector<Node *> children;
        std::vector<Action> unexplored_actions;

        ~Node()
        {
            for (Node *node : children)
            {
                delete node;
            }
        }
    };

    Action MCTSAgentGreedy::next_action(Field field)
    {
        Node *root = new Node();
        root->wins = 0;
        root->total = 0;
        root->state = field;
        bool is_red = field.robots[robot_index].is_red;
        root->parent = nullptr;
        root->unexplored_actions = field.legal_actions(robot_index);
        for (int i = 0; i < NUM_ITERATIONS; i++)
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
            // expansion when non-terminal
            if (node->state.time_remaining > 0)
            {
                Node *child = new Node();
                child->wins = 0;
                child->total = 0;
                // do agent action
                std::uniform_int_distribution<uint32_t> uniform_dist(0, node->unexplored_actions.size() - 1);
                auto selected_index = uniform_dist(rng);
                child->state = node->state;
                child->action = node->unexplored_actions[selected_index];
                child->state.perform_action(robot_index, child->action);
                node->unexplored_actions.erase(node->unexplored_actions.begin() + selected_index);
                // do opponent action
                Action opp_action = opp_greedy.next_action(child->state);
                child->state.perform_action(opp_index, opp_action);
                // decrement time
                child->state.time_remaining--;
                child->parent = node;
                node->children.push_back(child);
                child->unexplored_actions = child->state.legal_actions(robot_index);
                node = child;
            }
            // rollout
            Field rollout = node->state;
            double reward = 0;
            if (rollout_cache.find(rollout) != rollout_cache.end())
            {
                reward = rollout_cache.at(rollout);
            }
            else
            {
                GreedyAgent self_greedy(robot_index);
                while (rollout.time_remaining > 0)
                {
                    Action self_action = self_greedy.next_action(rollout);
                    rollout.perform_action(robot_index, self_action);
                    Action opp_action = opp_greedy.next_action(rollout);
                    rollout.perform_action(opp_index, opp_action);
                    rollout.time_remaining--;
                }
                auto [red_score, blue_score] = rollout.calculate_scores();
                if ((is_red && red_score > blue_score) || (!is_red && blue_score > red_score))
                {
                    reward = 1;
                }
                else if (blue_score == red_score)
                {
                    reward = 0.5;
                }
                rollout_cache.insert_or_assign(node->state, reward);
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