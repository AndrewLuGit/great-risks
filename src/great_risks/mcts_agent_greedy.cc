#include "mcts_agent_greedy.hh"

#include <algorithm>
#include <thread>

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
    };

    void mcts_thread(Node *root, size_t iterations, GreedyAgent &self_greedy, GreedyAgent &opp_greedy, std::mt19937 &rng, tsl::robin_map<Field, float> &rollout_cache, uint8_t index, uint8_t opp_index, std::mutex &mtx) {
        bool is_red = root->state.robots[index].is_red;
        Node *nodes = new Node[iterations];
        for (size_t i = 0; i < iterations; i++)
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
                Node *child = &nodes[i];
                child->wins = 0;
                child->total = 0;
                // do agent action
                child->state = node->state;
                child->action = node->unexplored_actions.back();
                node->unexplored_actions.pop_back();
                child->state.perform_action(index, child->action);
                // do opponent action
                Action opp_action = opp_greedy.next_action(child->state);
                child->state.perform_action(opp_index, opp_action);
                // decrement time
                child->state.time_remaining--;
                child->parent = node;
                node->children.push_back(child);
                child->unexplored_actions = child->state.legal_actions(index);
                mtx.lock();
                std::shuffle(child->unexplored_actions.begin(), child->unexplored_actions.end(), rng);
                mtx.unlock();
                node = child;
            }
            // rollout
            Field rollout = node->state;
            float reward = 0;
            mtx.lock();
            auto cached = rollout_cache.find(rollout);
            bool is_cached = cached != rollout_cache.end();
            mtx.unlock();
            if (is_cached)
            {
                reward = cached->second;
            }
            else
            {
                std::vector<Field> rollouts;
                rollouts.reserve(rollout.time_remaining + 1);
                rollouts.emplace_back(rollout);
                while (rollout.time_remaining > 0)
                {
                    Action self_action = self_greedy.next_action(rollout);
                    rollout.perform_action(index, self_action);
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
                mtx.lock();
                for (const auto &rollout : rollouts) {
                    rollout_cache.insert_or_assign(rollout, reward);
                }
                mtx.unlock();
                //rollout_cache.insert_or_assign(node->state, reward);
            }
            // backpropagation
            while (node != root->parent)
            {
                node->total++;
                node->wins += reward;
                node = node->parent;
            }
        }
        delete[] nodes;
    }

    Action MCTSAgentGreedy::next_action(Field field)
    {
        Node root;
        root.wins = 0;
        root.total = 0;
        root.state = field;
        //bool is_red = field.robots[robot_index].is_red;
        root.parent = nullptr;
        root.unexplored_actions = field.legal_actions(robot_index);
        size_t num_threads = root.unexplored_actions.size();
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        while (!root.unexplored_actions.empty()) {
            Node *child = new Node();
            child->wins = 0;
            child->total = 0;
            // do agent action
            child->state = root.state;
            child->action = root.unexplored_actions.back();
            root.unexplored_actions.pop_back();
            child->state.perform_action(robot_index, child->action);
            // do opponent action
            Action opp_action = opp_greedy.next_action(child->state);
            child->state.perform_action(opp_index, opp_action);
            // decrement time
            child->state.time_remaining--;
            child->parent = &root;
            root.children.push_back(child);
            child->unexplored_actions = child->state.legal_actions(robot_index);
            std::shuffle(child->unexplored_actions.begin(), child->unexplored_actions.end(), rng);
            threads.emplace_back(mcts_thread, child, NUM_ITERATIONS / num_threads, std::ref(self_greedy), std::ref(opp_greedy), std::ref(rng), std::ref(rollout_cache), robot_index, opp_index, std::ref(mtx));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        Action selected_action = root.children[0]->action;
        double highest_win_rate = 0.0;
        for (const Node *const&child : root.children)
        {
            double win_rate = child->wins / child->total;
            if (win_rate > highest_win_rate)
            {
                highest_win_rate = win_rate;
                selected_action = child->action;
            }
        }
        for (auto &child : root.children) {
            delete child;
        }
        return selected_action;
    }
}  // namespace great_risks