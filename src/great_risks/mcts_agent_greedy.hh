#pragma once

#include "greedy_agent.hh"

#include <mutex>
#include <random>
#include <unordered_map>
#include <tsl/robin_map.h>

namespace great_risks
{
    class MCTSAgentGreedy : public Agent
    {
    private:
        GreedyAgent self_greedy;
        GreedyAgent opp_greedy;
        uint8_t opp_index;
        std::mt19937 rng;
        tsl::robin_map<Field, float> rollout_cache;
        std::mutex mtx;

    public:
        MCTSAgentGreedy(uint8_t index, uint8_t opp_index, uint32_t seed = 5489)
          : Agent(index), self_greedy(index), opp_greedy(opp_index), opp_index(opp_index)
        {
            rng.seed(seed);
        }
        ~MCTSAgentGreedy() override = default;

        Action next_action(Field field) override;
    };
}  // namespace great_risks