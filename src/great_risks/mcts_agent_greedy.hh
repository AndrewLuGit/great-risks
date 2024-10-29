#pragma once

#include "greedy_agent.hh"

#include <random>
#include <unordered_map>
#include <tsl/robin_map.h>

namespace great_risks
{
    class MCTSAgentGreedy : public Agent
    {
    private:
        GreedyAgent opp_greedy;
        uint8_t opp_index;
        std::mt19937 rng;
        tsl::robin_map<Field, float> rollout_cache;

    public:
        MCTSAgentGreedy(uint8_t index, uint8_t opp_index, uint32_t seed = 5489)
          : Agent(index), opp_greedy(opp_index), opp_index(opp_index)
        {
            rng.seed(seed);
        }

        Action next_action(Field field) override;
    };
}  // namespace great_risks