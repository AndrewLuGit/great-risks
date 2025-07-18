#pragma once

#include "reduced_game.hh"
#include "greedy_agent_reduced.hh"

#include <random>
#include <unordered_map>

namespace great_risks
{
    class MCTSAgentReduced : public ReducedAgent
    {
    private:
        GreedyAgentReduced greedy;
        uint8_t opp_index;
        std::mt19937 rng;
        std::unordered_map<ReducedField, double> rollout_cache;

    public:
        MCTSAgentReduced(uint8_t index, uint8_t opp_index, uint32_t seed = 5489)
          : ReducedAgent(index), greedy(opp_index), opp_index(opp_index)
        {
            rng.seed(seed);
        };

        Action next_action(ReducedField field) override;
    };
}  // namespace great_risks