#pragma once

#include "agent.hh"

#include <random>
#include <tsl/robin_map.h>

namespace great_risks
{
    class MCTSAgentRandom : public Agent
    {
    private:
        std::mt19937 rng;
        std::array<std::unordered_map<Field, int>, 2> rollout_cache;

    public:
        MCTSAgentRandom(uint8_t index, uint32_t seed = 5489) : Agent(index), rng(seed) {};
        ~MCTSAgentRandom() override = default;
        Action next_action(Field field) override;
    };
}  // namespace great_risks