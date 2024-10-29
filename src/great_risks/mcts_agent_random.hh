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
        std::array<tsl::robin_map<Field, int>, 4> rollout_cache;

    public:
        MCTSAgentRandom(uint8_t index, uint32_t seed = 5489) : Agent(index), rng(seed) {};
        Action next_action(Field field) override;
    };
}  // namespace great_risks