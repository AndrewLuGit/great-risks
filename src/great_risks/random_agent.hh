#pragma once

#include "agent.hh"

namespace great_risks
{
    class RandomAgent : public Agent
    {
        using Agent::Agent;

    public:
        Action next_action(Field field) override;
    };
}  // namespace great_risks