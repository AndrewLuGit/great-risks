#pragma once

#include "agent.hh"

namespace great_risks {
    class GreedyAgent: public Agent {
        using Agent::Agent;
        public:
            Action next_action(Field field) override;
    };
}