#pragma once

#include "reduced_game.hh"

namespace great_risks {
    class GreedyAgentReduced: public ReducedAgent {
        using ReducedAgent::ReducedAgent;
        public:
            Action next_action(ReducedField field) override;
    };
}