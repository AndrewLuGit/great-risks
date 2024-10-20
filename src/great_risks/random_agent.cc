#include "random_agent.hh"

#include <cstdlib>

namespace great_risks
{
    Action RandomAgent::next_action(Field Field)
    {
        auto actions = Field.legal_actions(robot_index);
        return actions[rand() % actions.size()];
    }
}  // namespace great_risks