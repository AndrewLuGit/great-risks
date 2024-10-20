#pragma once

#include "simulator.hh"

namespace great_risks
{
    class Agent
    {
    protected:
        std::uint8_t robot_index;

    public:
        Agent(std::uint8_t robot_index) : robot_index(robot_index) {};
        virtual Action next_action(Field field) = 0;
    };
}  // namespace great_risks