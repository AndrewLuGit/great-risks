from core.types import StepMetadata

import jax
import jax.numpy as jnp

from flax import struct

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

class AgentState(struct.PyTreeNode):
    position: jax.Array = struct.field(default_factory=lambda: jnp.int32([0, 0]))
    has_goal: jax.Array = struct.field(default_factory=lambda: FALSE)
    rings: jax.Array = struct.field(default_factory=lambda: jnp.int32(0))

@struct.dataclass
class State:
    current_player: jax.Array = struct.field(default_factory=lambda: jnp.int32(0))
    player_states: list[AgentState] = struct.field(default_factory=lambda: [AgentState(), AgentState()])
    rings: jax.Array = struct.field(default_factory=lambda: jnp.zeros(25, dtype=jnp.int32))
    goals: jax.Array = struct.field(default_factory=lambda: jnp.zeros(25, dtype=jnp.int32))
    step_count: jax.Array = struct.field(default_factory=lambda: jnp.int32(0))

INIT_RINGS = jnp.int32(
    [2, 0, 1, 0, 2,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     2, 0, 1, 0, 2])

class Actions:
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    GRAB_GOAL = 4
    PICK_UP_RING = 5

def legal_actions(state: State):
    player_state = state.player_states[state.current_player]
    position = 5 * player_state[0] + player_state[1]
    legal_action_mask = jnp.zeros(6, dtype=jnp.bool_)
    if player_state.position[0] > 0:
        legal_action_mask = legal_action_mask.at[Actions.MOVE_NORTH].set(TRUE)
    if player_state.position[0] < 4:
        legal_action_mask = legal_action_mask.at[Actions.MOVE_SOUTH].set(TRUE)
    if player_state.position[1] > 0:
        legal_action_mask = legal_action_mask.at[Actions.MOVE_WEST].set(TRUE)
    if player_state.position[1] < 4:
        legal_action_mask = legal_action_mask.at[Actions.MOVE_EAST].set(TRUE)
    if ~player_state.has_goal and state.goals[position] != 0:
        legal_action_mask = legal_action_mask.at[Actions.GRAB_GOAL].set(TRUE)
    if player_state.rings < 6 and state.rings[position] != 0:
        legal_action_mask = legal_action_mask.at[Actions.PICK_UP_RING].set(TRUE)
    return legal_action_mask

def reward(state: State):
    if state.step_count < 60:
        return False, jnp.float32([0.0, 0.0])
    elif player_states[0].rings > player_states[1].rings:
        return True, jnp.float32([1.0, 0.0])
    elif player_states[1].rings > player_states[0].rings:
        return True, jnp.float32([0.0, 1.0])
    return True, jnp.float32([0.5, 0.5])

def init(key):
    init_state = State(
        current_player=jnp.int32(jax.random.bernoulli(key)),
        player_states=[AgentState(position=jnp.int32([2, 0])), AgentState(position=jnp.int32([2, 4]))],
        rings=INIT_RINGS,
        goals=jnp.zeros(25, dtype=jnp.int32).at[7].set(1).at[12].set(1).at[17].set(1)
    )
    terminated, rewards = reward(init_state)
    legal_action_mask = legal_actions(init_state)
    return init_state, StepMetadata(rewards, legal_action_mask, terminated, init_state.current_player, init_state.step_count)

MOVE_MAP = jnp.array(
    (
        (-1, 0),
        (1, 0),
        (0, 1),
        (0, -1)
    )
)

#def step(state: State, action):
#    if action <= Actions.MOVE_WEST:


state = State()
player_states = state.player_states
player_states[1] = player_states[1].replace(has_goal=TRUE)
state_b = state.replace(current_player=1-state.current_player, player_states=player_states)
print(repr(state))
print(repr(state_b))
