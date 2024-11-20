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
    player_states: jax.Array = struct.field(default_factory=lambda: jnp.zeros((2,4), dtype=jnp.int32))
    rings: jax.Array = struct.field(default_factory=lambda: jnp.zeros(25, dtype=jnp.int32))
    goals: jax.Array = struct.field(default_factory=lambda: jnp.zeros(25, dtype=jnp.int32))
    step_count: jax.Array = struct.field(default_factory=lambda: jnp.int32(0))

INIT_RINGS = jnp.int32(
    [2, 0, 1, 0, 2,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     2, 0, 1, 0, 2])

INIT_PLAYERS = jnp.int32([[2, 0, 0, 0], [2, 4, 0, 0]])

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
    legal_action_mask = jnp.zeros(6, dtype=jnp.bool_)\
        .at[Actions.MOVE_NORTH].set(player_state[0] > 0)\
        .at[Actions.MOVE_SOUTH].set(player_state[0] < 4)\
        .at[Actions.MOVE_WEST].set(player_state[1] > 0)\
        .at[Actions.MOVE_EAST].set(player_state[1] < 4)\
        .at[Actions.GRAB_GOAL].set((player_state[2] == 0) & (state.goals[position] != 0))\
        .at[Actions.PICK_UP_RING].set((player_state[2] == 1) & (player_state[3] < 6) & (state.rings[position] != 0))
    return legal_action_mask

def reward(state: State):
    terminal = state.step_count >= 60
    player_0_rings = state.player_states[0][3]
    player_1_rings = state.player_states[1][3]
    winner = jax.lax.select(player_0_rings > player_1_rings, 0, 1)
    rewards = jax.lax.select(player_0_rings == player_1_rings, jnp.zeros(2, dtype=jnp.float32), jnp.float32([-1, -1]).at[winner].set(1))
    return terminal, rewards

def move(state: State, direction):
    player_position = state.player_states[state.current_player][0:2]
    new_position = jnp.clip(player_position + direction, min=jnp.array((0, 0)), max=jnp.array((4, 4)))
    new_position = jax.lax.select(jnp.all(new_position == state.player_states[1-state.current_player][0:2]), player_position, new_position)
    new_state = state.player_states[state.current_player].at[0:2].set(new_position)
    return state.replace(player_states=state.player_states.at[state.current_player].set(new_state))

def grab_goal(state: State):
    position = 5 * state.player_states[state.current_player][0] + state.player_states[state.current_player][1]
    new_goals = state.goals.at[position].set(0)
    new_state = state.player_states[state.current_player].at[2].set(1)
    return state.replace(player_states=state.player_states.at[state.current_player].set(new_state), goals=new_goals)

def pick_up_ring(state: State):
    position = 5 * state.player_states[state.current_player][0] + state.player_states[state.current_player][1]
    new_rings = state.rings.at[position].set(state.rings[position] - 1)
    new_state = state.player_states[state.current_player].at[3].set(state.player_states[state.current_player][3] + 1)
    return state.replace(player_states=state.player_states.at[state.current_player].set(new_state), rings=new_rings)

def step(state: State, action):
    actions = (
        lambda: move(state, jnp.array((-1, 0))),
        lambda: move(state, jnp.array((1, 0))),
        lambda: move(state, jnp.array((0, 1))),
        lambda: move(state, jnp.array((0, -1))),
        lambda: grab_goal(state),
        lambda: pick_up_ring(state)
    )
    new_state = jax.lax.switch(action, actions)
    new_state = new_state.replace(current_player=1-state.current_player, step_count=state.step_count+1)
    terminal, rewards = reward(new_state)
    legal_action_mask = legal_actions(new_state)
    return new_state, StepMetadata(rewards=rewards, action_mask=legal_action_mask, terminated=terminal, cur_player_id=new_state.current_player, step=new_state.step_count)

def observe(state: State):
    ring_grid = state.rings.reshape((5, 5))
    goal_grid = state.goals.reshape((5, 5))
    player_0_state = 1 + state.player_states[0][2] + state.player_states[0][3]
    player_1_state = -1 - state.player_states[1][2] - state.player_states[1][3]
    multiplier = jax.lax.select(state.current_player == 0, 1, -1)
    player_grid= jnp.zeros((5, 5), dtype=jnp.int32).at[state.player_states[0][0], state.player_states[0][1]].set(multiplier * player_0_state).at[state.player_states[1][0], state.player_states[1][1]].set(multiplier * player_1_state)
    return jnp.stack((ring_grid, goal_grid, player_grid))

def init(key):
    init_state = State(
        current_player=jnp.int32(jax.random.bernoulli(key)),
        player_states=INIT_PLAYERS,
        rings=INIT_RINGS,
        goals=jnp.zeros(25, dtype=jnp.int32).at[7].set(1).at[12].set(1).at[17].set(1)
    )
    terminated, rewards = reward(init_state)
    legal_action_mask = legal_actions(init_state)
    return init_state, StepMetadata(rewards=rewards, action_mask=legal_action_mask, terminated=terminated, cur_player_id=init_state.current_player, step=init_state.step_count)

def render_text(frames, p_ids, title, frame_dir):
    trained_agent_id = p_ids[0]
    with open(f"{frame_dir}/{title}.txt", "w") as file:
        for frame in frames:
            state = frame.env_state
            corrected_frame = state.replace(current_player=trained_agent_id)
            file.write(str(observe(corrected_frame)[2]))
            file.write("\n")
    return f"{frame_dir}/{title}.txt"
