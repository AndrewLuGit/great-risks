from core.types import StepMetadata

import jax
import jax.numpy as jnp

from flax import struct

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

# player state: [x, y, goal, rings]
# goal state: [x, y, red rings, blue rings, top ring]

@struct.dataclass
class State:
    current_player: jax.Array = struct.field(default_factory=lambda: jnp.int32(0))
    player_states: jax.Array = struct.field(default_factory=lambda: jnp.zeros((2, 4), dtype=jnp.int32))
    rings: jax.Array = struct.field(default_factory=lambda: jnp.zeros((2, 25), dtype=jnp.int32))
    goals: jax.Array = struct.field(default_factory=lambda: jnp.zeros((3, 5), dtype=jnp.int32))
    goal_grid: jax.Array = struct.field(default_factory=lambda: jnp.zeros(25, dtype=jnp.int32))
    step_count: jax.Array = struct.field(default_factory=lambda: jnp.int32(0))

INIT_RINGS = jnp.int32(
    [[2, 0, 1, 0, 2,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     2, 0, 1, 0, 2],
    [2, 0, 1, 0, 2,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     2, 0, 1, 0, 2]])

INIT_PLAYERS = jnp.int32([[2, 0, -1, 0], [2, 4, -1, 0]])

INIT_GOALS = jnp.int32([[1, 2, 0, 0, -1], [2, 2, 0, 0, -1], [3, 2, 0, 0, -1]])

INIT_GOAL_GRID = jnp.int32(
    [0, 0, 0, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 2, 0, 0,
     0, 0, 3, 0, 0,
     0, 0, 0, 0, 0])

TOP_RING_MAP = jnp.int32([1, -1, 0])

class Actions:
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    GRAB_GOAL = 4
    PICK_UP_RING = 5
    RELEASE_GOAL = 6

def legal_actions(state: State):
    player_state = state.player_states[state.current_player]
    position = 5 * player_state[0] + player_state[1]
    legal_action_mask = jnp.zeros(7, dtype=jnp.bool_)\
        .at[Actions.MOVE_NORTH].set(player_state[0] > 0)\
        .at[Actions.MOVE_SOUTH].set(player_state[0] < 4)\
        .at[Actions.MOVE_WEST].set(player_state[1] > 0)\
        .at[Actions.MOVE_EAST].set(player_state[1] < 4)\
        .at[Actions.GRAB_GOAL].set((player_state[2] == -1) & (state.goal_grid[position] != 0))\
        .at[Actions.PICK_UP_RING].set((player_state[2] != -1) & (player_state[3] < 6) & (state.rings[state.current_player][position] != 0))\
        .at[Actions.RELEASE_GOAL].set((player_state[2] != -1) & (state.goal_grid[position] == 0))
    return legal_action_mask

def goal_scores(goal):
    reward = goal[2:4] + jnp.zeros(2, dtype=jnp.int32).at[goal[4]].set(jax.lax.select(goal[4] == -1, 0, 2))
    on_edge = (goal[1] == 0) | (goal[1] == 4)
    in_positive_corner = (goal[0] == 4) & on_edge
    in_negative_corner = (goal[0] == 0) & on_edge
    multiplier = jax.lax.select(in_positive_corner, 2, 1)
    multiplier_2 = jax.lax.select(in_negative_corner, -1, multiplier)
    return multiplier_2 * reward

def scores(state: State):
    return jnp.sum(jax.vmap(goal_scores)(state.goals), axis=0)

def reward(state: State):
    terminal = state.step_count >= 60
    player_scores = scores(state)
    player_0_rings = player_scores[0]
    player_1_rings = player_scores[1]
    winner = jax.lax.select(player_0_rings > player_1_rings, 0, 1)
    rewards = jax.lax.select(player_0_rings == player_1_rings, jnp.zeros(2, dtype=jnp.float32), jnp.float32([-1, -1]).at[winner].set(1))
    return terminal, rewards

def move(state: State, direction):
    player_position = state.player_states[state.current_player][0:2]
    new_position = jnp.clip(player_position + direction, min=jnp.array((0, 0)), max=jnp.array((4, 4)))
    new_position = jax.lax.select(jnp.all(new_position == state.player_states[1-state.current_player][0:2]), player_position, new_position)
    new_states = state.player_states.at[state.current_player, 0:2].set(new_position)
    goal_index = state.player_states[state.current_player, 2]
    new_goals = jax.lax.select(goal_index == -1, state.goals, state.goals.at[goal_index, 0:2].set(new_position))
    return state.replace(player_states=new_states, goals=new_goals)

def grab_goal(state: State):
    position = 5 * state.player_states[state.current_player][0] + state.player_states[state.current_player][1]
    goal_index = state.goal_grid[position] - 1
    new_goal_grid = state.goal_grid.at[position].set(0)
    new_states = state.player_states.at[state.current_player, 2].set(goal_index)
    return state.replace(player_states=new_states, goal_grid=new_goal_grid)

def pick_up_ring(state: State):
    position = 5 * state.player_states[state.current_player, 0] + state.player_states[state.current_player, 1]
    new_rings = state.rings.at[state.current_player, position].set(state.rings[state.current_player, position] - 1)
    new_state = state.player_states.at[state.current_player, 3].set(state.player_states[state.current_player, 3] + 1)
    goal_index = state.player_states[state.current_player, 2]
    new_goals = state.goals.at[goal_index, 2 + state.current_player].set(state.goals[goal_index, 2 + state.current_player] + 1).at[goal_index, 4].set(state.current_player)
    return state.replace(player_states=new_state, rings=new_rings, goals=new_goals)

def release_goal(state: State):
    position = 5 * state.player_states[state.current_player][0] + state.player_states[state.current_player][1]
    new_goal_grid = state.goal_grid.at[position].set(state.player_states[state.current_player, 2] + 1)
    new_states = state.player_states.at[state.current_player, 2].set(-1).at[state.current_player, 3].set(0)
    return state.replace(player_states=new_states, goal_grid=new_goal_grid)

def step(state: State, action):
    actions = (
        lambda: move(state, jnp.array((-1, 0))),
        lambda: move(state, jnp.array((1, 0))),
        lambda: move(state, jnp.array((0, 1))),
        lambda: move(state, jnp.array((0, -1))),
        lambda: grab_goal(state),
        lambda: pick_up_ring(state),
        lambda: release_goal(state)
    )
    new_state = jax.lax.switch(action, actions)
    new_state = new_state.replace(current_player=1-state.current_player, step_count=state.step_count+1)
    terminal, rewards = reward(new_state)
    legal_action_mask = legal_actions(new_state)
    return new_state, StepMetadata(rewards=rewards, action_mask=legal_action_mask, terminated=terminal, cur_player_id=new_state.current_player, step=new_state.step_count)

def observe(state: State):
    red_rings = state.rings[0].reshape((5, 5))
    blue_rings = state.rings[1].reshape((5, 5))
    red_scored = jnp.zeros((5, 5), dtype=jnp.int32)\
        .at[state.goals[0, 0], state.goals[0, 1]].set(state.goals[0, 2])\
        .at[state.goals[1, 0], state.goals[1, 1]].set(state.goals[1, 2])\
        .at[state.goals[2, 0], state.goals[2, 1]].set(state.goals[2, 2])
    blue_scored = jnp.zeros((5, 5), dtype=jnp.int32)\
        .at[state.goals[0, 0], state.goals[0, 1]].set(state.goals[0, 3])\
        .at[state.goals[1, 0], state.goals[1, 1]].set(state.goals[1, 3])\
        .at[state.goals[2, 0], state.goals[2, 1]].set(state.goals[2, 3])
    top_rings = jnp.zeros((5, 5), dtype=jnp.int32)\
        .at[state.goals[0, 0], state.goals[0, 1]].set(TOP_RING_MAP[state.goals[0, 4]])\
        .at[state.goals[1, 0], state.goals[1, 1]].set(TOP_RING_MAP[state.goals[1, 4]])\
        .at[state.goals[2, 0], state.goals[2, 1]].set(TOP_RING_MAP[state.goals[2, 4]])
    goals = jax.lax.min(state.goal_grid.reshape((5, 5)), jnp.ones((5, 5), dtype=jnp.int32))
    players = jnp.zeros((5, 5), dtype=jnp.int32)\
        .at[state.player_states[0, 0], state.player_states[0, 1]].set(1)\
        .at[state.player_states[1, 0], state.player_states[1, 1]].set(-1)
    return jax.lax.select(state.current_player == 0,
        jnp.stack((red_rings, blue_rings, red_scored, blue_scored, top_rings, goals, players)),
        jnp.stack((blue_rings, red_rings, blue_scored, red_scored, top_rings * -1, goals, players * -1)))

def init(key):
    init_state = State(
        current_player=jnp.int32(jax.random.bernoulli(key)),
        player_states=INIT_PLAYERS,
        rings=INIT_RINGS,
        goals=INIT_GOALS,
        goal_grid=INIT_GOAL_GRID
    )
    terminated, rewards = reward(init_state)
    legal_action_mask = legal_actions(init_state)
    return init_state, StepMetadata(rewards=rewards, action_mask=legal_action_mask, terminated=terminated, cur_player_id=init_state.current_player, step=init_state.step_count)
