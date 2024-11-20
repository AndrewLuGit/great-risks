from complex_game_def import *
import jax

state, meta = init(jax.random.key(0))
while True:
    print(observe(state))
    print(meta.action_mask)
    print(meta.rewards)
    move = input()
    state, meta = step(state, int(move))