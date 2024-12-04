from complex_game_def import *
from PIL import Image, ImageDraw
import jax

state, meta = init(jax.random.key(0))
image = Image.new("RGB", (227, 247), "white")
draw = ImageDraw.Draw(image)
for i in range(1, 227, 45):
    draw.line([(i, 1), (i, 226)], "black")
    draw.line([(1, i), (226, i)], "black")
for i in range(5):
    for j in range(5):
        red_rings = state.rings[0, 5 * i + j]
        blue_rings = state.rings[1, 5 * i + j]
        if red_rings > 0:
            draw.text((12 + 45 * j, 23 + 45 * i), text=str(red_rings), fill="red")
        if blue_rings > 0:
            draw.text((34 + 45 * j, 23 + 45 * i), text=str(blue_rings), fill="blue")
draw.text((23 + 45 * state.player_states[0, 1], 23 + 45 * state.player_states[0, 0]), text="R", fill="red")
draw.text((23 + 45 * state.player_states[1, 1], 23 + 45 * state.player_states[1, 0]), text="R", fill="blue")
for i in range(3):
    draw.text((23 + 45 * state.goals[i, 1], 23 + 45 * state.goals[i, 0]), text="G", fill="black")

player_scores = scores(state)
draw.text((113, 237), text=f"step: {state.step_count} red: {player_scores[0]} blue: {player_scores[1]}", fill="black", anchor="mm")

image.show()