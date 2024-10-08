import subprocess
import json
import sys
from tkinter import *

if __name__ == "__main__":
    sub = subprocess.Popen(["build/agent_game"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
    root = Tk()
    label = Label(root)
    label.grid(column=0, row=0)
    label["justify"] = "left"
    canvas = Canvas(root, width=497, height=497)
    canvas.grid(column=1, row=0)
    state = json.loads(sub.stdout.readline())
    for i in range(1, 500, 45):
        canvas.create_line([(i, 1), (i, 496)])
        canvas.create_line([(1, i), (496, i)])
    def render(state):
        canvas.delete("elements")
        for i in range(11):
            for j in range(11):
                red_rings = state["red_rings"][i][j]
                blue_rings = state["blue_rings"][i][j]
                if red_rings > 0:
                    canvas.create_text(12 + 45 * j, 23 + 45 * i, text=str(red_rings), tags="elements", fill="red")
                if blue_rings > 0:
                    canvas.create_text(34 + 45 * j, 23 + 45 * i, text=str(blue_rings), tags="elements", fill="blue")
        for robot in state["robots"]:
            canvas.create_text(23 + 45 * robot["y"], 23 + 45 * robot["x"], text="R", tags="elements", fill="red" if robot["is_red"] else "blue")
        label_text = ""
        newline = '\n'
        for i, goal in enumerate(state["goals"]):
            label_text += f"Goal {i}: {json.dumps(goal)}{newline}"
            if (goal["x"] != -1):
                canvas.create_text(23 + 45 * goal["y"], 23 + 45 * goal["x"], text="G", tags="elements")
        for i, stake in enumerate(state["stakes"]):
            label_text += f"Stake {i}: {json.dumps(stake)}{newline}"
        label_text += f"Robot: {json.dumps(state["robots"][0])}{newline}"
        label_text += f"Red Score: {state["scores"]["red"]}, Blue Score: {state["scores"]["blue"]}{newline}"
        label_text += f"Time Remaining: {state["time_remaining"]}"
        label["text"] = label_text
    def update_field(event):
        action = 0
        match event.keysym:
            case "Up":
                action = 0
            case "Down":
                action = 1
            case "Right":
                action = 2
            case "Left":
                action = 3
            case "a":
                action = 4
            case "s":
                action = 5
            case "d":
                action = 6
            case "f":
                action = 7
            case "g":
                action = 8
            case "h":
                action = 9
            case "j":
                action = 10
            case "k":
                action = 11
            case "l":
                action = 12
            case "z":
                action = 13
            case "x":
                action = 14
        message = f'{{"action":{action}}}'
        sub.stdin.write(message.encode())
        sub.stdin.flush()
        state = json.loads(sub.stdout.readline())
        render(state)
    render(state)
    root.bind('<KeyPress>', update_field)
    root.mainloop()