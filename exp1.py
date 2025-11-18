"""An autonomous cleaning robot navigates a 5x5 grid where certain cells contain dirt (reward:
+1) and obstacles (penalty: -1). The robot starts at the top-left corner and must find an
optimal policy to clean the entire grid efficiently. Implement the grid environment as an
MDP and write a Python program to simulate the robot’s navigation using different policies."""



import random

# 5x5 grid (1 = dirt, -1 = obstacle, 0 = empty)
grid = [
    [0, 0, 1, -1, 0],
    [0, -1, 0, 0, 1],
    [1, 0, 0, -1, 0],
    [0, 1, -1, 0, 0],
    [0, 0, 0, 1, -1]
]

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Starting position
state = (0, 0)

def step(state, action):
    i, j = state

    if action == "UP":
        ni, nj = i - 1, j
    elif action == "DOWN":
        ni, nj = i + 1, j
    elif action == "LEFT":
        ni, nj = i, j - 1
    else:
        ni, nj = i, j + 1

    # stay inside grid
    if ni < 0 or ni > 4 or nj < 0 or nj > 4:
        return state, 0

    reward = grid[ni][nj]
    return (ni, nj), reward

# --- SIMPLE RANDOM POLICY FOR PRACTICAL LEARNING ---
print("Robot Navigation using Random Policy\n")

total_reward = 0

for step_no in range(15):  # 15 steps
    action = random.choice(ACTIONS)
    next_state, reward = step(state, action)

    print(f"Step {step_no+1}: {state} --{action}--> {next_state}, Reward = {reward}")

    state = next_state
    total_reward += reward

print("\nTotal Reward Collected =", total_reward)
