"""A robot navigates a warehouse to pick and place items. Define states (locations in the
warehouse), actions (move in four directions), and rewards (picking an item: +2, reaching
the goal: +5, hitting an obstacle: -2). Implement a policy evaluation algorithm to determine
the value function for a given policy in Python."""

import numpy as np

# ----- Warehouse Grid (5x5) -----
# 2 = item (+2)
# 5 = goal (+5)
# -2 = obstacle (-2)
# 0 = empty
warehouse = np.array([
    [0,  0,  2,  0,  5],
    [0, -2,  0,  2,  0],
    [0,  0,  0, -2,  0],
    [2,  0, -2,  0,  0],
    [0,  0,  0,  2, -2]
])

ROWS, COLS = 5, 5

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# ---- Transition function ----
def step(state, action):
    r, c = state

    if action == "UP":    nr, nc = r - 1, c
    if action == "DOWN":  nr, nc = r + 1, c
    if action == "LEFT":  nr, nc = r, c - 1
    if action == "RIGHT": nr, nc = r, c + 1

    # Stay inside grid
    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
        return state, 0

    reward = warehouse[nr][nc]
    return (nr, nc), reward


# ---- FIXED POLICY (Example): always move RIGHT ----
def policy(state):
    return "RIGHT"


# ---- POLICY EVALUATION ----
def policy_evaluation(gamma=0.9, threshold=1e-4):
    V = np.zeros((ROWS, COLS))  # initialize value function

    while True:
        delta = 0
        for r in range(ROWS):
            for c in range(COLS):
                action = policy((r, c))     # get action from fixed policy
                next_state, reward = step((r, c), action)

                # Bellman expectation update
                new_value = reward + gamma * V[next_state]

                delta = max(delta, abs(new_value - V[r][c]))
                V[r][c] = new_value

        if delta < threshold:
            break

    return V


# ---- Run Policy Evaluation ----
values = policy_evaluation()
print("Value Function for the Given Policy:\n")
print(values)
