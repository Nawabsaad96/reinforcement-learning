import numpy as np
import random

# Grid: 0=free, -1=obstacle, 1=dirt/cleaning reward
grid = np.array([
    [0,  0,  1],
    [0, -1,  0],
    [1,  0,  0]
])

A = [(0,1), (0,-1), (1,0), (-1,0)]   # actions: right, left, down, up
alpha, gamma, eps = 0.1, 0.9, 0.2    # SARSA parameters
Q = np.zeros((3,3,4))

def step(state, action):
    x, y = state
    dx, dy = action
    nx, ny = x + dx, y + dy

    # check boundary / obstacle
    if nx<0 or nx>2 or ny<0 or ny>2 or grid[nx][ny] == -1:
        return state, -2  # penalty for invalid move

    reward = grid[nx][ny]
    return (nx, ny), reward

def choose_action(state):
    return random.randint(0,3) if random.random() < eps else np.argmax(Q[state])

# training episodes
for ep in range(300):
    s = (0, 0)
    a = choose_action(s)

    while True:
        ns, r = step(s, A[a])
        na = choose_action(ns)

        Q[s][a] += alpha * (r + gamma * Q[ns][na] - Q[s][a])

        s, a = ns, na
        if r == 1:  # cleaned dirt
            break

# Final learned cleaning policy
policy = np.argmax(Q, axis=2)
print("Optimal Cleaning Policy Grid (0:R,1:L,2:D,3:U):")
print(policy)
