import numpy as np
import random

# 0=empty, 1=food, -1=ghost
grid = np.array([
    [0,  0,  1],
    [0, -1,  0],
    [0,  0,  0]
])

A = [(0,1),(0,-1),(1,0),(-1,0)]   # R,L,D,U
alpha, gamma, eps = 0.1, 0.9, 0.1
Q = np.zeros((3,3,4))

def step(s, a):
    x, y = s
    dx, dy = a
    nx, ny = x + dx, y + dy

    # boundaries
    if nx<0 or nx>2 or ny<0 or ny>2:
        return s, -2  # wall penalty

    reward = 1 if grid[nx,ny] == 1 else -5 if grid[nx,ny] == -1 else 0
    done = (reward == 1 or reward == -5)

    return (nx, ny), reward, done

def choose_action(s):
    return random.randint(0,3) if random.random() < eps else np.argmax(Q[s])

# -------- TRAINING --------
for ep in range(400):
    s = (2,0)  # Pac-Man start point
    while True:
        a = choose_action(s)
        ns, r, done = step(s, A[a])

        Q[s][a] += alpha * (r + gamma*np.max(Q[ns]) - Q[s][a])
        s = ns
        if done:
            break

# -------- EVALUATE --------
s = (2,0)
path = [s]
for _ in range(10):
    a = np.argmax(Q[s])
    s, r, done = step(s, A[a])
    path.append(s)
    if done:
        break

print("Learned Path:", path)
print("Q-Table:")
print(np.round(Q,2))
