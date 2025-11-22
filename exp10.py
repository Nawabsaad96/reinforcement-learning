import numpy as np, random

# Maze: 0=free, 1=goal, -1=trap
maze = np.array([
    [0,  0,  0],
    [0, -1,  1],
    [0,  0,  0]
])

A = [(0,1),(0,-1),(1,0),(-1,0)]   # R,L,D,U
gamma = 0.9
alpha = 0.1

# State-value function
V = {(r,c): 0 for r in range(3) for c in range(3)}

def step(s,a):
    r,c = s
    nr, nc = r+A[a][0], c+A[a][1]

    # wall check
    if nr<0 or nr>2 or nc<0 or nc>2:
        return s, -1

    # rewards
    if maze[nr][nc] == 1: return (nr,nc), 10
    if maze[nr][nc] == -1: return (nr,nc), -10
    return (nr,nc), -1

# -------- TD(0) LEARNING --------
for _ in range(500):
    s = (random.randint(0,2), random.randint(0,2))
    while maze[s] == 0:          # stop if trap/goal
        a = random.randint(0,3)  # epsilon-greedy simple
        s2, r = step(s,a)
        V[s] += alpha*(r + gamma*V[s2] - V[s])     # TD(0) update
        s = s2

# ---------- DISPLAY ----------
print("State Values:")
for r in range(3):
    print([round(V[(r,c)],2) for c in range(3)])
