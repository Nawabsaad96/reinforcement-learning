import numpy as np, random

# 3x3 city grid, (1,1) = delivery point
goal = (1,1)
A = [(0,1),(0,-1),(1,0),(-1,0)]   # right, left, down, up
gamma = 0.9

def step(s,a):
    r,c = s
    nr, nc = r+A[a][0], c+A[a][1]
    if nr<0 or nr>2 or nc<0 or nc>2:
        return s, -1
    if (nr,nc)==goal:
        return (nr,nc), 20
    return (nr,nc), -1

# Initialize Q and policy
Q = {(r,c): [0]*4 for r in range(3) for c in range(3)}
pi = {(r,c): random.randint(0,3) for r in range(3) for c in range(3)}
returns = {(r,c): [[] for _ in range(4)] for r in range(3) for c in range(3)}

# ------------- MONTE CARLO CONTROL -------------
for eps in range(500):
    s = (random.randint(0,2), random.randint(0,2))
    episode = []

    # generate episode
    while True:
        a = random.randint(0,3) if random.random()<0.2 else pi[s]
        s2, r = step(s,a)
        episode.append((s,a,r))
        s = s2
        if s == goal or len(episode) > 20: break

    # return calculation + Q update
    G = 0
    for s,a,r in reversed(episode):
        G = r + gamma*G
        returns[s][a].append(G)
        Q[s][a] = np.mean(returns[s][a])
        pi[s] = np.argmax(Q[s])

# -------- OUTPUT ----------
print("Optimal Drone Policy (0=R,1=L,2=D,3=U):")
for r in range(3):
    print([pi[(r,c)] for c in range(3)])
