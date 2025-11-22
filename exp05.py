import numpy as np

# 3x3 city grid, 1 = pick-up point
grid = [[0,0,0],
        [0,1,0],
        [0,0,0]]

A = [(0,1),(0,-1),(1,0),(-1,0)]  # R,L,D,U
V = np.zeros((3,3))
g = 0.9

def step(r,c,a):
    nr, nc = r+A[a][0], c+A[a][1]
    if nr<0 or nc<0 or nr>2 or nc>2: return r,c,-1
    if grid[nr][nc]==1: return nr,nc,10  # reached passenger
    return nr,nc,-1

# -------- VALUE ITERATION --------
for _ in range(50):
    newV = np.zeros_like(V)
    for r in range(3):
        for c in range(3):
            qs = []
            for a in range(4):
                nr,nc,re = step(r,c,a)
                qs.append(re + g*V[nr][nc])
            newV[r][c] = max(qs)
    V = newV

# Extract optimal policy
P = np.zeros((3,3), dtype=int)
for r in range(3):
    for c in range(3):
        P[r][c] = np.argmax([
            step(r,c,a)[2] + g*V[step(r,c,a)[0], step(r,c,a)[1]]
            for a in range(4)
        ])

print("Optimal Policy (0=R,1=L,2=D,3=U):")
print(P)
print("Value Function:")
print(V)
