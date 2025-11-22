import numpy as np

grid = [[0,0,0],
        [0,1,0],
        [0,0,0]]

A = [(0,1),(0,-1),(1,0),(-1,0)]   # R,L,D,U
V = np.zeros((3,3))
P = np.zeros((3,3), dtype=int)
g = 0.9

def step(r,c,a):
    nr, nc = r+A[a][0], c+A[a][1]
    if nr<0 or nc<0 or nr>2 or nc>2: return r,c,-1
    if grid[nr][nc]==1: return nr,nc,10
    return nr,nc,-1

while True:
    # Policy Evaluation
    for _ in range(10):
        for r in range(3):
            for c in range(3):
                a = P[r][c]
                nr,nc,rw = step(r,c,a)
                V[r][c] = rw + g*V[nr][nc]

    # Policy Improvement
    stable = True
    for r in range(3):
        for c in range(3):
            vals = [step(r,c,a)[2] + g*V[step(r,c,a)[0], step(r,c,a)[1]]
                    for a in range(4)]
            best = np.argmax(vals)
            if best != P[r][c]:
                P[r][c] = best
                stable = False

    if stable: break

print("Optimal Policy (0=R,1=L,2=D,3=U):")
print(P)
print("Value:")
print(V)
