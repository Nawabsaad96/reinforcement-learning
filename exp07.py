import numpy as np
import matplotlib.pyplot as plt

# Warehouse Grid (1 = delivery point)
grid = np.array([
    [0,0,0],
    [0,1,0],
    [0,0,0]
])

actions = [(0,1),(0,-1),(1,0),(-1,0)]  # R, L, D, U
gamma = 0.9

def step(r,c,a):
    nr, nc = r+a[0], c+a[1]
    if nr<0 or nr>2 or nc<0 or nc>2:
        return r,c,-1
    if grid[nr][nc] == 1:
        return nr,nc,10
    return nr,nc,-1

# --------- Bellman Expectation Backup ---------
def bellman_value(policy, iterations=30):
    V = np.zeros((3,3))
    for _ in range(iterations):
        newV = np.zeros_like(V)
        for r in range(3):
            for c in range(3):
                a = policy[r][c]
                nr,nc,re = step(r,c,actions[a])
                newV[r][c] = re + gamma * V[nr][nc]
        V = newV
    return V

# --------- Policies ----------
random_policy = np.random.randint(0,4,(3,3))
right_policy  = np.zeros((3,3), dtype=int)
goal_policy   = np.array([
    [2,2,2],
    [1,0,1],
    [3,3,3]
])  # simple greedy moves to center

# Compute Value Functions
V_random = bellman_value(random_policy)
V_right  = bellman_value(right_policy)
V_goal   = bellman_value(goal_policy)

# -------- Visualization --------
titles = ["Random Policy", "Move-Right Policy", "Greedy-to-Goal Policy"]
values = [V_random, V_right, V_goal]

plt.figure(figsize=(9,3))
for i, V in enumerate(values):
    plt.subplot(1,3,i+1)
    plt.imshow(V, cmap='viridis')
    plt.title(titles[i])
    plt.colorbar()
plt.tight_layout()
plt.show()
