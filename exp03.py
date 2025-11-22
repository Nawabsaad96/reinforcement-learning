import random, math

prices = [50, 70, 100]
probs  = [0.25, 0.18, 0.10]   # true buy probability

def reward(arm):  # revenue = price if bought else 0
    return prices[arm] if random.random() < probs[arm] else 0

# ---------------- EPSILON-GREEDY ----------------
def eps_greedy(eps=0.1, T=5000):
    n = len(prices)
    counts = [0]*n
    values = [0]*n
    total = 0
    for _ in range(T):
        arm = random.randint(0,n-1) if random.random()<eps else values.index(max(values))
        r = reward(arm); total += r
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]
    return total

# -------------------- UCB1 -----------------------
def ucb(T=5000):
    n = len(prices)
    counts = [1]*n
    values = [reward(i) for i in range(n)]
    total = sum(values)
    for t in range(n, T):
        u = [values[i] + math.sqrt(2*math.log(t+1)/counts[i]) for i in range(n)]
        arm = u.index(max(u))
        r = reward(arm); total += r
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]
    return total

# -------------- THOMPSON SAMPLING ----------------
def thompson(T=5000):
    n = len(prices)
    s = [1]*n; f = [1]*n
    total = 0
    for _ in range(T):
        arm = max(range(n), key=lambda i: random.betavariate(s[i], f[i]))
        r = reward(arm); total += r
        if r>0: s[arm]+=1
        else:   f[arm]+=1
    return total

# Run
print("Epsilon-Greedy:", eps_greedy())
print("UCB:", ucb())
print("Thompson:", thompson())
