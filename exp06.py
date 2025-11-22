import random, math

# True click-through rates of ads
ctr = [0.10, 0.15, 0.30]   # Ad C is best

def reward(a):
    return 1 if random.random() < ctr[a] else 0

# ---------- EPSILON-GREEDY ----------
def eps_greedy(eps=0.1, T=5000):
    n = len(ctr)
    v = [0]*n; c = [0]*n; rsum = 0
    for _ in range(T):
        a = random.randint(0,n-1) if random.random()<eps else v.index(max(v))
        r = reward(a); rsum += r
        c[a]+=1; v[a]+= (r - v[a])/c[a]
    return rsum

# ---------- UCB ----------
def ucb(T=5000):
    n=len(ctr); c=[1]*n; v=[reward(i) for i in range(n)]; rsum=sum(v)
    for t in range(n, T):
        u=[v[i]+math.sqrt(2*math.log(t+1)/c[i]) for i in range(n)]
        a=u.index(max(u)); r=reward(a); rsum+=r
        c[a]+=1; v[a]+= (r - v[a])/c[a]
    return rsum

# ---------- THOMPSON ----------
def thompson(T=5000):
    n=len(ctr); s=[1]*n; f=[1]*n; rsum=0
    for _ in range(T):
        a=max(range(n), key=lambda i: random.betavariate(s[i],f[i]))
        r=reward(a); rsum+=r
        if r: s[a]+=1
        else: f[a]+=1
    return rsum

# Run comparison
eg = eps_greedy()
uc = ucb()
ts = thompson()

print("Clicks (5000 users)")
print("Epsilon-Greedy:", eg)
print("UCB:", uc)
print("Thompson:", ts)
print("Best:", max([("EG",eg),("UCB",uc),("TS",ts)], key=lambda x:x[1])[0])
