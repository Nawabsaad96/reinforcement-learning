import numpy as np

# States = queue sizes 0–3 cars on each road
S = [(i,j) for i in range(4) for j in range(4)]
A = [0,1]   # 0 = NS green, 1 = EW green
g = 0.9

def step(s,a):
    ns, ew = s
    # cars arrive randomly (0 or 1)
    ns += np.random.randint(2)
    ew += np.random.randint(2)

    # green direction moves 1 car
    if a==0 and ns>0: ns -= 1
    if a==1 and ew>0: ew -= 1

    ns = min(ns,3); ew = min(ew,3)
    r = -(ns + ew)          # minimize waiting cars
    return (ns,ew), r

# Initialize
V = {s:0 for s in S}
pi = {s:0 for s in S}

# -------- POLICY ITERATION --------
def policy_iteration():
    global V, pi

    while True:
        # ---- Policy Evaluation ----
        for _ in range(20):
            V_new = {}
            for s in S:
                a = pi[s]
                s2, r = step(s,a)
                V_new[s] = r + g*V[s2]
            V = V_new

        # ---- Policy Improvement ----
        stable = True
        for s in S:
            Q = []
            for a in A:
                s2,r = step(s,a)
                Q.append(r + g*V[s2])
            best = np.argmax(Q)
            if best != pi[s]:
                pi[s] = best
                stable = False

        if stable: break

policy_iteration()

print("Optimal Policy (0=NS green, 1=EW green):")
for s in sorted(pi.keys()):
    print(s, "->", pi[s])
