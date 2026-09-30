"""Min T-count of 2-qutrit DIAGONAL phase functions zeta^{f(x,y)} built from single-qutrit diagonal
C+T gadgets on the 4 linear forms x, y, x+y, x+2y (each gadget diag(1,z^a,z^b), a+b=0 mod 3,
costs 1 T unless a,b = 0 mod 3).  f is taken modulo a global constant."""
import itertools, numpy as np
forms = [(1, 0), (0, 1), (1, 1), (1, 2)]
gad = [(a, b) for a in range(9) for b in range(9) if (a+b) % 3 == 0]
def cost(g): return 0 if g[0] % 3 == 0 and g[1] % 3 == 0 else 1
pts = [(x, y) for x in range(3) for y in range(3)]
# table: vector over 9 points for each (form, gadget)
vec = {}
for fi, f in enumerate(forms):
    for g in gad:
        vec[(fi, g)] = np.array([(0, g[0], g[1])[(f[0]*x+f[1]*y) % 3] for x, y in pts])
def mincost(target):
    t = np.array([target(x, y) for x, y in pts]) % 9
    best = None
    for gs in itertools.product(gad, repeat=4):
        c = sum(cost(g) for g in gs)
        if best is not None and c >= best[0]: continue
        tot = sum(vec[(fi, g)] for fi, g in enumerate(gs)) % 9
        d = (tot - t) % 9
        if np.all(d == d[0]): best = (c, gs)
    return best
d2 = lambda v: 1 if v == 2 else 0
print("C2(Z)  3*[x=2]*y        :", mincost(lambda x, y: 3*d2(x)*y))
print("C2(S') 3*[x=2][y=2]      :", mincost(lambda x, y: 3*d2(x)*d2(y)))
print("CZ-type 3*x*y            :", mincost(lambda x, y: 3*x*y))
print("3*[x=2]*[y=1]            :", mincost(lambda x, y: 3*d2(x)*(1 if y == 1 else 0)))
