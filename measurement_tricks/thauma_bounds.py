"""thauma_bounds.py -- max-thauma / min-thauma (Wang-Wilde-Su, PRL 124, 090505 (2020); NJP 21 103002) lower bounds.
theta_max(rho) = log min{ ||V||_{W,1} : V >= rho }  (SDP), theta_min(psi) = -log max{ <psi|V|psi> : ||V||_{W,1} <= 1, V >= 0 }.
Both are monotone under stabilizer operations and additive on tensor products, so (catalyst cancels)
    t >= theta(Choi(G)) / theta(T|+>)   for deterministic gadgets with measurements / feed-forward / catalysts."""
import numpy as np, itertools, cvxpy as cp
z = np.exp(2j*np.pi/9); w = z**3
X = np.roll(np.eye(3), 1, axis=0); Zm = np.diag([1, w, w*w]); mp = np.linalg.matrix_power
P0 = np.zeros((3, 3))
for j in range(3): P0[(-j) % 3, j] = 1
def D(a1, a2): return w**(2*a1*a2) * mp(Zm, a1) @ mp(X, a2)
A1 = [D(a1, a2) @ P0 @ D(a1, a2).conj().T for a1 in range(3) for a2 in range(3)]
def Aops(n):
    out = []
    for pts in itertools.product(A1, repeat=n):
        Op = np.array([[1]])
        for p in pts: Op = np.kron(Op, p)
        out.append(Op / 3**n)
    return out
def theta_max(psi, n):
    psi = np.asarray(psi, complex); psi /= np.linalg.norm(psi); d = 3**n; rho = np.outer(psi, psi.conj())
    V = cp.Variable((d, d), hermitian=True)
    Wv = [cp.real(cp.trace(Au @ V)) for Au in Aops(n)]
    prob = cp.Problem(cp.Minimize(sum(cp.abs(x) for x in Wv)), [V - rho >> 0])
    prob.solve(solver='SCS', eps=1e-9, max_iters=200000)
    return np.log(prob.value)
def theta_min(psi, n):
    psi = np.asarray(psi, complex); psi /= np.linalg.norm(psi); d = 3**n; rho = np.outer(psi, psi.conj())
    V = cp.Variable((d, d), hermitian=True)
    Wv = [cp.real(cp.trace(Au @ V)) for Au in Aops(n)]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(rho @ V))), [sum(cp.abs(x) for x in Wv) <= 1, V >> 0])
    prob.solve(solver='SCS', eps=1e-9, max_iters=200000)
    return -np.log(prob.value)
T = [1, z, z**8]
def choi(dg):
    c = np.zeros(9, complex)
    for x in range(3): c[3*x+x] = dg[x]
    return c
for nm, f in (('max-thauma', theta_max), ('min-thauma', theta_min)):
    tT = f(T, 1)
    print(f"{nm}: theta(T|+>) = {tT:.5f}")
    for name, dg in (('R', [1, 1, -1]), ('L', [1, 1, z]), ('T (sanity)', [1, z, z**8])):
        tc = f(choi(dg), 2)
        print(f"   {name:11s} theta(Choi) = {tc:.5f}  ->  t >= {tc/tT:.4f}")
