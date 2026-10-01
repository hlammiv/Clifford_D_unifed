"""verify_meas.py -- STANDALONE verifier for measurement + Clifford-feed-forward gadgets on (system, ancilla).
Usage: python3 verify_meas.py gadget.json
JSON: gates (time order; names H,S,X,Z,T and *dg on qutrit 0/1, SUM01: |a,b>->|a,b+a>, SUM10: |a,b>->|a+b,b>),
      'target' diagonal exponents, measurement = ancilla in computational basis (or Fourier basis if
      "measure_basis": "fourier"), correction_exponents[m] = c with correction diag(w^c0, w^c1, w^c2).
Checks (1) EXACT arithmetic in Q(zeta_9) (Fractions; zeta^6 = -zeta^3 - 1): for every outcome m the
block (I (x) <m|) U (I (x) |0>) equals lambda_m * D_m^dag * G with D_m the correction, G the target;
(2) numerically on 300 random input states: probabilities and post-measurement states."""
import sys, json, itertools, numpy as np
from fractions import Fraction as Fr

# ---------- exact Q(zeta_9) arithmetic: element = tuple of 6 Fractions (coefficients of 1..zeta^5) ----------
def add(a, b): return tuple(x+y for x, y in zip(a, b))
def mul(a, b):
    c = [Fr(0)]*11
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y: c[i+j] += x*y
    for k in range(10, 5, -1):                       # zeta^k = -zeta^{k-3} - zeta^{k-6}
        if c[k]: c[k-3] -= c[k]; c[k-6] -= c[k]; c[k] = Fr(0)
    return tuple(c[:6])
ZERO = (Fr(0),)*6; ONE = (Fr(1),) + (Fr(0),)*5
def zpow(k):
    k %= 9; e = ONE; zt = (Fr(0), Fr(1)) + (Fr(0),)*4
    for _ in range(k): e = mul(e, zt)
    return e
def scal(a, f): return tuple(x*f for x in a)
INV_SQRTM3 = scal(add(ONE, scal(zpow(3), 2)), Fr(-1, 3))   # 1/sqrt(-3) = (1+2w)/(-3)
def mat(M): return [[M[i][j] for j in range(len(M[0]))] for i in range(len(M))]
def mmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    return [[ (lambda s: s)(sum_( [mul(A[i][t], B[t][j]) for t in range(k) if A[i][t] != ZERO and B[t][j] != ZERO] ))
              for j in range(m)] for i in range(n)]
def sum_(lst):
    s = ZERO
    for x in lst: s = add(s, x)
    return s
def kron(A, B):
    return [[mul(A[i//len(B)][j//len(B)], B[i % len(B)][j % len(B)]) for j in range(len(A)*len(B))] for i in range(len(A)*len(B))]
def diagm(es): return [[(zpow(es[i]) if i == j else ZERO) for j in range(3)] for i in range(3)]
I3x = diagm([0, 0, 0])
G1X = {'H': [[scal(mul(INV_SQRTM3, zpow(3*j*k)), 1) for k in range(3)] for j in range(3)],
       'S': diagm([0, 0, 3]), 'Z': diagm([0, 3, 6]), 'T': diagm([0, 1, 8]),
       'X': [[ONE if i == (j+1) % 3 else ZERO for j in range(3)] for i in range(3)]}
def conjT(A):
    def cj(a):  # complex conjugation zeta -> zeta^-1 = zeta^8
        out = ZERO
        for i, x in enumerate(a):
            if x: out = add(out, scal(zpow(-i), x))
        return out
    return [[cj(A[j][i]) for j in range(len(A))] for i in range(len(A[0]))]
for k in list(G1X): G1X[k+'dg'] = conjT(G1X[k])
def SUMx(c):
    M = [[ZERO]*9 for _ in range(9)]
    for a, b in itertools.product(range(3), repeat=2):
        y = (a, (a+b) % 3) if c == 0 else ((a+b) % 3, b)
        M[3*y[0]+y[1]][3*a+b] = ONE
    return M
G2X = {'SUM01': SUMx(0), 'SUM10': SUMx(1)}; G2X['SUM01dg'] = conjT(G2X['SUM01']); G2X['SUM10dg'] = conjT(G2X['SUM10'])
def gate_exact(g):
    if g in G2X: return G2X[g]
    return kron(G1X[g[:-1]], I3x) if g[-1] == '0' else kron(I3x, G1X[g[:-1]])
def tonum(a): return sum(complex(float(x))*np.exp(2j*np.pi*i/9) for i, x in enumerate(a))

spec = json.load(open(sys.argv[1]))
gl = spec['gates']; tgt = spec['target_exponents']           # target = diag(zeta^{e} or -1 handled below)
Tcount = sum(1 for g in gl if g[:-1] in ('T', 'Tdg'))
# exact unitary applied to the isometry I (x) |0>
E = [[(ONE if (i == 3*j) else ZERO) for j in range(3)] for i in range(9)]
V = E
for g in gl: V = mmul(gate_exact(g), V)
fourier = spec.get('measure_basis', 'computational') == 'fourier'
Gt = [[(spec_val := None) or ZERO for _ in range(3)] for _ in range(3)]
for x in range(3): Gt[x][x] = scal(ONE, -1) if tgt[x] == 'neg' else zpow(int(tgt[x]))
print(f"gadget: {spec.get('name', sys.argv[1])}\n  gates ({len(gl)}, T-count {Tcount}): {' '.join(gl)}")
okall = True
for m in range(3):
    if fourier:   # <f_m| = (1/sqrt3) sum_a w^{-m a} <a|  ; up to the scalar 1/sqrt3, which is irrelevant
        B = [[sum_([mul(zpow(-3*m*a), V[3*x+a][j]) for a in range(3)]) for j in range(3)] for x in range(3)]
    else:
        B = [[V[3*x+m][j] for j in range(3)] for x in range(3)]
    c = spec['correction_exponents'][str(m)]
    DB = mmul(diagm([3*k for k in c]), B)                       # corrected block D_m B_m
    # find lambda = DB[0][0] / Gt[0][0] ; check DB == lambda * Gt exactly  (Gt diagonal, entries units)
    lam = mul(DB[0][0], conjT([[Gt[0][0]]])[0][0])
    ok = all(DB[i][j] == (mul(lam, Gt[i][j]) if i == j else ZERO) for i in range(3) for j in range(3))
    okall &= ok
    print(f"  outcome m={m}: correction diag(w^{c}) ; EXACT block == lambda*target : {ok} ; |lambda|^2 = {abs(tonum(lam))**2/(3 if fourier else 1):.6f}")
# numeric check on random states
Vn = np.array([[tonum(V[i][j]) for j in range(3)] for i in range(9)])
Gn = np.array([[tonum(Gt[i][j]) for j in range(3)] for i in range(3)])
rng = np.random.default_rng(2026); worst = 0; probs = np.zeros(3)
F = np.array([[np.exp(2j*np.pi*a*m/3) for m in range(3)] for a in range(3)])/np.sqrt(3)
for _ in range(300):
    psi = rng.normal(size=3)+1j*rng.normal(size=3); psi /= np.linalg.norm(psi)
    out = Vn @ psi
    for m in range(3):
        st = np.array([np.vdot(F[:, m], out[3*x:3*x+3]) if fourier else out[3*x+m] for x in range(3)])
        p = np.vdot(st, st).real; probs[m] = p
        st = np.diag([np.exp(2j*np.pi*k/3) for k in spec['correction_exponents'][str(m)]]) @ st / np.sqrt(p)
        t = Gn @ psi; worst = max(worst, np.linalg.norm(st - np.vdot(t, st)*t))
print(f"  numeric: outcome probabilities {np.round(probs, 6)} (state independent), max residual over 300 states x 3 outcomes {worst:.1e}")
print("  RESULT:", "PASS (deterministic, Clifford feed-forward)" if okall and worst < 1e-10 else "FAIL")
